$ErrorActionPreference = "Stop"

$python = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"
$trainer = Join-Path $PSScriptRoot "train_accuracy_oriented_dense.py"
$evaluator = Join-Path $PSScriptRoot "evaluate_accuracy_oriented_dense.py"
$dataPath = Join-Path $PSScriptRoot "..\cifar_data"
$sourceCheckpoint = Join-Path $PSScriptRoot "accuracy_oriented_results_v3\cifar10\gradual_refinement\seed_42\best_checkpoint.pth"
$outputDirectory = Join-Path $PSScriptRoot "accuracy_oriented_results_v6\cifar10\light_regularized_refinement\seed_42"
$lastCheckpoint = Join-Path $outputDirectory "last_checkpoint.pth"
$bestCheckpoint = Join-Path $outputDirectory "best_checkpoint.pth"
$trainingSummary = Join-Path $outputDirectory "training_summary.json"
$evaluationDirectory = Join-Path $outputDirectory "evaluation_tta"

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Python environment not found: $python"
}
if (-not (Test-Path -LiteralPath $sourceCheckpoint -PathType Leaf)) {
    throw "V3 best checkpoint not found: $sourceCheckpoint"
}

$activeTrainers = @(
    Get-CimInstance Win32_Process | Where-Object {
        $_.Name -match "python" -and
        $_.CommandLine -match "train_accuracy_oriented_dense.py"
    }
)
if ($activeTrainers.Count -gt 0) {
    $activePids = ($activeTrainers | ForEach-Object { $_.ProcessId }) -join ", "
    throw "Another accuracy-oriented trainer is active (PID: $activePids)."
}

$complete = $false
if (Test-Path -LiteralPath $trainingSummary -PathType Leaf) {
    $summary = Get-Content -LiteralPath $trainingSummary -Raw | ConvertFrom-Json
    $complete = ($summary.early_stopped -eq $true -or [int]$summary.last_epoch -ge 19)
}

$arguments = @(
    $trainer,
    "--dataset", "cifar10",
    "--data_path", $dataPath,
    "--output_dir", $outputDirectory,
    "--seed", "42",
    "--split_seed", "2026",
    "--download", "false",
    "--epochs", "20",
    "--batch_size", "128",
    "--num_workers", "4",
    "--lr_backbone", "2e-6",
    "--lr_classifier", "1e-5",
    "--min_lr", "5e-7",
    "--warmup_epochs", "2",
    "--weight_decay", "0.01",
    "--label_smoothing", "0.02",
    "--mixup_alpha", "0.02",
    "--cutmix_alpha", "0.1",
    "--randaugment_num_ops", "2",
    "--randaugment_magnitude", "2",
    "--random_erasing", "0",
    "--drop_path", "0.01",
    "--augmentation_schedule", "static",
    "--ema_decay", "0.999",
    "--early_stopping_patience", "8",
    "--amp", "true",
    "--device", "cuda"
)

if ($complete) {
    Write-Host "CIFAR-10 light-regularized refinement is already complete."
}
elseif (Test-Path -LiteralPath $lastCheckpoint -PathType Leaf) {
    Write-Host "Resuming CIFAR-10 light-regularized refinement."
    $arguments += @("--resume", $lastCheckpoint)
    & $python @arguments
    if ($LASTEXITCODE -ne 0) { throw "Light-regularized refinement resume failed" }
}
else {
    Write-Host "Starting from the v3 best EMA checkpoint with persistent light augmentation."
    $arguments += @("--refinement_checkpoint", $sourceCheckpoint)
    & $python @arguments
    if ($LASTEXITCODE -ne 0) { throw "Light-regularized refinement failed" }
}

if (-not (Test-Path -LiteralPath $bestCheckpoint -PathType Leaf)) {
    throw "Best checkpoint unavailable: $bestCheckpoint"
}

& $python $evaluator `
    --checkpoint $bestCheckpoint `
    --dataset cifar10 `
    --data_path $dataPath `
    --output_dir $evaluationDirectory `
    --batch_size 128 `
    --num_workers 4 `
    --device cuda
if ($LASTEXITCODE -ne 0) { throw "TTA evaluation failed" }

$report = Get-Content -LiteralPath (Join-Path $evaluationDirectory "tta_evaluation.json") -Raw | ConvertFrom-Json
$standard = $report.results | Where-Object { $_.mode -eq "none" }
$tta = $report.results | Where-Object { $_.mode -eq "flip_shift" }
$ttaAccuracy = [double]$tta.accuracy

Write-Host ("Single-view test accuracy: {0:N2}%" -f [double]$standard.accuracy)
Write-Host ("10-view TTA test accuracy: {0:N2}%" -f $ttaAccuracy)
if ($ttaAccuracy -gt 96.0) {
    Write-Host "Success: refinement exceeded 96% TTA accuracy."
}
elseif ($ttaAccuracy -ge 95.5) {
    Write-Host "Promising: repeat the recipe for the remaining seeds."
}
else {
    Write-Warning "Below 95.5%: do not continue reducing augmentation."
}
