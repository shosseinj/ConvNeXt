$ErrorActionPreference = "Stop"

$python = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"
$trainer = Join-Path $PSScriptRoot "train_accuracy_oriented_dense.py"
$evaluator = Join-Path $PSScriptRoot "evaluate_accuracy_oriented_dense.py"
$dataPath = Join-Path $PSScriptRoot "..\cifar_data"
$sourceCheckpoint = Join-Path $PSScriptRoot "accuracy_oriented_results_v2\cifar10\fully_dense_refinement\seed_42\best_checkpoint.pth"
$outputDirectory = Join-Path $PSScriptRoot "accuracy_oriented_results_v3\cifar10\gradual_refinement\seed_42"
$lastCheckpoint = Join-Path $outputDirectory "last_checkpoint.pth"
$bestCheckpoint = Join-Path $outputDirectory "best_checkpoint.pth"
$trainingSummary = Join-Path $outputDirectory "training_summary.json"
$evaluationDirectory = Join-Path $outputDirectory "evaluation_tta"

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Python environment not found: $python"
}
if (-not (Test-Path -LiteralPath $sourceCheckpoint -PathType Leaf)) {
    throw "Previous CIFAR-10 refinement checkpoint not found: $sourceCheckpoint"
}

$activeTrainers = @(
    Get-CimInstance Win32_Process | Where-Object {
        $_.Name -match "python" -and
        $_.CommandLine -match "train_accuracy_oriented_dense.py"
    }
)
if ($activeTrainers.Count -gt 0) {
    $activePids = ($activeTrainers | ForEach-Object { $_.ProcessId }) -join ", "
    throw "Another accuracy-oriented trainer is active (PID: $activePids). Wait for it to finish before starting CIFAR-10 gradual refinement."
}

$complete = $false
if (Test-Path -LiteralPath $trainingSummary -PathType Leaf) {
    $summary = Get-Content -LiteralPath $trainingSummary -Raw | ConvertFrom-Json
    $complete = ($summary.early_stopped -eq $true -or [int]$summary.last_epoch -ge 59)
}

$arguments = @(
    $trainer,
    "--dataset", "cifar10",
    "--data_path", $dataPath,
    "--output_dir", $outputDirectory,
    "--seed", "42",
    "--split_seed", "2026",
    "--download", "false",
    "--epochs", "60",
    "--batch_size", "128",
    "--num_workers", "4",
    "--lr_backbone", "1e-5",
    "--lr_classifier", "5e-5",
    "--min_lr", "1e-6",
    "--warmup_epochs", "3",
    "--weight_decay", "0.02",
    "--label_smoothing", "0.05",
    "--mixup_alpha", "0.1",
    "--cutmix_alpha", "0.5",
    "--randaugment_num_ops", "2",
    "--randaugment_magnitude", "7",
    "--random_erasing", "0.05",
    "--drop_path", "0.05",
    "--augmentation_schedule", "refinement60",
    "--ema_decay", "0.999",
    "--early_stopping_patience", "15",
    "--amp", "true",
    "--device", "cuda"
)

if ($complete) {
    Write-Host "CIFAR-10 gradual refinement is already complete."
}
elseif (Test-Path -LiteralPath $lastCheckpoint -PathType Leaf) {
    Write-Host "Resuming CIFAR-10 gradual refinement."
    $arguments += @("--resume", $lastCheckpoint)
    & $python @arguments
    if ($LASTEXITCODE -ne 0) { throw "CIFAR-10 gradual refinement resume failed" }
}
else {
    Write-Host "Starting gradual refinement from the 94.94% validation checkpoint."
    $arguments += @("--refinement_checkpoint", $sourceCheckpoint)
    & $python @arguments
    if ($LASTEXITCODE -ne 0) { throw "CIFAR-10 gradual refinement failed" }
}

if (-not (Test-Path -LiteralPath $bestCheckpoint -PathType Leaf)) {
    throw "Gradual-refinement best checkpoint unavailable: $bestCheckpoint"
}

& $python $evaluator `
    --checkpoint $bestCheckpoint `
    --dataset cifar10 `
    --data_path $dataPath `
    --output_dir $evaluationDirectory `
    --batch_size 128 `
    --num_workers 4 `
    --device cuda
if ($LASTEXITCODE -ne 0) { throw "CIFAR-10 gradual-refinement TTA evaluation failed" }

$report = Get-Content -LiteralPath (Join-Path $evaluationDirectory "tta_evaluation.json") -Raw | ConvertFrom-Json
$tta = $report.results | Where-Object { $_.mode -eq "flip_shift" }
$accuracy = [double]$tta.accuracy
Write-Host ("Final CIFAR-10 10-view TTA accuracy: {0:N2}%" -f $accuracy)
if ($accuracy -gt 96.0) {
    Write-Host "Success: gradual refinement exceeded 96% TTA accuracy."
}
elseif ($accuracy -ge 95.5) {
    Write-Host "Promising: repeat the same recipe for the remaining seeds."
}
else {
    Write-Warning "Below 95.5%: additional seeds are not recommended for this recipe."
}
