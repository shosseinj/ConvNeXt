$ErrorActionPreference = "Stop"

$python = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"
$trainer = Join-Path $PSScriptRoot "train_continuous_ttfs_cifar10_32x32_stem1.py"
$evaluator = Join-Path $PSScriptRoot "Evaluation\evaluate_sparsity.py"
$dataPath = Join-Path $PSScriptRoot "..\cifar_data"
$sourceCheckpoint = Join-Path $PSScriptRoot "accuracy_oriented_results_v7\cifar100\interpolated_imagenet_pretrain\seed_42\best_checkpoint.pth"
$outputDirectory = Join-Path $PSScriptRoot "fine_tune_results_v4\cifar100\ann_to_fully_ttfs\seed_42"
$lastCheckpoint = Join-Path $outputDirectory "last_checkpoint.pth"
$bestCheckpoint = Join-Path $outputDirectory "best_checkpoint.pth"
$trainingSummary = Join-Path $outputDirectory "training_summary.json"

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Python environment not found: $python"
}
if (-not (Test-Path -LiteralPath $sourceCheckpoint -PathType Leaf)) {
    throw "CIFAR-100 ANN source checkpoint not found: $sourceCheckpoint"
}

$activeTrainers = @(Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match "python" -and
    $_.CommandLine -match "train_continuous_ttfs_cifar10_32x32_stem1.py"
})
if ($activeTrainers.Count -gt 0) {
    $activePids = ($activeTrainers | ForEach-Object { $_.ProcessId }) -join ", "
    throw "Another TTFS trainer is active (PID: $activePids)."
}

$complete = $false
if (Test-Path -LiteralPath $trainingSummary -PathType Leaf) {
    $summary = Get-Content -LiteralPath $trainingSummary -Raw | ConvertFrom-Json
    $complete = ($summary.early_stopped -eq $true -or [int]$summary.last_epoch -ge 249)
}

$arguments = @(
    $trainer,
    "--dataset", "cifar100",
    "--data_path", $dataPath,
    "--output_dir", $outputDirectory,
    "--download", "false",
    "--experiment_name", "ann_to_fully_ttfs",
    "--experiment_notes", "Fully-TTFS fine-tuning from the CIFAR-100 seed-42 accuracy-oriented ANN EMA checkpoint",
    "--dims", "96,192,384,768",
    "--depths", "2,2,6,2",
    "--dw_kernel_size", "3",
    "--dwconv_mode", "ttfs",
    "--downsample_mode", "ttfs",
    "--stage_delays", "0.05,0.02,0.01,0.01",
    "--pw1_mode", "ttfs",
    "--pw2_mode", "ttfs",
    "--residual_operator", "min",
    "--ttfs_norm_mode", "score_layernorm",
    "--final_score_norm", "true",
    "--epochs", "250",
    "--batch_size", "128",
    "--num_workers", "4",
    "--val_size", "5000",
    "--lr", "0.0001",
    "--min_lr", "1e-6",
    "--warmup_epochs", "5",
    "--lr_scheduler_patience", "3",
    "--lr_scheduler_factor", "0.85",
    "--weight_decay", "0.05",
    "--label_smoothing", "0.1",
    "--mixup_alpha", "0.2",
    "--cutmix_alpha", "1.0",
    "--randaugment", "true",
    "--randaugment_num_ops", "2",
    "--randaugment_magnitude", "9",
    "--random_erasing", "0.1",
    "--head_dropout", "0.1",
    "--spike_dropout", "0",
    "--drop_path", "0",
    "--t_min", "0",
    "--t_max", "1",
    "--ema", "false",
    "--early_stopping_patience", "30",
    "--early_stopping_min_delta", "0.02",
    "--seed", "42",
    "--split_seed", "2026",
    "--amp", "true",
    "--device", "cuda"
)

if ($complete) {
    Write-Host "ANN-to-Fully-TTFS CIFAR-100 training is already complete."
}
elseif (Test-Path -LiteralPath $lastCheckpoint -PathType Leaf) {
    Write-Host "Resuming ANN-to-Fully-TTFS CIFAR-100 training."
    $arguments += @("--resume", $lastCheckpoint)
    & $python @arguments
    if ($LASTEXITCODE -ne 0) { throw "TTFS training resume failed" }
}
else {
    Write-Host "Starting CIFAR-100 ANN-to-Fully-TTFS conversion and fine-tuning."
    $arguments += @("--ann_pretrained_checkpoint", $sourceCheckpoint)
    & $python @arguments
    if ($LASTEXITCODE -ne 0) { throw "ANN-to-Fully-TTFS training failed" }
}

if (-not (Test-Path -LiteralPath $bestCheckpoint -PathType Leaf)) {
    throw "Best Fully-TTFS checkpoint unavailable: $bestCheckpoint"
}

Write-Host "Evaluating validation-selected best checkpoint and activation sparsity."
& $python $evaluator `
    --dataset cifar100 `
    --checkpoint $bestCheckpoint `
    --data_path $dataPath `
    --dw_kernel_size 3 `
    --batch_size 128 `
    --workers 4 `
    --device cuda
if ($LASTEXITCODE -ne 0) { throw "Fully-TTFS sparsity evaluation failed" }

Write-Host "CIFAR-100 ANN-to-Fully-TTFS seed-42 run and 39-point evaluation complete."
