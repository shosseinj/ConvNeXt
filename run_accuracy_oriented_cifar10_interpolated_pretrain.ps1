# $ErrorActionPreference = "Stop"

# $python = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"
# $trainer = Join-Path $PSScriptRoot "train_accuracy_oriented_dense.py"
# $evaluator = Join-Path $PSScriptRoot "evaluate_accuracy_oriented_dense.py"
# $dataPath = Join-Path $PSScriptRoot "..\cifar_data"
# $outputDirectory = Join-Path $PSScriptRoot "accuracy_oriented_results_v7\cifar10\interpolated_imagenet_pretrain\seed_42"
# $lastCheckpoint = Join-Path $outputDirectory "last_checkpoint.pth"
# $bestCheckpoint = Join-Path $outputDirectory "best_checkpoint.pth"
# $trainingSummary = Join-Path $outputDirectory "training_summary.json"
# $evaluationDirectory = Join-Path $outputDirectory "evaluation_tta"

# if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
#     throw "Python environment not found: $python"
# }

# $activeTrainers = @(Get-CimInstance Win32_Process | Where-Object {
#     $_.Name -match "python" -and $_.CommandLine -match "train_accuracy_oriented_dense.py"
# })
# if ($activeTrainers.Count -gt 0) {
#     $activePids = ($activeTrainers | ForEach-Object { $_.ProcessId }) -join ", "
#     throw "Another accuracy-oriented trainer is active (PID: $activePids)."
# }

# $complete = $false
# if (Test-Path -LiteralPath $trainingSummary -PathType Leaf) {
#     $summary = Get-Content -LiteralPath $trainingSummary -Raw | ConvertFrom-Json
#     $complete = ($summary.early_stopped -eq $true -or [int]$summary.last_epoch -ge 299)
# }

# $arguments = @(
#     $trainer,
#     "--dataset", "cifar10",
#     "--data_path", $dataPath,
#     "--output_dir", $outputDirectory,
#     "--seed", "42",
#     "--split_seed", "2026",
#     "--download", "false",
#     "--imagenet_checkpoint", "official",
#     "--interpolate_imagenet_convs", "true",
#     "--epochs", "300",
#     "--batch_size", "128",
#     "--num_workers", "4",
#     "--lr_transferred", "2e-5",
#     "--lr_new", "2e-4",
#     "--min_lr", "1e-6",
#     "--warmup_epochs", "10",
#     "--weight_decay", "0.05",
#     "--label_smoothing", "0.1",
#     "--mixup_alpha", "0.2",
#     "--cutmix_alpha", "1.0",
#     "--randaugment_num_ops", "2",
#     "--randaugment_magnitude", "9",
#     "--random_erasing", "0.1",
#     "--drop_path", "0.1",
#     "--augmentation_schedule", "static",
#     "--ema_decay", "0.9999",
#     "--early_stopping_patience", "50",
#     "--amp", "true",
#     "--device", "cuda"
# )

# if ($complete) {
#     Write-Host "Interpolated-pretraining run is already complete."
# }
# elseif (Test-Path -LiteralPath $lastCheckpoint -PathType Leaf) {
#     Write-Host "Resuming interpolated-pretraining CIFAR-10 run."
#     $arguments += @("--resume", $lastCheckpoint)
#     & $python @arguments
#     if ($LASTEXITCODE -ne 0) { throw "Training resume failed" }
# }
# else {
#     Write-Host "Starting fresh CIFAR-10 training with interpolated ImageNet convolution kernels."
#     & $python @arguments
#     if ($LASTEXITCODE -ne 0) { throw "Training failed" }
# }

# if (-not (Test-Path -LiteralPath $bestCheckpoint -PathType Leaf)) {
#     throw "Best checkpoint unavailable: $bestCheckpoint"
# }

# & $python $evaluator `
#     --checkpoint $bestCheckpoint `
#     --dataset cifar10 `
#     --data_path $dataPath `
#     --output_dir $evaluationDirectory `
#     --batch_size 128 `
#     --num_workers 4 `
#     --device cuda
# if ($LASTEXITCODE -ne 0) { throw "TTA evaluation failed" }

# $report = Get-Content -LiteralPath (Join-Path $evaluationDirectory "tta_evaluation.json") -Raw | ConvertFrom-Json
# $standard = $report.results | Where-Object { $_.mode -eq "none" }
# $tta = $report.results | Where-Object { $_.mode -eq "flip_shift" }
# Write-Host ("Single-view test accuracy: {0:N2}%" -f [double]$standard.accuracy)
# Write-Host ("10-view TTA test accuracy: {0:N2}%" -f [double]$tta.accuracy)



$ErrorActionPreference = "Stop"

$python = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"
$trainer = Join-Path $PSScriptRoot "train_accuracy_oriented_dense.py"
$evaluator = Join-Path $PSScriptRoot "evaluate_accuracy_oriented_dense.py"
$dataPath = Join-Path $PSScriptRoot "..\cifar_data"

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Python environment not found: $python"
}

# Prevent accidentally starting this experiment while another trainer is running.
$activeTrainers = @(Get-CimInstance Win32_Process | Where-Object {
    $_.Name -match "python" -and $_.CommandLine -match "train_accuracy_oriented_dense.py"
})

if ($activeTrainers.Count -gt 0) {
    $activePids = ($activeTrainers | ForEach-Object { $_.ProcessId }) -join ", "
    throw "Another accuracy-oriented trainer is active (PID: $activePids)."
}


function Run-Experiment {
    param (
        [Parameter(Mandatory = $true)]
        [string]$Dataset,

        [Parameter(Mandatory = $true)]
        [int]$Seed
    )

    Write-Host ""
    Write-Host "============================================================"
    Write-Host "Starting experiment"
    Write-Host "Dataset : $Dataset"
    Write-Host "Seed    : $Seed"
    Write-Host "============================================================"
    Write-Host ""

    $outputDirectory = Join-Path `
        $PSScriptRoot `
        "accuracy_oriented_results_v7\$Dataset\interpolated_imagenet_pretrain\seed_$Seed"

    $lastCheckpoint = Join-Path $outputDirectory "last_checkpoint.pth"
    $bestCheckpoint = Join-Path $outputDirectory "best_checkpoint.pth"
    $trainingSummary = Join-Path $outputDirectory "training_summary.json"
    $evaluationDirectory = Join-Path $outputDirectory "evaluation_tta"

    # ---------------------------------------------------------
    # Check whether this run has already completed
    # ---------------------------------------------------------

    $complete = $false

    if (Test-Path -LiteralPath $trainingSummary -PathType Leaf) {
        $summary = Get-Content -LiteralPath $trainingSummary -Raw | ConvertFrom-Json

        $complete = (
            $summary.early_stopped -eq $true -or
            [int]$summary.last_epoch -ge 299
        )
    }

    # ---------------------------------------------------------
    # Training arguments
    # ---------------------------------------------------------

    $arguments = @(
        $trainer,

        "--dataset", $Dataset,
        "--data_path", $dataPath,
        "--output_dir", $outputDirectory,

        "--seed", "$Seed",
        "--split_seed", "2026",

        "--download", "false",

        "--imagenet_checkpoint", "official",
        "--interpolate_imagenet_convs", "true",

        "--epochs", "300",
        "--batch_size", "128",
        "--num_workers", "4",

        "--lr_transferred", "2e-5",
        "--lr_new", "2e-4",
        "--min_lr", "1e-6",
        "--warmup_epochs", "10",

        "--weight_decay", "0.05",

        "--label_smoothing", "0.1",
        "--mixup_alpha", "0.2",
        "--cutmix_alpha", "1.0",

        "--randaugment_num_ops", "2",
        "--randaugment_magnitude", "9",
        "--random_erasing", "0.1",

        "--drop_path", "0.1",

        "--augmentation_schedule", "static",

        "--ema_decay", "0.9999",

        "--early_stopping_patience", "50",

        "--amp", "true",
        "--device", "cuda"
    )

    # ---------------------------------------------------------
    # Train / resume
    # ---------------------------------------------------------

    if ($complete) {
        Write-Host "Training already complete:"
        Write-Host "  Dataset: $Dataset"
        Write-Host "  Seed:    $Seed"
    }
    elseif (
        (Test-Path -LiteralPath $bestCheckpoint -PathType Leaf) -or
        (Test-Path -LiteralPath $lastCheckpoint -PathType Leaf)
    ) {

        Write-Host "Resuming training:"
        Write-Host "  Dataset: $Dataset"
        Write-Host "  Seed:    $Seed"

        $resumeCheckpoint = if (Test-Path -LiteralPath $bestCheckpoint -PathType Leaf) {
            $bestCheckpoint
        }
        else {
            $lastCheckpoint
        }
        Write-Host "  Checkpoint: $resumeCheckpoint"
        $arguments += @("--resume", $resumeCheckpoint)

        & $python @arguments

        if ($LASTEXITCODE -ne 0) {
            throw "Training resume failed for $Dataset seed $Seed"
        }
    }
    else {

        Write-Host "Starting fresh training:"
        Write-Host "  Dataset: $Dataset"
        Write-Host "  Seed:    $Seed"

        & $python @arguments

        if ($LASTEXITCODE -ne 0) {
            throw "Training failed for $Dataset seed $Seed"
        }
    }

    # ---------------------------------------------------------
    # Make sure best checkpoint exists
    # ---------------------------------------------------------

    if (-not (Test-Path -LiteralPath $bestCheckpoint -PathType Leaf)) {
        throw "Best checkpoint unavailable: $bestCheckpoint"
    }

    # ---------------------------------------------------------
    # Evaluation
    # ---------------------------------------------------------

    Write-Host ""
    Write-Host "Evaluating best checkpoint..."
    Write-Host ""

    & $python $evaluator `
        --checkpoint $bestCheckpoint `
        --dataset $Dataset `
        --data_path $dataPath `
        --output_dir $evaluationDirectory `
        --batch_size 128 `
        --num_workers 4 `
        --device cuda

    if ($LASTEXITCODE -ne 0) {
        throw "TTA evaluation failed for $Dataset seed $Seed"
    }

    # ---------------------------------------------------------
    # Read evaluation result
    # ---------------------------------------------------------

    $evaluationReport = Join-Path $evaluationDirectory "tta_evaluation.json"

    if (-not (Test-Path -LiteralPath $evaluationReport -PathType Leaf)) {
        throw "Evaluation report missing: $evaluationReport"
    }

    $report = Get-Content -LiteralPath $evaluationReport -Raw | ConvertFrom-Json

    $standard = $report.results | Where-Object {
        $_.mode -eq "none"
    }

    $tta = $report.results | Where-Object {
        $_.mode -eq "flip_shift"
    }

    Write-Host ""
    Write-Host "============================================================"
    Write-Host "Experiment completed"
    Write-Host "Dataset : $Dataset"
    Write-Host "Seed    : $Seed"

    if ($null -ne $standard) {
        Write-Host (
            "Single-view test accuracy: {0:N2}%" `
            -f [double]$standard.accuracy
        )
    }

    if ($null -ne $tta) {
        Write-Host (
            "10-view TTA test accuracy: {0:N2}%" `
            -f [double]$tta.accuracy
        )
    }

    Write-Host "============================================================"
    Write-Host ""
}


# =============================================================
# EXPERIMENT ORDER
# =============================================================

# Existing CIFAR-10 seed 42 is NOT rerun here.
# Run the two additional CIFAR-10 seeds first.

# Run-Experiment -Dataset "cifar10" -Seed 6543
# Run-Experiment -Dataset "cifar10" -Seed 7777


# Then run CIFAR-100 using all three seeds.

# Run-Experiment -Dataset "cifar100" -Seed 42
# Run-Experiment -Dataset "cifar100" -Seed 6543
Run-Experiment -Dataset "cifar100" -Seed 7777


Write-Host ""
Write-Host "############################################################"
Write-Host "ALL REQUESTED EXPERIMENTS COMPLETED"
Write-Host "############################################################"
