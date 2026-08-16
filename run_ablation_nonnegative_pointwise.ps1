$ErrorActionPreference = "Stop"

$python = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"
$trainer = Join-Path $PSScriptRoot "train_continuous_ttfs_cifar10_32x32_stem1.py"
$evaluator = Join-Path $PSScriptRoot "Evaluation\evaluate_sparsity.py"
$summarizer = Join-Path $PSScriptRoot "Evaluation\summarize_nonnegative_pointwise_ablation.py"
$dataPath = "..\cifar_data"
$seeds = @(42, 6543, 7777)

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Python environment not found: $python"
}

foreach ($seed in $seeds) {
    $sourceCheckpoint = Join-Path $PSScriptRoot (
        "fine_tune_results_v3\cifar100\fully_ttfs\seed_$seed\best_checkpoint.pth"
    )
    $outputDirectory = Join-Path $PSScriptRoot (
        "fine_tune_results_v3\cifar100\ablation_nonnegative_pointwise\seed_$seed"
    )
    $lastCheckpoint = Join-Path $outputDirectory "last_checkpoint.pth"
    $bestCheckpoint = Join-Path $outputDirectory "best_checkpoint.pth"
    $trainingSummary = Join-Path $outputDirectory "training_summary.json"
    $initialCheckpoint = Join-Path $outputDirectory (
        "initial_constraint\initial_constrained_checkpoint.pth"
    )

    if (-not (Test-Path -LiteralPath $sourceCheckpoint -PathType Leaf)) {
        throw "Fully-TTFS source checkpoint not found: $sourceCheckpoint"
    }

    $trainArguments = @(
        $trainer,
        "--dataset", "cifar100",
        "--data_path", $dataPath,
        "--output_dir", $outputDirectory,
        "--download", "false",
        "--experiment_name", "nonnegative_pointwise",
        "--experiment_notes", "Pointwise-only non-negative effective PW1/PW2 weights via ReLU, initialized from the matched fully-TTFS best checkpoint",
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
        "--force_positive_weights", "false",
        "--force_positive_pointwise_weights", "true",
        "--epochs", "50",
        "--batch_size", "128",
        "--num_workers", "4",
        "--val_size", "5000",
        "--lr", "1e-5",
        "--min_lr", "1e-6",
        "--warmup_epochs", "2",
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
        "--grad_clip", "5",
        "--t_min", "0",
        "--t_max", "1",
        "--ema", "false",
        "--ema_decay", "0.9998",
        "--early_stopping_patience", "15",
        "--early_stopping_min_delta", "0.02",
        "--seed", "$seed",
        "--amp", "true",
        "--device", "cuda"
    )

    $trainingComplete = $false
    if (Test-Path -LiteralPath $trainingSummary -PathType Leaf) {
        $summary = Get-Content -LiteralPath $trainingSummary -Raw | ConvertFrom-Json
        $trainingComplete = (
            $summary.early_stopped -eq $true -or
            [int]$summary.last_epoch -ge 49
        )
    }

    if ($trainingComplete) {
        Write-Host "Training already complete: seed=$seed"
    }
    elseif (Test-Path -LiteralPath $lastCheckpoint -PathType Leaf) {
        $trainArguments += @("--resume", $lastCheckpoint)
        Write-Host "Resuming pointwise-constrained fine-tuning: seed=$seed"
        & $python @trainArguments
        if ($LASTEXITCODE -ne 0) {
            throw "Training failed while resuming seed=$seed"
        }
    }
    else {
        $trainArguments += @(
            "--constrained_finetune_checkpoint", $sourceCheckpoint
        )
        Write-Host "Starting pointwise-constrained fine-tuning: seed=$seed"
        & $python @trainArguments
        if ($LASTEXITCODE -ne 0) {
            throw "Training failed for seed=$seed"
        }
    }

    if (-not (Test-Path -LiteralPath $initialCheckpoint -PathType Leaf)) {
        throw "Initial constrained checkpoint unavailable: $initialCheckpoint"
    }
    if (-not (Test-Path -LiteralPath $bestCheckpoint -PathType Leaf)) {
        throw "Best adapted checkpoint unavailable: $bestCheckpoint"
    }

    Write-Host "Evaluating immediate pointwise constraint: seed=$seed"
    & $python $evaluator `
        --dataset cifar100 `
        --checkpoint $initialCheckpoint `
        --data_path $dataPath `
        --dw_kernel_size 3 `
        --batch_size 128 `
        --workers 4 `
        --device cuda
    if ($LASTEXITCODE -ne 0) {
        throw "Immediate constraint evaluation failed: seed=$seed"
    }

    Write-Host "Evaluating adapted pointwise constraint: seed=$seed"
    & $python $evaluator `
        --dataset cifar100 `
        --checkpoint $bestCheckpoint `
        --data_path $dataPath `
        --dw_kernel_size 3 `
        --batch_size 128 `
        --workers 4 `
        --device cuda
    if ($LASTEXITCODE -ne 0) {
        throw "Adapted constraint evaluation failed: seed=$seed"
    }
}

& $python $summarizer
if ($LASTEXITCODE -ne 0) {
    throw "Ablation summary generation failed"
}

Write-Host "CIFAR-100 non-negative pointwise ablation complete."
