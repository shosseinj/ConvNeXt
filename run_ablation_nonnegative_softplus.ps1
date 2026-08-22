$ErrorActionPreference = "Stop"

$python = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"
$trainer = Join-Path $PSScriptRoot "train_continuous_ttfs_cifar10_32x32_stem1.py"
$evaluator = Join-Path $PSScriptRoot "Evaluation\evaluate_sparsity.py"
$summarizer = Join-Path $PSScriptRoot "Evaluation\summarize_pointwise_parameterization_ablation.py"
$dataPath = "..\cifar_data"
$seeds = @(42, 6543, 7777)

function Test-TorchCheckpoint([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        return $false
    }
    & $python -c "import sys, torch; torch.load(sys.argv[1], map_location='cpu', weights_only=False)" $Path
    return $LASTEXITCODE -eq 0
}

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Python environment not found: $python"
}

foreach ($seed in $seeds) {
    $outputDirectory = Join-Path $PSScriptRoot (
        "fine_tune_results_v3\cifar100\ablation_nonnegative_softplus\seed_$seed"
    )
    $trainingSummary = Join-Path $outputDirectory "training_summary.json"
    $bestCheckpoint = Join-Path $outputDirectory "best_checkpoint.pth"
    $sparsityReport = Join-Path $outputDirectory "activation_sparsity.md"

    $trainingComplete = $false
    if (Test-Path -LiteralPath $trainingSummary -PathType Leaf) {
        $summary = Get-Content -LiteralPath $trainingSummary -Raw | ConvertFrom-Json
        $trainingComplete = (
            $summary.early_stopped -eq $true -or
            [int]$summary.last_epoch -ge 249
        )
    }

    if (-not $trainingComplete) {
        $arguments = @(
            $trainer,
            "--dataset", "cifar100", "--data_path", $dataPath,
            "--output_dir", $outputDirectory, "--download", "false",
            "--experiment_name", "fully_ttfs_softplus",
            "--experiment_notes", "Fully-TTFS pointwise Softplus parameterization trained from scratch",
            "--dims", "96,192,384,768", "--depths", "2,2,6,2",
            "--dw_kernel_size", "3", "--dwconv_mode", "ttfs",
            "--downsample_mode", "ttfs", "--stage_delays", "0.05,0.02,0.01,0.01",
            "--pw1_mode", "ttfs", "--pw2_mode", "ttfs",
            "--residual_operator", "min", "--ttfs_norm_mode", "score_layernorm",
            "--final_score_norm", "true",
            "--pointwise_weight_parameterization", "softplus",
            "--epochs", "250", "--batch_size", "128", "--num_workers", "4",
            "--val_size", "5000", "--lr", "0.0001", "--min_lr", "1e-6",
            "--warmup_epochs", "5", "--lr_scheduler_patience", "3",
            "--lr_scheduler_factor", "0.85", "--weight_decay", "0.05",
            "--label_smoothing", "0.1", "--mixup_alpha", "0.2",
            "--cutmix_alpha", "1.0", "--randaugment", "true",
            "--randaugment_num_ops", "2", "--randaugment_magnitude", "9",
            "--random_erasing", "0.1", "--head_dropout", "0.1",
            "--spike_dropout", "0", "--drop_path", "0", "--t_min", "0",
            "--t_max", "1", "--ema", "false", "--ema_decay", "0.9998",
            "--early_stopping_patience", "30",
            "--early_stopping_min_delta", "0.02", "--seed", "$seed",
            "--amp", "true", "--device", "cuda"
        )

        $lastCheckpoint = Join-Path $outputDirectory "last_checkpoint.pth"
        $resumeCheckpoint = $null
        foreach ($candidate in @($lastCheckpoint, $bestCheckpoint)) {
            if (Test-TorchCheckpoint $candidate) {
                $resumeCheckpoint = $candidate
                break
            }
            if (Test-Path -LiteralPath $candidate -PathType Leaf) {
                $diagnostic = "$candidate.interrupted_corrupt_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
                Move-Item -LiteralPath $candidate -Destination $diagnostic
                Write-Warning "Preserved corrupt checkpoint as: $diagnostic"
            }
        }
        if ($resumeCheckpoint) {
            $arguments += @("--resume", $resumeCheckpoint)
            Write-Host "Resuming Softplus Fully-TTFS: seed=$seed"
        } else {
            Write-Host "Starting Softplus Fully-TTFS from scratch: seed=$seed"
        }
        & $python @arguments
        if ($LASTEXITCODE -ne 0) {
            throw "Softplus Fully-TTFS training failed: seed=$seed"
        }
    } else {
        Write-Host "Training already complete: seed=$seed"
    }

    if (-not (Test-Path -LiteralPath $sparsityReport -PathType Leaf)) {
        if (-not (Test-TorchCheckpoint $bestCheckpoint)) {
            throw "Valid best checkpoint unavailable: $bestCheckpoint"
        }
        & $python $evaluator --dataset cifar100 --checkpoint $bestCheckpoint `
            --data_path $dataPath --dw_kernel_size 3 --batch_size 128 `
            --workers 4 --device cuda
        if ($LASTEXITCODE -ne 0) {
            throw "Softplus sparsity evaluation failed: seed=$seed"
        }
    }
}

& $python $summarizer
if ($LASTEXITCODE -ne 0) {
    throw "Pointwise parameterization summary generation failed"
}
Write-Host "Softplus Fully-TTFS ablation complete."
