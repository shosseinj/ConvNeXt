$ErrorActionPreference = "Stop"

$python = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"
$trainer = Join-Path $PSScriptRoot "train_continuous_ttfs_cifar10_32x32_stem1.py"
$evaluator = Join-Path $PSScriptRoot "Evaluation\evaluate_sparsity.py"
$summarizer = Join-Path $PSScriptRoot "Evaluation\summarize_residual_ablation.py"
$dataPath = "..\cifar_data"
$seeds = @(42, 6543, 7777)
$operators = @("mean", "learnable_gate")

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Python environment not found: $python"
}

function Invoke-SparsityEvaluation {
    param(
        [Parameter(Mandatory = $true)][string]$Checkpoint,
        [Parameter(Mandatory = $true)][string]$Label
    )
    if (-not (Test-Path -LiteralPath $Checkpoint -PathType Leaf)) {
        throw "Best checkpoint unavailable for ${Label}: $Checkpoint"
    }
    Write-Host "Evaluating ${Label}"
    & $python $evaluator `
        --dataset cifar100 `
        --checkpoint $Checkpoint `
        --data_path $dataPath `
        --dw_kernel_size 3 `
        --batch_size 128 `
        --workers 4 `
        --device cuda
    if ($LASTEXITCODE -ne 0) {
        throw "Sparsity/SynOps evaluation failed for ${Label}"
    }
}

# The completed minimum campaign is evaluation-only and is never retrained.
foreach ($seed in $seeds) {
    $minimumCheckpoint = Join-Path $PSScriptRoot (
        "fine_tune_results_v3\cifar100\fully_ttfs\seed_$seed\best_checkpoint.pth"
    )
    Invoke-SparsityEvaluation `
        -Checkpoint $minimumCheckpoint `
        -Label "minimum seed=$seed"
}

foreach ($operator in $operators) {
    foreach ($seed in $seeds) {
        $sourceCheckpoint = Join-Path $PSScriptRoot (
            "results\cifar100\downsample_dense_dwconv_dense\seed_$seed\best_checkpoint.pth"
        )
        $outputDirectory = Join-Path $PSScriptRoot (
            "fine_tune_results_v3\cifar100\ablation_residual\$operator\seed_$seed"
        )
        $lastCheckpoint = Join-Path $outputDirectory "last_checkpoint.pth"
        $bestCheckpoint = Join-Path $outputDirectory "best_checkpoint.pth"
        $trainingSummary = Join-Path $outputDirectory "training_summary.json"

        if (-not (Test-Path -LiteralPath $sourceCheckpoint -PathType Leaf)) {
            throw "Matched dense source checkpoint unavailable: $sourceCheckpoint"
        }

        $arguments = @(
            $trainer,
            "--dataset", "cifar100",
            "--data_path", $dataPath,
            "--output_dir", $outputDirectory,
            "--download", "false",
            "--experiment_name", "residual_${operator}_seed${seed}",
            "--experiment_notes", "Reviewer 1 Comment 4: Fully-TTFS residual fusion ablation initialized from the matched dense best checkpoint",
            "--dims", "96,192,384,768",
            "--depths", "2,2,6,2",
            "--dw_kernel_size", "3",
            "--dwconv_mode", "ttfs",
            "--downsample_mode", "ttfs",
            "--stage_delays", "0.05,0.02,0.01,0.01",
            "--pw1_mode", "ttfs",
            "--pw2_mode", "ttfs",
            "--residual_operator", $operator,
            "--allow_pretrained_residual_operator_change", "true",
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
            "--grad_clip", "5",
            "--t_min", "0",
            "--t_max", "1",
            "--force_positive_weights", "false",
            "--force_positive_pointwise_weights", "false",
            "--ema", "false",
            "--early_stopping_patience", "30",
            "--early_stopping_min_delta", "0.02",
            "--seed", "$seed",
            "--amp", "true",
            "--device", "cuda"
        )

        $complete = $false
        if (Test-Path -LiteralPath $trainingSummary -PathType Leaf) {
            $summary = Get-Content -LiteralPath $trainingSummary -Raw | ConvertFrom-Json
            $complete = (
                $summary.early_stopped -eq $true -or
                [int]$summary.last_epoch -ge 249
            )
        }

        if ($complete) {
            Write-Host "Training already complete: operator=$operator seed=$seed"
        }
        elseif (Test-Path -LiteralPath $lastCheckpoint -PathType Leaf) {
            Write-Host "Resuming: operator=$operator seed=$seed"
            $arguments += @("--resume", $lastCheckpoint)
            & $python @arguments
            if ($LASTEXITCODE -ne 0) {
                throw "Resume failed: operator=$operator seed=$seed"
            }
        }
        else {
            Write-Host "Starting: operator=$operator seed=$seed"
            $arguments += @("--pretrained_checkpoint", $sourceCheckpoint)
            & $python @arguments
            if ($LASTEXITCODE -ne 0) {
                throw "Training failed: operator=$operator seed=$seed"
            }
        }

        Invoke-SparsityEvaluation `
            -Checkpoint $bestCheckpoint `
            -Label "$operator seed=$seed"
    }
}

& $python $summarizer
if ($LASTEXITCODE -ne 0) {
    throw "Residual-fusion report generation failed"
}

Write-Host "CIFAR-100 Fully-TTFS residual-fusion ablation complete."
