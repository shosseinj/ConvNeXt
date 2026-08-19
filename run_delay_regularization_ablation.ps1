$ErrorActionPreference = "Stop"

$python = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"
$trainer = Join-Path $PSScriptRoot "train_continuous_ttfs_cifar10_32x32_stem1.py"
$summarizer = Join-Path $PSScriptRoot "Evaluation\summarize_delay_regularization_ablation.py"
$dataPath = "..\cifar_data"
$seeds = @(42, 6543, 7777)
$settings = @(
    @{ name = "lambda_0p01"; weight = "0.01" },
    @{ name = "lambda_0p1"; weight = "0.1" }
)

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

foreach ($setting in $settings) {
    foreach ($seed in $seeds) {
        $outputDirectory = Join-Path $PSScriptRoot (
            "results\cifar100\ablation_delay_regularization\$($setting.name)\seed_$seed"
        )
        $trainingSummary = Join-Path $outputDirectory "training_summary.json"
        if (Test-Path -LiteralPath $trainingSummary -PathType Leaf) {
            $summary = Get-Content -LiteralPath $trainingSummary -Raw | ConvertFrom-Json
            if ([double]$summary.delay_regularization_weight -eq [double]$setting.weight) {
                Write-Host "Skipping completed run: $($setting.name), seed=$seed"
                continue
            }
            throw "Completed summary has wrong regularization weight: $trainingSummary"
        }

        $arguments = @(
            $trainer,
            "--dataset", "cifar100",
            "--data_path", $dataPath,
            "--output_dir", $outputDirectory,
            "--download", "false",
            "--experiment_name", "delay_regularization_$($setting.name)_seed_$seed",
            "--experiment_notes", "Delay regularization ablation: explicit mean effective D_mid/D_out penalty",
            "--dims", "96,192,384,768", "--depths", "2,2,6,2",
            "--dw_kernel_size", "3", "--dwconv_mode", "dense",
            "--downsample_mode", "dense", "--stage_delays", "0.05,0.02,0.01,0.01",
            "--pw1_mode", "ttfs", "--pw2_mode", "ttfs",
            "--residual_operator", "min", "--ttfs_norm_mode", "score_layernorm",
            "--final_score_norm", "true", "--delay_regularization_weight", $setting.weight,
            "--epochs", "300", "--batch_size", "128", "--lr", "0.0002",
            "--min_lr", "1e-6", "--warmup_epochs", "10",
            "--lr_scheduler_patience", "3", "--lr_scheduler_factor", "0.85",
            "--weight_decay", "0.05", "--label_smoothing", "0.1",
            "--mixup_alpha", "0.2", "--cutmix_alpha", "1.0",
            "--randaugment", "true", "--randaugment_num_ops", "2",
            "--randaugment_magnitude", "9", "--random_erasing", "0.1",
            "--head_dropout", "0.1", "--spike_dropout", "0", "--drop_path", "0",
            "--t_min", "0", "--t_max", "1", "--ema", "true",
            "--ema_decay", "0.9998", "--early_stopping_patience", "30",
            "--early_stopping_min_delta", "0.02", "--seed", "$seed",
            "--amp", "true", "--device", "cuda"
        )

        $lastCheckpoint = Join-Path $outputDirectory "last_checkpoint.pth"
        $bestCheckpoint = Join-Path $outputDirectory "best_checkpoint.pth"
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
            Write-Host "Resuming $($setting.name), seed=$seed from $resumeCheckpoint"
        } else {
            Write-Host "Starting $($setting.name), seed=$seed"
        }

        & $python @arguments
        if ($LASTEXITCODE -ne 0) {
            throw "Training failed: $($setting.name), seed=$seed"
        }
    }
}

& $python $summarizer
if ($LASTEXITCODE -ne 0) {
    throw "Delay regularization summary generation failed"
}
Write-Host "Delay regularization ablation complete."
