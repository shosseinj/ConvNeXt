

$ErrorActionPreference = "Stop"

$python = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"
$trainer = Join-Path $PSScriptRoot "train_continuous_ttfs_cifar10_32x32_stem1.py"

$dataset = "cifar100"
$dataPath = "..\cifar_data"

$seeds = @(42, 6543, 7777)

# Reference already exists:
# 0.05,0.02,0.01,0.01

$settings = @(
    @{
        name   = "low"
        delays = "0.025,0.01,0.005,0.005"
    },
    @{
        name   = "high"
        delays = "0.10,0.04,0.02,0.02"
    }
)

foreach ($seed in $seeds) {

    foreach ($setting in $settings) {

        $outputDirectory = Join-Path $PSScriptRoot (
            "results\cifar100\ablation_delay_initialization\$($setting.name)\seed_$seed"
        )

        $trainingSummary = Join-Path $outputDirectory "training_summary.json"
        if (Test-Path -LiteralPath $trainingSummary) {
            Write-Host "Skipping completed run: $($setting.name), seed=$seed"
            continue
        }

        $resumeArguments = @()
        $lastCheckpoint = Join-Path $outputDirectory "last_checkpoint.pth"
        $bestCheckpoint = Join-Path $outputDirectory "best_checkpoint.pth"
        $resumeCheckpoint = if (Test-Path -LiteralPath $lastCheckpoint) {
            $lastCheckpoint
        } elseif (Test-Path -LiteralPath $bestCheckpoint) {
            $bestCheckpoint
        } else {
            $null
        }
        if ($resumeCheckpoint) {
            $resumeArguments = @("--resume", $resumeCheckpoint)
            Write-Host "Resuming from: $resumeCheckpoint"
        }

        Write-Host ""
        Write-Host "=============================================="
        Write-Host "Delay initialization: $($setting.name)"
        Write-Host "Stage delays: $($setting.delays)"
        Write-Host "Seed: $seed"
        Write-Host "=============================================="

        & $python $trainer `
            --dataset $dataset `
            --data_path $dataPath `
            --output_dir $outputDirectory `
            --download false `
            --experiment_name "delay_init_$($setting.name)_seed_$seed" `
            --experiment_notes "Reviewer 1 Comment 4 - Delay initialization ablation" `
            --dims "96,192,384,768" `
            --depths "2,2,6,2" `
            --dw_kernel_size 3 `
            --dwconv_mode dense `
            --downsample_mode dense `
            --stage_delays $setting.delays `
            --pw1_mode ttfs `
            --pw2_mode ttfs `
            --residual_operator min `
            --ttfs_norm_mode score_layernorm `
            --final_score_norm true `
            --epochs 300 `
            --batch_size 128 `
            --lr 0.0002 `
            --min_lr 1e-6 `
            --warmup_epochs 10 `
            --lr_scheduler_patience 3 `
            --lr_scheduler_factor 0.85 `
            --weight_decay 0.05 `
            --label_smoothing 0.1 `
            --mixup_alpha 0.2 `
            --cutmix_alpha 1.0 `
            --randaugment true `
            --randaugment_num_ops 2 `
            --randaugment_magnitude 9 `
            --random_erasing 0.1 `
            --head_dropout 0.1 `
            --spike_dropout 0 `
            --drop_path 0 `
            --t_min 0 `
            --t_max 1 `
            --ema true `
            --ema_decay 0.9998 `
            --early_stopping_patience 30 `
            --early_stopping_min_delta 0.02 `
            --seed $seed `
            --amp true `
            --device cuda `
            @resumeArguments

        if ($LASTEXITCODE -ne 0) {
            throw "Training failed: $($setting.name), seed=$seed"
        }
    }
}

Write-Host "Delay initialization ablation finished."
