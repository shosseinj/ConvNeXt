$ErrorActionPreference = "Stop"

$python = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"
$trainer = Join-Path $PSScriptRoot "train_continuous_ttfs_cifar10_32x32_stem1.py"
$evaluator = Join-Path $PSScriptRoot "Evaluation\evaluate_sparsity.py"
$summarizer = Join-Path $PSScriptRoot "Evaluation\summarize_finetuned_ttfs.py"

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Python environment not found: $python"
}

$campaigns = @(
    @{
        Dataset = "cifar10"
        Seeds = @(42, 6543, 7777)
        SourceRoot = "results\cifar10\downsample_dense_dwconv_dense"
        DataPath = "..\cifar_data"
        BatchSize = 128
        ValidationSize = 5000
    },
    @{
        Dataset = "cifar100"
        Seeds = @(42, 6543, 7777)
        SourceRoot = "results\cifar100\downsample_dense_dwconv_dense"
        DataPath = "..\cifar_data"
        BatchSize = 128
        ValidationSize = 5000
    },
    @{
        Dataset = "tinyimagenet"
        Seeds = @(42, 2344, 5435)
        SourceRoot = "results\tinyimagenet\test_downsample_ttfs_dwconv_dense"
        DataPath = "..\cifar_data\tiny-imagenet-200"
        BatchSize = 32
        ValidationSize = 10000
    }
)

foreach ($campaign in $campaigns) {
    foreach ($seed in $campaign.Seeds) {
        $dataset = $campaign.Dataset
        $sourceCheckpoint = Join-Path $PSScriptRoot (
            "$($campaign.SourceRoot)\seed_$seed\best_checkpoint.pth"
        )
        $outputDirectory = Join-Path $PSScriptRoot (
            "fine_tune_results\$dataset\fully_ttfs\seed_$seed"
        )
        $lastCheckpoint = Join-Path $outputDirectory "last_checkpoint.pth"
        $bestCheckpoint = Join-Path $outputDirectory "best_checkpoint.pth"
        $trainingSummary = Join-Path $outputDirectory "training_summary.json"

        if (-not (Test-Path -LiteralPath $sourceCheckpoint -PathType Leaf)) {
            throw "Dense source checkpoint not found: $sourceCheckpoint"
        }

        $trainArguments = @(
            $trainer,
            "--dataset", $dataset,
            "--data_path", $campaign.DataPath,
            "--output_dir", $outputDirectory,
            "--download", "false",
            "--experiment_name", "fully_ttfs",
            "--experiment_notes", "Fine-tune fully TTFS ConvNeXt from the matched dense best checkpoint",
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
            "--epochs", "150",
            "--batch_size", "$($campaign.BatchSize)",
            "--num_workers", "4",
            "--val_size", "$($campaign.ValidationSize)",
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
            "--ema", "true",
            "--ema_decay", "0.9998",
            "--early_stopping_patience", "30",
            "--early_stopping_min_delta", "0.02",
            "--seed", "$seed",
            "--amp", "true",
            "--device", "cuda"
        )

        if (Test-Path -LiteralPath $trainingSummary -PathType Leaf) {
            Write-Host "Training already complete: dataset=$dataset seed=$seed"
        }
        else {
            if (Test-Path -LiteralPath $lastCheckpoint -PathType Leaf) {
                $trainArguments += @("--resume", $lastCheckpoint)
                Write-Host "Resuming TTFS fine-tuning: dataset=$dataset seed=$seed"
            }
            else {
                $trainArguments += @("--pretrained_checkpoint", $sourceCheckpoint)
                Write-Host "Starting dense-to-TTFS fine-tuning: dataset=$dataset seed=$seed"
            }
            & $python @trainArguments
            if ($LASTEXITCODE -ne 0) {
                throw "Training failed: dataset=$dataset seed=$seed"
            }
        }

        if (-not (Test-Path -LiteralPath $bestCheckpoint -PathType Leaf)) {
            throw "Best checkpoint unavailable after training: $bestCheckpoint"
        }
        Write-Host "Evaluating activation sparsity: dataset=$dataset seed=$seed"
        & $python $evaluator `
            --dataset $dataset `
            --checkpoint $bestCheckpoint `
            --data_path $campaign.DataPath `
            --dw_kernel_size 3 `
            --batch_size $campaign.BatchSize `
            --workers 4 `
            --device cuda
        if ($LASTEXITCODE -ne 0) {
            throw "Sparsity evaluation failed: dataset=$dataset seed=$seed"
        }
    }
}

& $python $summarizer `
    --root (Join-Path $PSScriptRoot "fine_tune_results") `
    --output (Join-Path $PSScriptRoot "fine_tune_results\FULLY_TTFS_FINE_TUNING_SUMMARY.md")
if ($LASTEXITCODE -ne 0) {
    throw "Fine-tuning summary generation failed"
}

