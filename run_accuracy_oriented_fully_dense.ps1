$ErrorActionPreference = "Stop"

$python = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"
$trainer = Join-Path $PSScriptRoot "train_accuracy_oriented_dense.py"
$evaluator = Join-Path $PSScriptRoot "evaluate_accuracy_oriented_dense.py"
$summarizer = Join-Path $PSScriptRoot "Evaluation\summarize_accuracy_oriented_dense.py"
$dataPath = Join-Path $PSScriptRoot "..\cifar_data"
$seeds = @(42, 6543, 7777)
$datasets = @("cifar10", "cifar100")

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Python environment not found: $python"
}

foreach ($seed in $seeds) {
    foreach ($dataset in $datasets) {
        $outputDirectory = Join-Path $PSScriptRoot "accuracy_oriented_results\$dataset\fully_dense\seed_$seed"
        $lastCheckpoint = Join-Path $outputDirectory "last_checkpoint.pth"
        $bestCheckpoint = Join-Path $outputDirectory "best_checkpoint.pth"
        $trainingSummary = Join-Path $outputDirectory "training_summary.json"
        $evaluationDirectory = Join-Path $outputDirectory "evaluation_tta"
        $complete = $false

        if (Test-Path -LiteralPath $trainingSummary -PathType Leaf) {
            $summary = Get-Content -LiteralPath $trainingSummary -Raw | ConvertFrom-Json
            $complete = ($summary.early_stopped -eq $true -or [int]$summary.last_epoch -ge 299)
        }

        $arguments = @(
            $trainer,
            "--dataset", $dataset,
            "--data_path", $dataPath,
            "--output_dir", $outputDirectory,
            "--seed", "$seed",
            "--split_seed", "2026",
            "--download", "false",
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
            "--ema_decay", "0.9999",
            "--early_stopping_patience", "50",
            "--amp", "true",
            "--device", "cuda"
        )

        if ($complete) {
            Write-Host "Training already complete: dataset=$dataset seed=$seed"
        }
        elseif (Test-Path -LiteralPath $lastCheckpoint -PathType Leaf) {
            Write-Host "Resuming: dataset=$dataset seed=$seed"
            $arguments += @("--resume", $lastCheckpoint, "--imagenet_checkpoint", "")
            & $python @arguments
            if ($LASTEXITCODE -ne 0) { throw "Training resume failed: dataset=$dataset seed=$seed" }
        }
        else {
            Write-Host "Starting: dataset=$dataset seed=$seed"
            $arguments += @("--imagenet_checkpoint", "official")
            & $python @arguments
            if ($LASTEXITCODE -ne 0) { throw "Training failed: dataset=$dataset seed=$seed" }
        }

        if (-not (Test-Path -LiteralPath $bestCheckpoint -PathType Leaf)) {
            throw "Best checkpoint unavailable: dataset=$dataset seed=$seed"
        }
        Write-Host "Evaluating standard and 10-view TTA: dataset=$dataset seed=$seed"
        & $python $evaluator `
            --checkpoint $bestCheckpoint `
            --dataset $dataset `
            --data_path $dataPath `
            --output_dir $evaluationDirectory `
            --batch_size 128 `
            --num_workers 4 `
            --device cuda
        if ($LASTEXITCODE -ne 0) { throw "Evaluation failed: dataset=$dataset seed=$seed" }
    }
}

& $python $summarizer
if ($LASTEXITCODE -ne 0) { throw "Final summarization failed" }

Write-Host "Fully-dense CIFAR-10/CIFAR-100 campaign complete."
