$ErrorActionPreference = "Stop"

$python = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"
$evaluator = Join-Path $PSScriptRoot "evaluate_ttfs_cifar10_tta.py"
$summarizer = Join-Path $PSScriptRoot "Evaluation\summarize_residual_ablation.py"
$dataPath = "..\cifar_data"
$seeds = @(42, 6543, 7777)
$operators = @("min", "mean", "learnable_gate")

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Python environment not found: $python"
}

foreach ($operator in $operators) {
    foreach ($seed in $seeds) {
        if ($operator -eq "min") {
            $runDirectory = Join-Path $PSScriptRoot (
                "fine_tune_results_v3\cifar100\fully_ttfs\seed_$seed"
            )
        }
        else {
            $runDirectory = Join-Path $PSScriptRoot (
                "fine_tune_results_v3\cifar100\ablation_residual\$operator\seed_$seed"
            )
        }
        $checkpoint = Join-Path $runDirectory "best_checkpoint.pth"
        $outputDirectory = Join-Path $runDirectory "evaluation_tta"

        if (-not (Test-Path -LiteralPath $checkpoint -PathType Leaf)) {
            throw "Best checkpoint unavailable: operator=$operator seed=$seed path=$checkpoint"
        }

        Write-Host "Running 10-view TTA: operator=$operator seed=$seed"
        & $python $evaluator `
            --checkpoint $checkpoint `
            --dataset cifar100 `
            --data_path $dataPath `
            --output_dir $outputDirectory `
            --batch_size 128 `
            --num_workers 4 `
            --device cuda `
            --amp true `
            --download false `
            --save_confusion_matrix false `
            --weights_source model `
            --tta_modes "none,flip_shift"
        if ($LASTEXITCODE -ne 0) {
            throw "TTA evaluation failed: operator=$operator seed=$seed"
        }
    }
}

& $python $summarizer
if ($LASTEXITCODE -ne 0) {
    throw "Residual-ablation TTA report generation failed"
}

Write-Host "Residual-fusion 10-view TTA evaluation complete."
