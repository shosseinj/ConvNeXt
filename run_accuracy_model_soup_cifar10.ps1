$ErrorActionPreference = "Stop"

$python = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"
$builder = Join-Path $PSScriptRoot "Evaluation\build_accuracy_model_soup.py"
$evaluator = Join-Path $PSScriptRoot "evaluate_accuracy_oriented_dense.py"
$dataPath = Join-Path $PSScriptRoot "..\cifar_data"
$outputDirectory = Join-Path $PSScriptRoot "accuracy_oriented_results_v5\cifar10\model_soup\seed_42"
$evaluationDirectory = Join-Path $outputDirectory "evaluation_tta"
$soupCheckpoint = Join-Path $outputDirectory "best_checkpoint.pth"
$searchReport = Join-Path $outputDirectory "model_soup_search.json"
$checkpoints = @(
    (Join-Path $PSScriptRoot "accuracy_oriented_results_v2\cifar10\fully_dense_refinement\seed_42\best_checkpoint.pth"),
    (Join-Path $PSScriptRoot "accuracy_oriented_results_v3\cifar10\gradual_refinement\seed_42\best_checkpoint.pth"),
    (Join-Path $PSScriptRoot "accuracy_oriented_results_v4\cifar10\low_augmentation_refinement\seed_42\best_checkpoint.pth")
)

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Python environment not found: $python"
}
foreach ($checkpoint in $checkpoints) {
    if (-not (Test-Path -LiteralPath $checkpoint -PathType Leaf)) {
        throw "Model-soup source checkpoint not found: $checkpoint"
    }
}

$activeTrainers = @(
    Get-CimInstance Win32_Process | Where-Object {
        $_.Name -match "python" -and
        $_.CommandLine -match "train_accuracy_oriented_dense.py"
    }
)
if ($activeTrainers.Count -gt 0) {
    $activePids = ($activeTrainers | ForEach-Object { $_.ProcessId }) -join ", "
    throw "Accuracy training is active (PID: $activePids). Wait before running the model-soup search."
}

if ((Test-Path -LiteralPath $soupCheckpoint -PathType Leaf) -and
    (Test-Path -LiteralPath $searchReport -PathType Leaf)) {
    Write-Host "Validation-selected model soup already exists; skipping the search."
}
else {
    Write-Host "Searching 66 validation-only EMA soup candidates."
    & $python $builder `
        --checkpoint $checkpoints[0] `
        --checkpoint $checkpoints[1] `
        --checkpoint $checkpoints[2] `
        --data_path $dataPath `
        --output_dir $outputDirectory `
        --split_seed 2026 `
        --batch_size 512 `
        --num_workers 4 `
        --device cuda
    if ($LASTEXITCODE -ne 0) { throw "CIFAR-10 model-soup validation search failed" }
}

& $python $evaluator `
    --checkpoint $soupCheckpoint `
    --dataset cifar10 `
    --data_path $dataPath `
    --output_dir $evaluationDirectory `
    --batch_size 128 `
    --num_workers 4 `
    --device cuda
if ($LASTEXITCODE -ne 0) { throw "CIFAR-10 model-soup TTA evaluation failed" }

$report = Get-Content -LiteralPath (Join-Path $evaluationDirectory "tta_evaluation.json") -Raw | ConvertFrom-Json
$tta = $report.results | Where-Object { $_.mode -eq "flip_shift" }
$accuracy = [double]$tta.accuracy
Write-Host ("Model-soup CIFAR-10 10-view TTA accuracy: {0:N2}%" -f $accuracy)
if ($accuracy -gt 96.0) {
    Write-Host "Success: the model soup exceeded 96% TTA accuracy."
}
elseif ($accuracy -ge 95.5) {
    Write-Host "Promising: repeat this procedure after corresponding checkpoints exist for the remaining seeds."
}
else {
    Write-Warning "Below 95.5%: model soup did not provide the required improvement."
}
