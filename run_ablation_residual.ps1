$dataset = "cifar100"
$downsample_mode = "dense"
$dwconv_mode = "dense"
$kernel_size = 3

$seeds = @(42, 2344, 5435)
$residual_operators = @("mean", "learnable_gate")

foreach ($seed in $seeds) {
    foreach  ($residual_operator in $residual_operators) {
        $experiment_name = $residual_operator
        $output_dir = ".\results\$dataset\ablation_residual\$residual_operator\seed_$seed"

        Write-Host "Running residual_operator=$residual_operator seed=$seed"
        python `
            ".\train_continuous_ttfs_cifar10_32x32_stem1.py" `
            --dataset $dataset `
            --data_path "..\cifar_data" `
            --output_dir $output_dir `
            --download false `
            --experiment_name $experiment_name `
            --experiment_notes "Residual operator ablation with dense depthwise and downsampling convolutions" `
            --dims "96,192,384,768" `
            --depths "2,2,6,2" `
            --dw_kernel_size $kernel_size `
            --dwconv_mode $dwconv_mode `
            --downsample_mode $downsample_mode `
            --stage_delays "0.05,0.02,0.01,0.01" `
            --pw1_mode ttfs `
            --pw2_mode ttfs `
            --residual_operator $residual_operator `
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
            --device cuda

        if ($LASTEXITCODE -ne 0) {
            throw "Training failed for residual_operator=$residual_operator seed=$seed"
        }
    }
}
