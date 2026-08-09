& "C:\Users\jafari.h\Desktop\ai_project\.venv\Scripts\python.exe" `  ".\train_continuous_ttfs_cifar10_32x32_stem1.py"`
--data_path "..\cifar_data" `  --output_dir ".\results\cifar10_ttfs_native32_k3_ttfs_score_layernorm_seed42"`
--experiment_name "cifar10_ttfs_native32_k3_ttfs_score_layernorm_seed42" `  --dims "96,192,384,768"`
--depths "2,2,6,2" `  --dw_kernel_size 3`
--stage_delays "0.05,0.02,0.01,0.01" `  --pw2_mode ttfs`
--ttfs_norm_mode score_layernorm `  --spike_dropout 0`
--head_dropout 0.1 `  --drop_path 0`
--t_min 0 `  --t_max 1`
--epochs 350 `  --lr 0.0002`
--min_lr 0.000001 `  --warmup_epochs 5`
--lr_scheduler_patience 6 `  --lr_scheduler_factor 0.5`
--early_stopping_patience 20 `  --early_stopping_min_delta 0.05`
--ema true `  --ema_decay 0.9998`
--device cuda

## resume training

```
& "C:\Users\jafari.h\Desktop\ai_project\.venv\Scripts\python.exe" `
  ".\train_continuous_ttfs_cifar10_32x32_stem1.py" `
  --data_path "..\cifar_data" `
  --output_dir ".\results\cifar10_ttfs_small_64_128_256_512_seed42" `
  --resume ".\results\cifar10_ttfs_small_64_128_256_512_seed42\last_checkpoint.pth" `
  --dims "64,128,256,512" `
  --depths "2,2,6,2" `
  --head_dropout 0.2 `
  --drop_path 0.0 `
  --t_min 0 `
  --t_max 1 `
  --stage_delays "0.4,0.0,0.0,0.0" `
  --epochs 200 `
  --lr 2e-4 `
  --min_lr 1e-6 `
  --warmup_epochs 10 `
  --weight_decay 0.1 `
  --mixup_alpha 0.2 `
  --early_stopping_patience 30 `
  --download false
```

## 89.3

.venv) PS C:\Users\jafari.h\Desktop\ai_project\ConvNeXt> & "C:\Users\jafari.h\Desktop\ai_project\.venv\Scripts\python.exe" `

> > ".\train_continuous_ttfs_cifar10_32x32_stem1.py" `  --data_path "..\cifar_data"`
> > --output_dir ".\results\cifar10_ttfs_native32_k3_ttfs_stage_delay_seed42" `  --experiment_name "cifar10_ttfs_native32_k3_ttfs_stage_delay_seed42"`
> > --dims "96,192,384,768" `  --depths "2,2,6,2"`
> > --dw_kernel_size 3 `   --stage_delays "0.05,0.02,0.01,0.01"  `
> > --pw2_mode ttfs `  --spike_dropout 0`
> > --head_dropout 0.1 `  --drop_path 0`
> > --t_min 0 `  --t_max 1`
> > --device cuda

## run evaluation

```
& "C:\Users\jafari.h\Desktop\ai_project\.venv\Scripts\python.exe" .\evaluate_ttfs_cifar10_tta.py `
  --checkpoint ".\results\cifar10\fully_ttfs\ttfs_dwconv_downsample\seed_8888\lr2e4\best_checkpoint.pth" `
  --dataset cifar10 `
  --data_path "..\cifar_data" `
  --output_dir ".\results\cifar10\fully_ttfs\evaluation\seed_8888_tta_10view" `
  --device cuda `
  --batch_size 128 `
  --num_workers 4 `
  --tta_modes "none,flip,flip_shift"
```

# run training

```
$seed = 7777

& "C:\Users\jafari.h\Desktop\ai_project\.venv\Scripts\python.exe" `
  ".\train_continuous_ttfs_cifar10_32x32_stem1.py" `
  --dataset cifar10 `
  --data_path "..\cifar_data" `
  --download false `
  --experiment_name "ttfs_dwconv_downsample" `
  --experiment_notes "Analytic TTFS depthwise and downsampling convolutions" `
  --dims "96,192,384,768" `
  --depths "2,2,6,2" `
  --dw_kernel_size 3 `
  --dwconv_mode ttfs `
  --downsample_mode ttfs `
  --stage_delays "0.05,0.02,0.01,0.01" `
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
  --device cuda
```
