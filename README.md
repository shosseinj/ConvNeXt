python .\train_continuous_ttfs_cifar10_32x32_stem1.py `  --data_path "..\cifar_data"`
--output_dir "results\cifar10_continuous_ttfs_32x32_stem1_seed42" `  --epochs 300`
--batch_size 128 `  --lr 0.0005`
--min_lr 0.000001 `  --warmup_epochs 10`
--weight_decay 0.05 `  --label_smoothing 0.1`
--drop_path 0.0 `  --t_min 0.0`
--t_max 1.0 `  --force_positive_weights false`
--init_delay 0.0 `  --stage_delays "0.4,0.0,0.0,0.0"`
--amp true `  --num_workers 4`
--seed 42 `
--download false

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
