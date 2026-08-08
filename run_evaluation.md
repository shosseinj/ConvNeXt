## Evaluation command

```
& "C:/Users/jafari.h/Desktop/ai_project/.venv/Scripts/python.exe" `
  ".\evaluate_ttfs_cifar10_tta.py" `
  --checkpoint ".\results\cifar10\fully_ttfs\clean_finetune_from_94_36\seed_42\preserved_best_94_40\best_checkpoint.pth" `
  --data_path "..\cifar_data" `
  --output_dir ".\results\cifar10\fully_ttfs\evaluation\tta_10view" `
  --weights_source auto `
  --tta_modes "none,flip,flip_shift" `
  --batch_size 256 `
  --num_workers 4 `
  --amp true `
  --device cuda
```

```

& "C:/Users/jafari.h/Desktop/ai_project/.venv/Scripts/python.exe" `
  ".\evaluate_ttfs_cifar10_tta.py" `
  --checkpoint ".\results\seed_42\best_checkpoint.pth" `
  --checkpoint ".\results\seed_43\best_checkpoint.pth" `
  --checkpoint ".\results\seed_44\best_checkpoint.pth" `
  --data_path "..\cifar_data" `
  --output_dir ".\results\cifar10\fully_ttfs\evaluation\three_seed_ensemble_tta" `
  --weights_source auto `
  --tta_modes "none,flip,flip_shift" `
  --batch_size 128 `
  --num_workers 4 `
  --amp true `
  --device cuda
```
