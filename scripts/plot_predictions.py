#!/usr/bin/env python3
"""Plot predictions vs labels for a few validation images.

Usage (from repo root):
python scripts/plot_predictions.py --checkpoint model_ckpt_test\checkpoints-best.pth --data_set CIFAR --eval_data_path ../CIFAR-10-images/test --nb_samples 16 --output out.png

This script re-uses the project's `get_args_parser` and `build_dataset` so transforms and dataset handling match training.
"""
import argparse
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision.utils import make_grid
from torchvision.transforms.functional import to_pil_image

import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from main import get_args_parser
from datasets import build_dataset
from timm.models import create_model
import utils
from engine import evaluate





def plot_grid(images, titles, output_path=None):
    # images: list of PIL images or tensors; we'll create a grid and annotate
    if len(images) == 0:
        raise RuntimeError('No images to plot')
    # convert PIL to tensor-like arrays
    tensors = []
    for im in images:
        if isinstance(im, torch.Tensor):
            tensors.append(im.cpu())
        else:
            tensors.append(torch.from_numpy(np.array(im)).permute(2,0,1).float() / 255.0)
    grid = make_grid(tensors, nrow=int(np.sqrt(len(tensors))), padding=2)
    plt.figure(figsize=(12,12))
    npimg = grid.mul(255).byte().permute(1,2,0).numpy()
    plt.imshow(npimg)
    plt.axis('off')
    # titles printed below using subplot grid for clarity
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
        print(f'Saved plot to {output_path}')
    else:
        plt.show()


def main():
    parent = get_args_parser()
    parser = argparse.ArgumentParser(parents=[parent])
    parser.add_argument('--nb_samples', type=int, default=16)
    parser.add_argument('--output', default='predictions.png')
    args = parser.parse_args()

    # force eval-style settings
    args.eval_data_path = getattr(args, 'eval_data_path', None)
    device = torch.device(args.device if hasattr(args, 'device') else 'cpu')

    # build validation dataset using existing helper
    dataset_val, nb_classes = build_dataset(is_train=False, args=args)
    # ensure model uses dataset's number of classes (CIFAR vs ImageNet)
    args.nb_classes = nb_classes

    # small dataloader
    loader = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=0)

    # instantiate model same as main
    if getattr(args, 'spiking', False):
        from models.convnext import ConvNeXtSpiking
        model = ConvNeXtSpiking(in_chans=3, num_classes=args.nb_classes,
                                drop_path_rate=args.drop_path,
                                layer_scale_init_value=args.layer_scale_init_value,
                                head_init_scale=args.head_init_scale,
                                t_min=args.ttfs_tmin, t_max=args.ttfs_tmax)
    else:
        model = create_model(args.model, pretrained=False, num_classes=args.nb_classes,
                             drop_path_rate=args.drop_path, layer_scale_init_value=args.layer_scale_init_value,
                             head_init_scale=args.head_init_scale)

    # load checkpoint using helper that removes mismatched head keys
    if args.load_weights:
                load_path = args.load_weights
                print("Requested to load weights from: %s" % load_path)
                # normalize and strip trailing separators
                load_path = os.path.normpath(load_path)
                # if a directory was provided, try to find the most recent .pth/.pt file inside
                if os.path.isdir(load_path):
                    cand = [os.path.join(load_path, f) for f in os.listdir(load_path)
                            if f.lower().endswith(('.pth', '.pt'))]
                    if len(cand) == 0:
                        raise FileNotFoundError(f"No checkpoint files (.pth/.pt) found in directory: {load_path}")
                    # choose the most recently modified checkpoint
                    load_path = sorted(cand, key=os.path.getmtime)[-1]
                    print(f"Found checkpoint in directory, using: {load_path}")

                if not os.path.isfile(load_path):
                    raise FileNotFoundError(f"Checkpoint file not found: {load_path}")

                checkpoint = torch.load(load_path, map_location='cpu')

                # If the checkpoint is a dict with nested model keys (e.g. {'model': ..., 'optimizer': ...}),
                # extract the actual state_dict using args.model_key (same logic as finetune handling).
                checkpoint_model = None
                for model_key in args.model_key.split('|'):
                    if model_key in checkpoint:
                        checkpoint_model = checkpoint[model_key]
                        print(f"Load state_dict from checkpoint key = {model_key}")
                        break
                if checkpoint_model is None:
                    checkpoint_model = checkpoint

                # If the classifier head shape does not match (e.g., ImageNet -> CIFAR), remove it so load succeeds.
                # state_dict = model.state_dict()
                # for k in ['head.weight', 'head.bias']:
                #     if k in checkpoint_model and checkpoint_model[k].shape != state_dict.get(k, None).shape:
                #         print(f"Removing key {k} from pretrained checkpoint (shape mismatch: {checkpoint_model[k].shape} vs {state_dict.get(k, None)})")
                #         del checkpoint_model[k]

                utils.load_state_dict(model, checkpoint_model, prefix=args.model_prefix)
                # clear load_weights so downstream code does not attempt to reload
                args.load_weights = ''
    model.to(device)
    model.eval()


    if args.eval:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        sampler_val = torch.utils.data.DistributedSampler(
            dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=int(1.5 * args.batch_size),
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False
        )
        print(f"Eval only mode")
        test_stats = evaluate(data_loader_val, model, device, use_amp=args.use_amp)
        print(f"Accuracy of the network on {len(dataset_val)} test images: {test_stats['acc1']:.5f}%")
    

    imgs = []
    titles = []
    probs_list = []
    with torch.no_grad():
        for i, item in enumerate(loader):
            if i >= args.nb_samples:
                break
            # dataset may return (img, label) or (spike_times, orig_img, label)
            if isinstance(item, (list, tuple)) and len(item) == 2:
                img, label = item
                orig_img = None
            elif isinstance(item, (list, tuple)) and len(item) == 3:
                # TTFSWrapper might return (spike_times, orig_img, label)
                img, orig_img, label = item
            else:
                # fallback: try first two
                img, label = item[0], item[1]
                orig_img = None

            inp = img.to(device)
            if inp.dim() == 3:
                inp = inp.unsqueeze(0)

            out = model(inp)
            if isinstance(out, (list, tuple)):
                logits = out[0]
            else:
                logits = out
            pred = logits.argmax(dim=1).item()
            true = int(label.item()) if isinstance(label, torch.Tensor) else int(label)
            # compute probability for predicted class
            soft = torch.nn.functional.softmax(logits, dim=1)
            pred_prob = float(soft[0, pred].cpu().item())

            # prefer original un-normalized image if available
            if orig_img is not None:
                if isinstance(orig_img, torch.Tensor):
                    display_img = to_pil_image(orig_img.cpu())
                else:
                    display_img = orig_img
            else:
                # try to convert input back to an image
                img_t = img.clone()
                if isinstance(img_t, torch.Tensor):
                    # if single sample, ensure C,H,W
                    if img_t.dim() == 4:
                        img_t = img_t[0]
                    # clamp and convert
                    img_t = img_t.clamp(0,1)
                    display_img = to_pil_image(img_t.cpu())
                else:
                    display_img = None

            imgs.append(display_img)
            probs_list.append(pred_prob)
            # obtain class names if available; fall back to CIFAR names or indices
            if hasattr(dataset_val, 'classes'):
                class_names = dataset_val.classes
            else:
                # common fallback for CIFAR
                if getattr(args, 'data_set', '').upper().startswith('CIFAR'):
                    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                                   'dog', 'frog', 'horse', 'ship', 'truck']
                elif hasattr(dataset_val, 'class_to_idx') and isinstance(dataset_val.class_to_idx, dict):
                    # invert mapping idx->name
                    names = [None] * args.nb_classes
                    for name, idx in dataset_val.class_to_idx.items():
                        if 0 <= idx < len(names):
                            names[idx] = name
                    class_names = [n if n is not None else str(i) for i, n in enumerate(names)]
                else:
                    class_names = [str(i) for i in range(args.nb_classes)]

            # safe lookup
            try:
                true_name = class_names[true]
            except Exception:
                true_name = str(true)
            try:
                pred_name = class_names[pred]
            except Exception:
                pred_name = str(pred)
            titles.append(f'True: {true_name} (#{true})\nPred: {pred_name} (#{pred}) {pred_prob:.2f}')

    # annotate titles below image in saved figure by composing a simple grid plot
    # For simplicity we save the grid; titles can be inspected in console as well.
    if len(imgs) == 0:
        print('No images collected')
        return

    # Save grid image
    grid = make_grid([torch.from_numpy(np.array(i)).permute(2,0,1).float()/255.0 if not isinstance(i, torch.Tensor) else i for i in imgs], nrow=int(np.sqrt(len(imgs))), padding=2)
    npimg = grid.mul(255).byte().permute(1,2,0).numpy()
    plt.figure(figsize=(12,12))
    plt.imshow(npimg)
    plt.axis('off')
    # print titles and probs to stdout
    for idx, (t, p) in enumerate(zip(titles, probs_list)):
        print(f'{idx}: {t}  prob={p:.4f}')

    # Create subplot grid with titles overlayed for clearer per-sample info
    n = len(imgs)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.array(axes).reshape(-1)
    for ax in axes[n:]:
        ax.axis('off')
    for i, (im, title) in enumerate(zip(imgs, titles)):
        ax = axes[i]
        if isinstance(im, torch.Tensor):
            img_disp = im
            if img_disp.dim() == 3:
                img_disp = img_disp.cpu()
            else:
                img_disp = img_disp[0].cpu()
            npimg = img_disp.mul(255).byte().permute(1,2,0).numpy()
            ax.imshow(npimg)
        else:
            ax.imshow(im)
        ax.set_title(title, fontsize=14)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(args.output, bbox_inches='tight')
    print(f'Saved predictions grid to {args.output}')


if __name__ == '__main__':
    main()
