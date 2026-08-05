#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math, os, random, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from models.convnext import ConvNeXtSpiking

def str2bool(v):
    if isinstance(v,bool): return v
    v=str(v).lower()
    if v in {'1','true','yes','y'}: return True
    if v in {'0','false','no','n'}: return False
    raise argparse.ArgumentTypeError('Boolean expected')

def args_parser():
    p=argparse.ArgumentParser('Continuous TTFS ConvNeXt on native CIFAR-10 32x32')
    p.add_argument('--data_path',default='../cifar_data')
    p.add_argument('--output_dir',default='results/cifar10_continuous_ttfs_32x32_stem1_seed42')
    p.add_argument('--resume',default='')
    p.add_argument('--download',type=str2bool,default=False)
    p.add_argument('--epochs',type=int,default=300)
    p.add_argument('--batch_size',type=int,default=128)
    p.add_argument('--num_workers',type=int,default=4)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--lr',type=float,default=5e-4)
    p.add_argument('--min_lr',type=float,default=1e-6)
    p.add_argument('--warmup_epochs',type=int,default=10)
    p.add_argument('--weight_decay',type=float,default=0.05)
    p.add_argument('--label_smoothing',type=float,default=0.1)
    p.add_argument('--grad_clip',type=float,default=5.0)
    p.add_argument('--drop_path',type=float,default=0.0)
    p.add_argument('--t_min',type=float,default=0.0)
    p.add_argument('--t_max',type=float,default=1.0)
    p.add_argument('--force_positive_weights',type=str2bool,default=False)
    p.add_argument('--init_delay',type=float,default=0.0)
    p.add_argument('--stage_delays',default='0.4,0.0,0.0,0.0')
    p.add_argument('--amp',type=str2bool,default=True)
    p.add_argument('--device',default='cuda')
    p.add_argument('--val_size',type=int,default=5000)
    p.add_argument('--print_freq',type=int,default=50)
    return p.parse_args()

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=True

def build_loaders(a):
    tr=transforms.Compose([transforms.RandomCrop(32,padding=4),transforms.RandomHorizontalFlip(),transforms.ToTensor()])
    ev=transforms.ToTensor()
    dtr=datasets.CIFAR10(a.data_path,train=True,transform=tr,download=a.download)
    dev=datasets.CIFAR10(a.data_path,train=True,transform=ev,download=False)
    dte=datasets.CIFAR10(a.data_path,train=False,transform=ev,download=a.download)
    g=torch.Generator().manual_seed(a.seed)
    idx=torch.randperm(len(dtr),generator=g).tolist()
    va=idx[:a.val_size]; tridx=idx[a.val_size:]
    common=dict(num_workers=a.num_workers,pin_memory=torch.cuda.is_available(),persistent_workers=a.num_workers>0)
    return (
        DataLoader(Subset(dtr,tridx),batch_size=a.batch_size,shuffle=True,**common),
        DataLoader(Subset(dev,va),batch_size=a.batch_size,shuffle=False,**common),
        DataLoader(dte,batch_size=a.batch_size,shuffle=False,**common),
    )

def make_model(a):
    delays=[float(x) for x in a.stage_delays.split(',')]
    if len(delays)!=4: raise ValueError('stage_delays must have 4 values')
    m=ConvNeXtSpiking(in_chans=3,num_classes=10,depths=(3,3,9,3),dims=(96,192,384,768),drop_path_rate=a.drop_path,t_min=a.t_min,t_max=a.t_max,force_positive_weights=a.force_positive_weights,init_delay=a.init_delay,stage_delays=delays)
    stem=nn.Conv2d(3,96,kernel_size=3,stride=1,padding=1,bias=True)
    nn.init.trunc_normal_(stem.weight,std=.02); nn.init.zeros_(stem.bias)
    m.downsample_layers[0]=nn.Sequential(stem) if isinstance(m.downsample_layers[0],nn.Sequential) else stem
    return m

def encode(x,a):
    if x.min().item() < -1e-6 or x.max().item() > 1+1e-6: raise ValueError('Input must be in [0,1]')
    return a.t_min + (1-x)*(a.t_max-a.t_min)

def lr_at(e,a):
    if e<a.warmup_epochs: return a.lr*(e+1)/max(1,a.warmup_epochs)
    p=(e-a.warmup_epochs)/max(1,a.epochs-a.warmup_epochs-1)
    return a.min_lr+.5*(a.lr-a.min_lr)*(1+math.cos(math.pi*min(max(p,0),1)))

def run_epoch(m,loader,crit,dev,a,opt=None,scaler=None):
    train=opt is not None; m.train(train)
    total=correct=0; loss_sum=0.; st=time.time()
    for i,(x,y) in enumerate(loader):
        x=x.to(dev,non_blocking=True); y=y.to(dev,non_blocking=True); x=encode(x,a)
        if train: opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=dev.type,dtype=torch.float16,enabled=a.amp and dev.type=='cuda'):
            out=m(x); loss=crit(out,y)
        if not torch.isfinite(loss): raise FloatingPointError('non-finite loss')
        if train:
            scaler.scale(loss).backward() if scaler.is_enabled() else loss.backward()
            if scaler.is_enabled(): scaler.unscale_(opt)
            if a.grad_clip>0: nn.utils.clip_grad_norm_(m.parameters(),a.grad_clip)
            if scaler.is_enabled(): scaler.step(opt); scaler.update()
            else: opt.step()
        bs=y.size(0); total+=bs; loss_sum+=loss.item()*bs; correct+=(out.argmax(1)==y).sum().item()
        if (i+1)%a.print_freq==0: print(json.dumps({'phase':'train' if train else 'validation','iteration':i+1,'loss':loss_sum/total,'accuracy':100*correct/total}),flush=True)
    return {'loss':loss_sum/max(total,1),'accuracy':100*correct/max(total,1),'samples':total,'seconds':time.time()-st}

def save(path,m,opt,scaler,epoch,best,a):
    tmp=path.with_suffix(path.suffix+'.tmp')
    torch.save({'model':m.state_dict(),'optimizer':opt.state_dict(),'scaler':scaler.state_dict(),'epoch':epoch,'best_val_accuracy':best,'args':vars(a)},tmp)
    os.replace(tmp,path)

def main():
    a=args_parser(); seed_all(a.seed)
    outdir=Path(a.output_dir); outdir.mkdir(parents=True,exist_ok=True)
    dev=torch.device(a.device if a.device.startswith('cuda') and torch.cuda.is_available() else 'cpu')
    train_loader,val_loader,test_loader=build_loaders(a)
    m=make_model(a).to(dev)
    # verify runtime sizes
    shapes={}; hs=[]
    for i,l in enumerate(m.downsample_layers):
        hs.append(l.register_forward_hook(lambda mod,inp,out,i=i: shapes.__setitem__(i,tuple(out.shape))))
    with torch.no_grad(): _=m(encode(torch.rand(1,3,32,32,device=dev),a))
    for h in hs: h.remove()
    print('Runtime downsample shapes:',shapes)
    assert shapes[0]==(1,96,32,32) and shapes[1]==(1,192,16,16) and shapes[2]==(1,384,8,8) and shapes[3]==(1,768,4,4)
    crit=nn.CrossEntropyLoss(label_smoothing=a.label_smoothing)
    opt=torch.optim.AdamW(m.parameters(),lr=a.lr,weight_decay=a.weight_decay)
    scaler=torch.amp.GradScaler('cuda',enabled=a.amp and dev.type=='cuda')
    start=0; best=-1.
    if a.resume:
        ck=torch.load(a.resume,map_location='cpu',weights_only=False); m.load_state_dict(ck['model'],strict=True); opt.load_state_dict(ck['optimizer']); scaler.load_state_dict(ck['scaler']); start=ck['epoch']+1; best=ck.get('best_val_accuracy',-1.)
    cfg={**vars(a),'input_resolution':[32,32],'stem':{'kernel_size':3,'stride':1,'padding':1,'out_channels':96},'spatial_schedule':[32,32,16,8,4],'temporal_formulation':'continuous analytic TTFS','simulation_steps':None}
    (outdir/'config.json').write_text(json.dumps(cfg,indent=2),encoding='utf-8')
    for e in range(start,a.epochs):
        lr=lr_at(e,a)
        for g in opt.param_groups: g['lr']=lr
        tr=run_epoch(m,train_loader,crit,dev,a,opt,scaler)
        with torch.inference_mode(): va=run_epoch(m,val_loader,crit,dev,a)
        row={'epoch':e,'learning_rate':lr,**{f'train_{k}':v for k,v in tr.items()},**{f'val_{k}':v for k,v in va.items()}}
        print(json.dumps(row),flush=True)
        with (outdir/'train_log.jsonl').open('a',encoding='utf-8') as f: f.write(json.dumps(row)+'\n')
        save(outdir/'last_checkpoint.pth',m,opt,scaler,e,best,a)
        if va['accuracy']>best:
            best=va['accuracy']; save(outdir/'best_checkpoint.pth',m,opt,scaler,e,best,a)
    ck=torch.load(outdir/'best_checkpoint.pth',map_location=dev,weights_only=False); m.load_state_dict(ck['model'],strict=True)
    with torch.inference_mode(): te=run_epoch(m,test_loader,crit,dev,a)
    summary={'best_epoch':ck['epoch'],'best_validation_accuracy':best,'test_metrics':te,'best_checkpoint':str(outdir/'best_checkpoint.pth')}
    (outdir/'training_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8'); print(json.dumps(summary,indent=2))

if __name__=='__main__': main()
