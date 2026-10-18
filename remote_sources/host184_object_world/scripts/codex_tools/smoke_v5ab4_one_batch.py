#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os, sys, time, traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from mmengine.config import Config
from mmengine.registry import init_default_scope
from mmengine.runner import Runner
from mmdet3d.registry import MODELS


def _loss_to_float(x):
    if torch.is_tensor(x):
        return float(x.detach().mean().cpu())
    if isinstance(x, (list, tuple)):
        vals=[_loss_to_float(v) for v in x]
        return float(sum(vals)) if vals else 0.0
    return float(x) if isinstance(x,(int,float)) else None


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--max-tries', type=int, default=20)
    ap.add_argument('--summary', required=True)
    ap.add_argument('--no-loss', action='store_true', help='only preprocess and build DINO FPN for frame 0')
    args=ap.parse_args()
    cfg=Config.fromfile(args.config)
    cfg.train_dataloader.batch_size=1
    cfg.train_dataloader.num_workers=0
    cfg.train_dataloader.persistent_workers=False
    cfg.model.dino_debug=True
    init_default_scope('mmdet3d')
    dataloader=Runner.build_dataloader(cfg.train_dataloader)
    model=MODELS.build(cfg.model)
    model.to(args.device)
    model.train()
    result={'config':args.config,'device':args.device,'max_tries':args.max_tries,'attempts':[]}
    t0=time.time()
    ok=False
    data_iter = iter(dataloader)
    for i in range(args.max_tries):
        attempt={'iter':i}
        try:
            batch = next(data_iter)
            data=model.data_preprocessor(batch, training=True)
            inputs=data['inputs']
            attempt['input_keys']=list(inputs.keys())
            attempt['num_points_batches']=len(inputs.get('points', []))
            if 'dino_feat_maps' in inputs:
                df=inputs['dino_feat_maps']
                attempt['dino_layout']='frame_major' if isinstance(df,list) and df and isinstance(df[0],list) else str(type(df))
                try:
                    attempt['dino_shape0']=tuple(df[0][0].shape)
                except Exception:
                    pass
            if args.no_loss:
                with torch.no_grad():
                    fpn=model._build_dino_fpn_cached(inputs, data['data_samples'], frame_i=0)
                attempt['fpn_shapes']=[tuple(x.shape) for x in fpn] if fpn is not None else None
            else:
                with torch.no_grad():
                    losses=model(**data, mode='loss')
                attempt['loss_keys']=list(losses.keys())
                attempt['losses']={k:_loss_to_float(v) for k,v in losses.items()}
            attempt['last_dino_valid_rate']=getattr(model,'_last_dino_valid_rate',None)
            attempt['last_dino_valid_rate_per_sample']=getattr(model,'_last_dino_valid_rate_per_sample',None)
            result['attempts'].append(attempt)
            ok=True
            break
        except Exception as exc:
            attempt['error']=repr(exc)
            attempt['traceback']=traceback.format_exc(limit=6)
            result['attempts'].append(attempt)
            if 'DINO cache' not in repr(exc) and 'FileNotFoundError' not in repr(exc):
                break
    result['ok']=ok
    result['elapsed_sec']=round(time.time()-t0,3)
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text(json.dumps(result, indent=2, ensure_ascii=False))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if not ok:
        raise SystemExit('smoke failed')

if __name__ == '__main__':
    main()
