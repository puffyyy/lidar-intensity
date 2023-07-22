import glob
import os
import os.path as osp
import sys
import numpy as np
import torch
import torch.utils.data as data

import inten
import otils as ot
import torchutils as tu

def waymo_collate(batch):
    elem = batch[0]
    for key in elem:
        if key == 'key':
            elem['key'] = np.array([elem['key']] * 4)
            continue
        if key == 'raw':
            continue
        raw_val = elem[key]
        left = raw_val[...,329:993]
        center = raw_val[...,993:1657]
        right = raw_val[...,1657:2321]
        back = np.concatenate((raw_val[...,332-1::-1], raw_val[...,-1:-332-1:-1]), axis=-1)
        elem[key] = torch.as_tensor(np.array([center,left,right,back]))
    return elem
if __name__ == '__main__':
    config = ot.io.load_multi_yml(sys.argv[1])
    if 'seed' in config:
        tu.seed_all(config['seed'])
    config['store_dir'] = config['checkpoint']
    checkpoints = sorted([int(item[:-4]) for item in os.listdir(config['checkpoint']) if str(item).endswith('.tar')])
    config['checkpoint'] = osp.join(config['checkpoint'], str(checkpoints[-1]) + '.tar')
    cp = ot.checkpoint.load_checkpoint(osp.join(config['base_dir'], config['store_dir'], config['checkpoint']), False)
    runner = inten.data.EvalWaymoRunner(config)
    val_dataset = data.DataLoader(inten.data.WaymoDataset(config['val']), collate_fn=waymo_collate, **config['val_loader'])
    runner.load_checkpoint(cp)
    runner(val_dataset, osp.join(config['base_dir'], config['store_dir'], config['base_save'], config['val_save']))
