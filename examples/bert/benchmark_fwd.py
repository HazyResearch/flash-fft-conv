# Copyright 2022 MosaicML Examples authors
# SPDX-License-Identifier: Apache-2.0

import os
import sys
from typing import Optional, cast

# Add folder root to path to allow us to use relative imports regardless of what directory the script is run from
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

import create_bert as bert_module
from omegaconf import DictConfig
from omegaconf import OmegaConf as om

import torch
from benchmark import benchmark_forward

def build_model(cfg: DictConfig):
    if cfg.name == 'bert':
        return bert_module.create_bert_mlm(
            pretrained_model_name=cfg.pretrained_model_name,
            model_config=cfg.get('model_config', None),
            tokenizer_name=cfg.get('tokenizer_name', None))
    else:
        raise ValueError(f'Not sure how to build model with name={cfg.name}')


def run_bert(model, u, attn_mask):
    encoder_outputs = model.model.bert.encoder(u, attn_mask)
    output = model.model.cls(encoder_outputs[0])
    return output

def main(cfg: DictConfig):
    print('Using config: ')
    print(om.to_yaml(cfg))

    # Build Model
    print('Initializing model...')
    model = build_model(cfg.model).cuda()
    B = cfg.batch_size
    L = cfg.max_seq_len
    dtype = cfg.dtype
    print('Batch size: ', B)
    print('max seq len: ', L)
    if 'hidden_size' not in cfg.model.model_config:
        H = 768
    else:
        H = cfg.model.model_config.hidden_size
    
    u = torch.randn(B, L, H).cuda()
    if dtype == 'half':
        u = u.half()
    if cfg.model.name == 'bert':
        attention_mask = torch.ones(B, L, dtype=torch.int64).cuda()
    else:
        attention_mask = torch.ones(L, L, dtype=torch.int64).cuda()
    
    if dtype == 'half':
        model = model.half()

    run_bert(model, u, attention_mask)
    repeats = 30

    # Run forward pass
    print('Running forward pass...')
    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=dtype == 'half'):
        _, ret = benchmark_forward(run_bert, model, u, attention_mask, repeats=repeats, verbose=True, amp_dtype=torch.float16, amp=dtype == 'half')

        time = ret._mean
        print('Time: ', time)
        print('Tokens/ms: ', B*L/time/1000)
        print('Seqs/s: ', B / time)

if __name__ == '__main__':
    yaml_path, args_list = sys.argv[1], sys.argv[2:]
    with open(yaml_path) as f:
        yaml_cfg = om.load(f)
    cli_cfg = om.from_cli(args_list)
    cfg = om.merge(yaml_cfg, cli_cfg)
    cfg = cast(DictConfig, cfg)  # for type checking
    main(cfg)