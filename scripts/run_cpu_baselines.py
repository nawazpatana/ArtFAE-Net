#!/usr/bin/env python3
"""Train frozen-feature baseline heads on CPU with matched budgets and explicit seeds."""
import argparse
import subprocess
import sys
from pathlib import Path
import yaml


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seeds',type=int,nargs='+',default=[42])
    parser.add_argument('--variants',choices=['B3','B4','B5'],nargs='+',default=['B3','B4','B5'])
    args=parser.parse_args()
    if not (args.cache/'metadata.json').is_file():parser.error('Feature cache must be complete')
    if len(args.seeds)!=len(set(args.seeds)):parser.error('Seeds must be distinct')
    args.output.mkdir(parents=True,exist_ok=False)
    configs=args.output/'configs';configs.mkdir()
    repository=Path(__file__).resolve().parents[1]
    for seed in args.seeds:
        for variant in args.variants:
            name=variant if args.seeds==[42] else f'{variant}_seed{seed}'
            config=yaml.safe_load((repository/'configs'/f'{variant}.yaml').read_text());config['seed']=seed
            config_file=configs/f'{name}.yaml';config_file.write_text(yaml.safe_dump(config,sort_keys=False))
            command=[sys.executable,str(repository/'scripts/artfae.py'),'train','--cache',str(args.cache),'--config',str(config_file),'--output',str(args.output/name)]
            with (args.output/f'{name}.log').open('w') as log:
                subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT)
            print(f'{name} completed; {args.output/name}',flush=True)
if __name__=='__main__':main()
