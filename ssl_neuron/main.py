"""GraphDINO training entry point.

Pick a compartment variant and go (run from the repo root):

    python -m ssl_neuron.main --variant full
    python -m ssl_neuron.main --variant dendrite
    python -m ssl_neuron.main --variant axon

`--variant X` is shorthand for `--config ssl_neuron/configs/X.json`; the config
already carries the variant's dataset path and checkpoint directory.
"""

import argparse
import json
import os
from pathlib import Path

from ssl_neuron.datasets import build_dataloader
from ssl_neuron.graphdino import create_model
from ssl_neuron.train import Trainer

CONFIG_DIR = Path(__file__).resolve().parent / 'configs'

parser = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument('--variant', type=str, default=None,
                    help='compartment variant: full | dendrite | axon')
parser.add_argument('--config', help='Path to config file (overrides --variant).', type=str,
                    default=None)
parser.add_argument('--resume', help='Path to checkpoint to resume from.', type=str, default=None)
parser.add_argument('--use-type', dest='use_type', action='store_true',
                    help='Include the 4-dim node-type one-hot in features (feat_dim=7). Overrides config.')
parser.add_argument('--no-use-type', dest='use_type', action='store_false',
                    help='Force xyz-only features (feat_dim=3). Overrides config.')
parser.add_argument('--ckpt-dir', dest='ckpt_dir', type=str, default=None,
                    help='Override trainer.ckpt_dir (use a separate dir to keep old ckpts intact).')
parser.add_argument('--data-path', dest='data_path', type=str, default=None,
                    help='Override data.path (directory with skeletons/ and the id splits).')
parser.add_argument('--max-iter', dest='max_iter', type=int, default=None,
                    help='Override optimizer.max_iter.')
parser.set_defaults(use_type=None)


def resolve_config_path(args):
    if args.config:
        return Path(args.config)
    if args.variant:
        path = CONFIG_DIR / f'{args.variant}.json'
        if not path.exists():
            raise SystemExit(f'no config for variant {args.variant!r} at {path}')
        return path
    raise SystemExit('pass --variant (full | dendrite | axon) or --config')


def main(args):
    config_path = resolve_config_path(args)
    config = json.load(open(config_path))

    # CLI overrides so old ckpts stay reproducible from their original config.
    if args.use_type is not None:
        config['data']['use_type'] = args.use_type
        config['data']['feat_dim'] = 7 if args.use_type else 3
    if args.ckpt_dir is not None:
        config['trainer']['ckpt_dir'] = args.ckpt_dir
    if args.data_path is not None:
        config['data']['path'] = args.data_path
    if args.max_iter is not None:
        config['optimizer']['max_iter'] = args.max_iter

    os.makedirs(config['trainer']['ckpt_dir'], exist_ok=True)

    print(f'Config:     {config_path}')
    print(f'Variant:    {config.get("variant", "?")} — {config.get("description", "")}')
    print(f'Dataset:    {config["data"]["path"]}')
    print(f'Checkpoints:{config["trainer"]["ckpt_dir"]}')
    print(f'use_type:   {config["data"].get("use_type", False)} '
          f'(feat_dim={config["data"]["feat_dim"]})')

    print('Loading dataset: {}'.format(config['data']['class']))
    train_loader, val_loader = build_dataloader(config)

    model = create_model(config)
    trainer = Trainer(config, model, [train_loader, val_loader], resume_path=args.resume)

    print('Start training.')
    trainer.train()
    print('Done.')


if __name__ == '__main__':
    main(parser.parse_args())
