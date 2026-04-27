import json
import argparse
from ssl_neuron.train import Trainer
from ssl_neuron.graphdino import create_model
from ssl_neuron.datasets import build_dataloader

parser = argparse.ArgumentParser()
parser.add_argument('--config', help='Path to config file.', type=str, default='./configs/config.json')
parser.add_argument('--resume', help='Path to checkpoint to resume from.', type=str, default=None)
parser.add_argument('--use-type', dest='use_type', action='store_true',
                    help='Include the 4-dim node-type one-hot in features (feat_dim=7). Overrides config.')
parser.add_argument('--no-use-type', dest='use_type', action='store_false',
                    help='Force xyz-only features (feat_dim=3). Overrides config.')
parser.add_argument('--ckpt-dir', dest='ckpt_dir', type=str, default=None,
                    help='Override trainer.ckpt_dir (use a separate dir to keep old ckpts intact).')
parser.set_defaults(use_type=None)


def main(args):
    # load config
    config = json.load(open(args.config))

    # CLI overrides so old ckpts stay reproducible from their original config.
    if args.use_type is not None:
        config['data']['use_type'] = args.use_type
        config['data']['feat_dim'] = 7 if args.use_type else 3
    if args.ckpt_dir is not None:
        config['trainer']['ckpt_dir'] = args.ckpt_dir
    
    # load data
    print('Loading dataset: {}'.format(config['data']['class']))
    train_loader, val_loader = build_dataloader(config)

    # build model 
    model = create_model(config)
    trainer = Trainer(config, model, [train_loader, val_loader], resume_path=args.resume)

    print('Start training.')
    trainer.train()
    print('Done.')


if __name__ == '__main__':    
    args = parser.parse_args()
    main(args)