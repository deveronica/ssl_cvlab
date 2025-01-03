import yaml
import argparse
from copy import deepcopy


def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def parse_args():
    parser = argparse.ArgumentParser(description='Deep Learning Model Training and Evaluation')
    

    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to the configuration file')

    parser.add_argument('--type', type=str, default='train',
                        help='Type of the run (train/eval)')
    parser.add_argument('--SEED', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--uri', type=str,
                        help='URI for data storage or logging')
    parser.add_argument('--size', type=int, default=128,
                        help='Image size for training')
    parser.add_argument('--percent', type=float, default=10.0,
                        help='Percentage of data to use')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--arch', type=str, default='resnet18',
                        help='Model architecture')
    parser.add_argument('--epochs', type=int, default=300,
                        help='Number of epochs to train')
    parser.add_argument('--num_classes', type=int, default=10,
                        help='Number of classes')
    parser.add_argument('--num_labeled', type=int, default=10,
                        help='Number of labeled examples')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for training')
    parser.add_argument('--eval_step', type=int, default=30,
                        help='Evaluation step interval')
    parser.add_argument('--expand_labels', action='store_true',
                        help='Expand labels to match the full dataset')
    parser.add_argument('--dataset', type=str, default='MSTAR',
                        help='Dataset name')
    parser.add_argument('--root', type=str, default='./input/MSTAR_10',
                        help='Root directory for the dataset')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')
    parser.add_argument('--mu', type=int, default=8,
                        help='Multiplication factor for some parameter')
    parser.add_argument('--in_channels', type=int, default=1,
                        help='Color Channels')
    
    args = parser.parse_args()
    return vars(args)


def get_configs():
    args = parse_args()
    cfg = load_config(args.get("config", "config.yaml"))
    # cfg = overwrite_configs(cfg, args)
    # 아직 default 구현이 안되어 있어서 무조건 덮어써짐

    return cfg


def overwrite_configs(cfg, args):
    new_config = deepcopy(cfg)
    for key, value in dict(args).items():
        keys = key.split('.')
        ref = new_config
        for k in keys[:-1]:
            ref = ref[k]
        ref[keys[-1]] = value
    return new_config