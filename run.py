from agents.fixmatch_trainer import Trainer
from utils import config_parser

from tuner import Tuner


def run(cfg):
    """Call the agents by config"""
    if cfg.action == 'train':
        Trainer(cfg)
    elif cfg.action == 'test':
        pass


def main():
    # parse the config
    cfg = config_parser.get_configs()

    # HPO(Hyperparameter Optimization)
    configs = Tuner(cfg).run()

    print(f"{len(configs)} settings will be run.")
    from argparse import Namespace
    for config in configs:
        # Dict to Namespace.
        config = Namespace(**config)
        run(config)


if __name__ == "__main__":
    main()