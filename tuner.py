import itertools

from utils.config_parser import overwrite_configs


class Tuner(object):
    def __init__(self, cfg):
        self.cfg = cfg

    def has_multiple_configs(self, cfg):
        return any(isinstance(value, list) for value in cfg.values())

    def generate_grid_search_configs(self, cfg):
        list_items = [(key, value) for key, value in cfg.items() if isinstance(value, list)]
        if not list_items:
            print("no-searching")
            return [cfg]
        keys, values = zip(*list_items)
        combinations = [dict(zip(keys, combination)) for combination in itertools.product(*values)]
        return [overwrite_configs(cfg, combination) for combination in combinations]

    def run(self):
        if self.has_multiple_configs:
            return self.generate_grid_search_configs(self.cfg)
        else:
            return [self.cfg]