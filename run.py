from argparse import Namespace

from agents.trainer import Trainer
from utils import config_parser
from tuner import Tuner


def create_agent(cfg):
    """Create and initialize an agent based on the action."""
    if cfg.action == 'train':
        return Trainer(cfg)
    elif cfg.action == 'test':
        # Future: Add test agent or other action handlers here
        return None
    return None


def main():
    # Step 1: Load and Parse the config
    raw_configs = config_parser.get_configs()

    # Step 2: Generate agent configs using the Tuner
    agent_configs = Tuner(raw_configs).run()

    # Step 3: Create agents for each configs
    agents = [create_agent(Namespace(**cfg)) for cfg in agent_configs]

    # Step 4: Execute a list of agents sequentially
    # Log the total number of agent configurations to be executed
    print(f"{len(agent_configs)} setting(s) will be run.")
    
    for agent in agents:
        if agent is not None:  # Skip None agents
            agent.run()


if __name__ == "__main__":
    main()