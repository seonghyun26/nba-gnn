from torch_geometric.graphgym.register import register_config
from yacs.config import CfgNode as CN


def set_cfg_wandb(cfg):
    """Weights & Biases tracker configuration.
    """

    # WandB group
    cfg.wandb = CN()

    # Use wandb or not
    cfg.wandb.use = True

    # Wandb entity name, should exist beforehand
    cfg.wandb.entity = ""

    # Wandb project name, will be created in your team if doesn't exist already
    cfg.wandb.project = ""

    # Optional run name
    cfg.wandb.name = ""
    
    # wandb.tags
    cfg.wandb.tags = []


register_config('cfg_wandb', set_cfg_wandb)
