"""
Usage:
Training:
python train.py --config-name=train_diffusion_lowdim_workspace
"""

import sys
import torch
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import hydra
from omegaconf import OmegaConf
import pathlib
from diffusion_policy.workspace.base_workspace import BaseWorkspace

# allows arbitrary python code execution in configs using the ${eval:''} resolver
OmegaConf.register_new_resolver("eval", eval, replace=True)

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'diffusion_policy','config'))
)
def main(cfg: OmegaConf):
    # resolve immediately so all the ${now:} resolvers
    # will use the same time.
    OmegaConf.resolve(cfg)

    # Adjust DataLoader parameters
    if hasattr(cfg, 'data_loader'):
        cfg.data_loader.num_workers = 4  # Set number of workers to a manageable level
        cfg.data_loader.batch_size = 1   # Adjust batch size based on your GPU's memory
        print("After modification:", OmegaConf.to_yaml(cfg.data_loader))  # Debug line
    
    # Force to use CPU if you encounter issues
    device = torch.device('cpu')  # Use 'cuda' if you switch to a system with NVIDIA

    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    workspace.run()

    workspace.run(device=device)

if __name__ == "__main__":
    main()
