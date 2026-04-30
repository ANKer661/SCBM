"""
Run this file to train models using a Hydra configuration, e.g.:
    python train.py +model=SCBM +data=CUB
"""

from pathlib import Path

import hydra
from omegaconf import DictConfig

from training.runner import ExperimentRunner


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(config: DictConfig) -> None:
    project_dir = Path(__file__).absolute().parent
    print("Project directory:", project_dir)
    print("Config:", config)
    ExperimentRunner(config).run()


if __name__ == "__main__":
    main()
