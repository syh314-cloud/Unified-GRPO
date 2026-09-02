"""Paper Stage-3 DriveLaW-Act training with the project's video-loss extension.

This entrypoint keeps the official ``UnifiedTrainer`` setup through
``JointVideoActionTrainer`` while requiring a Stage-2 DriveLaW-Video
initialization and full Video/Action DiT optimization.
"""

import os

import hydra
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

from navsim.agents.videodrive.run_training_joint_video_action import (
    JointVideoActionTrainer,
)
from navsim.agents.videodrive.run_training_videodrive import CONFIG_NAME, CONFIG_PATH


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    config_file = cfg.agent.config_file
    if not os.path.isabs(config_file):
        config_file = os.path.join(get_original_cwd(), config_file)

    trainer = JointVideoActionTrainer(cfg, config_file)
    if trainer.args.train_mode not in ("all", "action_full"):
        raise ValueError(
            "Stage-3 joint training requires train_mode: all (or action_full); "
            "do not use the frozen video_action_adapt mode."
        )

    trainer.prepare_dataset()
    if trainer.args.enable_val:
        trainer.prepare_val_dataset()
    trainer.prepare_models()
    trainer.prepare_trainable_parameters()
    frozen_parameters = [
        name for name, parameter in trainer.diffusion_model.named_parameters() if not parameter.requires_grad
    ]
    if frozen_parameters:
        preview = ", ".join(frozen_parameters[:5])
        raise RuntimeError(
            "Paper Stage-3 joint training requires every Video/Action DiT parameter "
            f"to be trainable; found frozen parameters: {preview}"
        )
    trainer.prepare_for_training()
    trainer.prepare_trackers()
    trainer.train()


if __name__ == "__main__":
    main()
