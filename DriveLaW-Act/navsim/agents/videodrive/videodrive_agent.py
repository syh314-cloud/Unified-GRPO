#    Copyright (C) 2026 Xiaomi Corporation.

#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at

#        http://www.apache.org/licenses/LICENSE-2.0

#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
from typing import Any, List, Dict, Optional, Union
import os
import torch
from torch.optim import Optimizer
import torch.optim as optim
from torch.optim.lr_scheduler import LRScheduler
from omegaconf import DictConfig, OmegaConf
from transformers.feature_extraction_utils import BatchFeature
import math
import argparse
import numpy as np
from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import AgentInput, SensorConfig, Trajectory
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from .utils.internvl_preprocess import load_image
from .utils.lr_scheduler import WarmupCosLR
from .utils.utils import format_number, build_from_configs
from .videodrive_features import VideoDriveFeatureBuilder ,TrajectoryTargetBuilder
from yaml import load, dump, Loader, Dumper

from PIL import Image
from einops import rearrange

from .utils.model_utils import load_condition_models, load_latent_models, load_vae_models, load_diffusion_model, count_model_parameters, unwrap_model
from .utils.model_loader import load_vae
from .idm_xvo_style import (
    build_drivelaw_idm_clip,
    compute_idm_consistency_reward,
    compute_musiq_reward,
)
# ----------------------------------------------------
from torch.utils.tensorboard import SummaryWriter
from .utils import init_logging, import_custom_class, save_video
from .utils.data_utils import get_latents, get_text_conditions, gen_noise_from_condition_frame_latent, randn_tensor, apply_color_jitter_to_video
from .video_models.pipeline.pipeline_ltx_condition import LTXVideoCondition
from diffusers.utils import export_to_video, load_video

import os
import uuid
import time

class VideoDriveAgent(AbstractAgent):
    def __init__(
        self,
        trajectory_sampling: TrajectorySampling,
        config_file,
        weight_dtype,
        device,
        view_mode: str = None,  # None: auto-detect; "front" for single view, "surround6" for multi-view
    ):
        super().__init__()
        self._trajectory_sampling = trajectory_sampling
        cd = load(open(config_file, "r"), Loader=Loader)
        args = argparse.Namespace(**cd)
        self.args = args
        
        # Auto-detect view_mode (priority: explicit arg > config_file > path-based inference).
        if view_mode is None:
            if hasattr(args, 'view_mode') and args.view_mode:
                view_mode = args.view_mode
            elif 'multi_view' in str(config_file).lower() or 'multiview' in str(config_file).lower():
                view_mode = "surround6"
            else:
                view_mode = "front"
        
        # Tokenizers
        self.tokenizer = None

        # Text encoders
        self.text_encoder = None

        # Denoisers
        self.diffusion_model = None
        self.unet = None

        # Autoencoders
        self.vae = None

        # Scheduler
        self.scheduler = None

        self.pipe = None
        self.musiq_metric = None

        self.weight_dtype = torch.bfloat16
        self.device = "cuda" # device
        

        self.last_inference_timing = {}
        
        self.view_mode = view_mode
        print(f"[VideoDriveAgent] Using view_mode: {view_mode} (detected from config_file: {config_file})")


    def name(self) -> str:
        return self.__class__.__name__

    def _get_musiq_metric(self) -> torch.nn.Module:
        if self.musiq_metric is None:
            try:
                import pyiqa
            except ImportError as exc:
                raise ImportError(
                    "PyIQA is required for Reward 3. Install it with `pip install pyiqa`."
                ) from exc
            self.musiq_metric = pyiqa.create_metric("musiq").to(self.device)
            self.musiq_metric.eval()
        return self.musiq_metric

    def initialize(self) -> None:
        print("Initializing models")
        device = self.device
        dtype = self.weight_dtype

        ### Load Tokenizer
        tokenizer_class = import_custom_class(
            self.args.tokenizer_class, getattr(self.args, "tokenizer_class_path", "transformers")
        )
        textenc_class = import_custom_class(
            self.args.textenc_class, getattr(self.args, "textenc_class_path", "transformers")
        )
        cond_models = load_condition_models(
            tokenizer_class, textenc_class,
            self.args.pretrained_model_name_or_path if not hasattr(self.args, "tokenizer_pretrained_model_name_or_path") else self.args.tokenizer_pretrained_model_name_or_path,
            load_weights=self.args.load_weights
        )
        self.tokenizer, text_encoder = cond_models["tokenizer"], cond_models["text_encoder"]
        self.text_encoder = text_encoder.to(device, dtype=dtype).eval()

        ### Load VAE
        vae_class = import_custom_class(
            self.args.vae_class, getattr(self.args, "vae_class_path", "transformers")
        )
        # if getattr(self.args, 'vae_path', False):
        #     self.vae = load_vae_models(vae_class, self.args.vae_path).to(device, dtype=dtype).eval()
        # else:
        #     self.vae = load_latent_models(vae_class, self.args.pretrained_model_name_or_path)["vae"].to(device, dtype=dtype).eval()
        self.vae = load_vae(self.args.pretrained_model_name_or_path).eval()

        if isinstance(self.vae.latents_mean, List):
            self.vae.latents_mean = torch.FloatTensor(self.vae.latents_mean)
        if isinstance(self.vae.latents_std, List):
            self.vae.latents_std = torch.FloatTensor(self.vae.latents_std)
        if self.vae is not None:
            if self.args.enable_slicing:
                self.vae.enable_slicing()
            if self.args.enable_tiling:
                self.vae.enable_tiling()
        self.SPATIAL_DOWN_RATIO = self.vae.spatial_compression_ratio
        self.TEMPORAL_DOWN_RATIO = self.vae.temporal_compression_ratio
        print(f'SPATIAL_DOWN_RATIO of VAE :{self.SPATIAL_DOWN_RATIO}')
        print(f'TEMPORAL_DOWN_RATIO of VAE :{self.TEMPORAL_DOWN_RATIO}')


        ### Load Diffusion Model
        diffusion_model_class = import_custom_class(
            self.args.diffusion_model_class, getattr(self.args, "diffusion_model_class_path", "transformers")
        )
        self.diffusion_model = load_diffusion_model(
            model_cls=diffusion_model_class,
            model_dir=self.args.diffusion_model['model_path'],
            load_weights=self.args.load_weights and getattr(self.args, "load_diffusion_model_weights", True),
            **self.args.diffusion_model['config']
        ).to(device, dtype=dtype)
        total_params = count_model_parameters(self.diffusion_model)
        print(f'Total parameters for transformer model:{total_params}')


        ### Load Diffuser Scheduler
        diffusion_scheduler_class = import_custom_class(
            self.args.diffusion_scheduler_class, getattr(self.args, "diffusion_scheduler_class_path", "diffusers")
        )
        if hasattr(self.args, "diffusion_scheduler_args"):
            self.scheduler = diffusion_scheduler_class(**self.args.diffusion_scheduler_args)
        else:
            self.scheduler = diffusion_scheduler_class()

        ### Import Inference Pipeline Class
        self.pipeline_class = import_custom_class(
            self.args.pipeline_class, getattr(self.args, "pipeline_class_path", "diffusers")
        )

        self.pipe = self.pipeline_class(
            self.scheduler, self.vae, self.text_encoder, self.tokenizer, self.diffusion_model
        )
        self.pipe.to(device)
        self.pipe.vae.enable_tiling()
        self.pipe.vae.enable_slicing()

        self.vae_compression = self.pipe.vae_spatial_compression_ratio

    def round_to_vae_resolution(self, height, width):
        """Adjust resolution to be compatible with the VAE compression ratio."""
        height = height - (height % self.vae_compression)
        width = width - (width % self.vae_compression)
        return height, width
        
    def get_sensor_config(self) -> SensorConfig:
        return SensorConfig.build_all_sensors(include=[0, 1, 2, 3,4,5,6,7,8,9,10,11])

    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        if self.view_mode == "front":
            return [TrajectoryTargetBuilder(
                trajectory_sampling=self._trajectory_sampling,
                view_mode="front",
            )]
        else:
            return [TrajectoryTargetBuilder(
                trajectory_sampling=self._trajectory_sampling,
                view_mode="surround6",
                surround_keys=(
                    "cam_b0",  # back
                    "cam_l2",  # back-left
                    "cam_l0",  # front-left
                    "cam_f0",  # front
                    "cam_r0",  # front-right
                    "cam_r2",  # back-right
                ),
            )]

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        if self.view_mode == "front":
            return [VideoDriveFeatureBuilder(
                view_mode="front",
            )]
        else:
            return [VideoDriveFeatureBuilder(
                view_mode="surround6",
                surround_keys=(
                    "cam_b0",  # back
                    "cam_l2",  # back-left
                    "cam_l0",  # front-left
                    "cam_f0",  # front
                    "cam_r0",  # front-right
                    "cam_r2",  # back-right
                ),
            )]

    def forward(self, features: Dict[str, torch.Tensor], targets=None) -> Dict[str, torch.Tensor]:
        if self.training:
            return self.forward_train(features, targets)
        else:
            return self.forward_test(features)

    def forward_train(self, features: Dict[str, torch.Tensor], targets=None) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    def _pad_to_8n1(self, x_5d: torch.Tensor) -> torch.Tensor:
        T = x_5d.shape[2]
        need = (1 - (T % 8)) % 8  # 使得 (T + need) % 8 == 1
        if need > 0:
            tail = x_5d[:, :, -1:, :, :].repeat(1, 1, need, 1, 1)
            x_5d = torch.cat([x_5d, tail], dim=2)
        return x_5d

    @staticmethod
    def _slice_feature_sample(features: Dict[str, torch.Tensor], index: int) -> Dict[str, torch.Tensor]:
        sample = {}
        for key, value in features.items():
            if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] > index:
                sample[key] = value[index:index + 1]
            else:
                sample[key] = value
        return sample

    def _prepare_joint_infer_inputs(self, features: Dict[str, torch.Tensor]):
        device = self.device
        args = self.args
        timing = {}

        if "images" not in features:
            raise ValueError("forward_test expects features['images'] = (T,C,H,W) conditioning frames.")
        cond_tensor = features["images"]

        if cond_tensor.ndim == 5:
            if cond_tensor.shape[0] != 1:
                raise ValueError(
                    "sample_joint_rollouts expects a single sample; "
                    "use `sample_joint_rollouts_batched` for batched features."
                )
            cond_tensor = cond_tensor.squeeze(0)
        elif cond_tensor.ndim == 3:
            cond_tensor = cond_tensor.unsqueeze(0)
        elif cond_tensor.ndim != 4:
            raise ValueError(f"Expected cond_tensor (T,C,H,W), got shape {cond_tensor.shape}")

        t_cond, _, _, _ = cond_tensor.shape
        assert t_cond == 4
        weight_dtype = torch.bfloat16

        context_dict = {
            "cmd_onehot": features["driving_command"].to(device, dtype=weight_dtype),
            "vel": features["vel"].to(device, dtype=weight_dtype),
            "acc": features["acc"].to(device, dtype=weight_dtype),
        }

        prompt = (
            "A high-quality, photorealistic dashboard camera view of autonomous driving. "
            "Based on the past 2 seconds videos, "
            "predict and generate the next 4 seconds of realistic driving continuation, "
            "Maintain temporal consistency, stable camera perspective, natural motion flow without jitter or artifacts, "
            "clear details, and realistic physics. "
        )

        torch.cuda.synchronize()
        t0 = time.perf_counter()
        cond_pils: List[Image.Image] = self._tensor_video_to_pils(cond_tensor)
        torch.cuda.synchronize()
        timing["image_preprocessing"] = time.perf_counter() - t0

        # Generate at the model's native temporal rate, then explicitly
        # resample to the IDM waypoint timestamps below.
        video_fps = int(getattr(args, "frame_rate", 8))
        condition_frames = int(getattr(args, "condition_frames", t_cond))
        num_waypoints = int(getattr(args, "num_waypoints", 8))
        waypoint_interval_s = float(self._trajectory_sampling.interval_length)
        required_future_frames = int(math.ceil(num_waypoints * waypoint_interval_s * video_fps))
        num_frames = int(
            getattr(args, "num_frames", condition_frames + required_future_frames + 1)
        )
        num_steps = int(getattr(args, "num_inference_step", 20))
        seed = int(getattr(args, "seed", 42))
        negative_prompt = getattr(
            args,
            "negative_prompt",
            "worst quality, low quality, blurry, jittery, distorted, motion blur, ghosting, "
            "flickering, stuttering, camera shake, unstable footage, warping, trailing artifacts, "
            "temporal inconsistency, jerky motion, choppy framerate",
        )

        action_chunk = int(getattr(args, "action_chunk", 8))
        action_dim = int(getattr(args, "action_dim", 3))

        if self.view_mode == "surround6":
            height, width = 256, 3072
        else:
            height, width = 768, 1344
            if hasattr(args, "height") and args.height:
                height = int(args.height)
            if hasattr(args, "width") and args.width:
                width = int(args.width)

        condition = LTXVideoCondition(video=cond_pils, frame_index=0)
        return {
            "cond_tensor": cond_tensor,
            "context_dict": context_dict,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "num_frames": num_frames,
            "video_fps": video_fps,
            "condition_frames": condition_frames,
            "num_waypoints": num_waypoints,
            "waypoint_interval_s": waypoint_interval_s,
            "num_steps": num_steps,
            "seed": seed,
            "action_chunk": action_chunk,
            "action_dim": action_dim,
            "height": height,
            "width": width,
            "condition": condition,
            "timing": timing,
        }

    def _postprocess_joint_rollout_outputs(
        self,
        cond_tensor: torch.Tensor,
        preds: Dict[str, torch.Tensor],
        video_fps: int,
        condition_frames: int,
        num_waypoints: int,
        waypoint_interval_s: float,
    ) -> Dict[str, object]:
        actions = preds["actions"]
        generated_video = preds["videos"]

        final_action_clip_value = 1.0
        if final_action_clip_value is not None:
            actions = actions.clone()
            actions.clamp_(-final_action_clip_value, final_action_clip_value)
        actions = denorm_odo(actions).detach().float().cpu()  # (G, action_chunk, action_dim)

        if not torch.is_tensor(generated_video):
            raise TypeError(f"Expected generated video to be a tensor, got {type(generated_video)}")
        generated_video = generated_video.detach().float().cpu()  # (G, F, C, H, W)
        if generated_video.ndim != 5:
            raise ValueError(f"Expected generated video shape (B,F,C,H,W), got {tuple(generated_video.shape)}")

        if video_fps <= 0:
            raise ValueError(f"video_fps must be positive, got {video_fps}")
        if condition_frames < 0 or condition_frames >= generated_video.shape[1]:
            raise ValueError(
                f"condition_frames={condition_frames} is invalid for generated video with "
                f"{generated_video.shape[1]} frames"
            )

        # Remove the conditioning prefix. The remaining frames are ordered at
        # video_fps, while IDM waypoints are ordered every waypoint_interval_s.
        future_video = generated_video[:, condition_frames:].contiguous()
        target_times = torch.arange(1, num_waypoints + 1, dtype=torch.float32) * waypoint_interval_s
        # Frame 0 is the first future sample, so t seconds maps to ceil(t*fps)-1.
        target_indices = torch.ceil(target_times * video_fps).long() - 1
        max_index = int(target_indices.max().item())
        if future_video.shape[1] <= max_index:
            raise ValueError(
                "Generated video is too short for IDM temporal alignment: "
                f"need {max_index + 1} future frames at {video_fps} FPS for "
                f"{num_waypoints} waypoints every {waypoint_interval_s:.3f}s, "
                f"but got {future_video.shape[1]}. Increase num_frames."
            )
        generated_future_video = future_video[:, target_indices].contiguous()

        batch_size = generated_future_video.shape[0]
        history_video = cond_tensor.unsqueeze(0).repeat(batch_size, 1, 1, 1, 1).detach().float().cpu()
        generated_future_video_bcthw = generated_future_video.permute(0, 2, 1, 3, 4).contiguous()
        idm_clip, idm_meta = build_drivelaw_idm_clip(
            history_video=history_video,
            future_video=generated_future_video,
            future_interval_s=self._trajectory_sampling.interval_length,
        )

        return {
            "actions_tensor": actions,
            "actions": actions.numpy(),
            "generated_video": generated_video,
            "generated_future_video": generated_future_video,
            "generated_future_video_bcthw": generated_future_video_bcthw,
            "idm_clip": idm_clip,
            "idm_meta": idm_meta,
        }

    @torch.inference_mode()
    def sample_joint_rollouts(
        self,
        features: Dict[str, torch.Tensor],
        sample_time: int = 8,
        seed: Optional[int] = None,
    ) -> Dict[str, object]:
        """
        Jointly sample `sample_time` action/video rollouts from the same conditioning input.

        This is the GRPO-ready entry point for scheme A:
          one pipe.infer call -> aligned actions + generated videos + IDM clips
        """
        device = self.device
        infer_inputs = self._prepare_joint_infer_inputs(features)
        timing = infer_inputs["timing"]

        generator_seed = infer_inputs["seed"] if seed is None else int(seed)
        generator = torch.Generator(device=device).manual_seed(generator_seed)

        torch.cuda.synchronize()
        t1 = time.perf_counter()
        out = self.pipe.infer(
            conditions=[infer_inputs["condition"]],
            prompt=infer_inputs["prompt"],
            negative_prompt=infer_inputs["negative_prompt"],
            num_inference_steps=infer_inputs["num_steps"],
            guidance_scale=1.0,
            height=infer_inputs["height"],
            width=infer_inputs["width"],
            generator=generator,
            num_frames=infer_inputs["num_frames"],
            frame_rate=infer_inputs["video_fps"],
            num_videos_per_prompt=sample_time,
            output_type="pt",
            return_action=True,
            return_video=True,
            action_chunk=infer_inputs["action_chunk"],
            action_dim=infer_inputs["action_dim"],
            context_dict=infer_inputs["context_dict"],
            noise_seed=generator_seed,
            pixel_wise_timestep=True,
            image_cond_noise_scale=0.0,
        )
        torch.cuda.synchronize()
        timing["pipe_infer"] = time.perf_counter() - t1

        preds = out.frames if hasattr(out, "frames") else out
        rollout = self._postprocess_joint_rollout_outputs(
            infer_inputs["cond_tensor"],
            preds,
            video_fps=infer_inputs["video_fps"],
            condition_frames=infer_inputs["condition_frames"],
            num_waypoints=infer_inputs["num_waypoints"],
            waypoint_interval_s=infer_inputs["waypoint_interval_s"],
        )
        rollout["timing"] = timing
        return rollout

    @torch.inference_mode()
    def sample_joint_rollouts_batched(
        self,
        features: Dict[str, torch.Tensor],
        sample_time: int = 8,
        seed: Optional[int] = None,
    ) -> Dict[str, object]:
        """
        Batched wrapper over `sample_joint_rollouts`.

        Returns tensors flattened over `(batch, group)` so they can be passed
        directly to GRPO reward consumers expecting `(B * G, ...)`.
        """
        if "images" not in features or not torch.is_tensor(features["images"]):
            raise ValueError("Expected batched `features['images']` tensor.")

        images = features["images"]
        if images.ndim == 4:
            rollout = self.sample_joint_rollouts(features, sample_time=sample_time, seed=seed)
            rollout["batch_size"] = 1
            rollout["group_size"] = sample_time
            return rollout
        if images.ndim != 5:
            raise ValueError(f"Expected images shaped (B,T,C,H,W) or (T,C,H,W), got {tuple(images.shape)}")

        batch_size = images.shape[0]
        rollouts = []
        for batch_index in range(batch_size):
            single_features = self._slice_feature_sample(features, batch_index)
            single_seed = None if seed is None else int(seed) + batch_index
            rollouts.append(
                self.sample_joint_rollouts(single_features, sample_time=sample_time, seed=single_seed)
            )

        merged: Dict[str, object] = {
            "actions_tensor": torch.cat([r["actions_tensor"] for r in rollouts], dim=0),
            "actions": np.concatenate([r["actions"] for r in rollouts], axis=0),
            "generated_video": torch.cat([r["generated_video"] for r in rollouts], dim=0),
            "generated_future_video": torch.cat([r["generated_future_video"] for r in rollouts], dim=0),
            "generated_future_video_bcthw": torch.cat(
                [r["generated_future_video_bcthw"] for r in rollouts], dim=0
            ),
            "idm_clip": torch.cat([r["idm_clip"] for r in rollouts], dim=0),
            "idm_meta": rollouts[0]["idm_meta"],
            "batch_size": batch_size,
            "group_size": sample_time,
            "timing": [r["timing"] for r in rollouts],
        }
        return merged

    @torch.inference_mode()
    def compute_idm_reward(
        self,
        idm_model: torch.nn.Module,
        rollout: Dict[str, object],
        lambda_pos: float = 1.0,
        lambda_yaw: float = 0.2,
        time_weights: Optional[torch.Tensor] = None,
        reduction: str = "none",
    ) -> Dict[str, torch.Tensor]:
        """
        Compute Reward 2 from a joint rollout:
          generated video -> IDM -> 8 waypoints -> align with sampled actions
        """
        idm_clip = rollout["idm_clip"]
        idm_meta = rollout["idm_meta"]
        sampled_waypoints = rollout["actions_tensor"]

        if not torch.is_tensor(idm_clip) or not torch.is_tensor(sampled_waypoints):
            raise TypeError("rollout must contain tensor entries `idm_clip` and `actions_tensor`")
        if idm_meta.future_frames != sampled_waypoints.shape[1]:
            raise ValueError(
                "IDM future frame count must match sampled trajectory horizon, "
                f"got {idm_meta.future_frames} vs {sampled_waypoints.shape[1]}"
            )

        device = self.device
        idm_clip = idm_clip.to(device)
        sampled_waypoints = sampled_waypoints.to(device)

        pred_waypoints = idm_model(idm_clip, meta=idm_meta)
        reward_dict = compute_idm_consistency_reward(
            pred_waypoints=pred_waypoints,
            sampled_waypoints=sampled_waypoints,
            lambda_pos=lambda_pos,
            lambda_yaw=lambda_yaw,
            time_weights=time_weights,
            reduction=reduction,
        )
        reward_dict["pred_waypoints"] = pred_waypoints
        return reward_dict

    @torch.inference_mode()
    def compute_musiq_reward(
        self,
        rollout: Dict[str, object],
        reduction: str = "none",
    ) -> Dict[str, torch.Tensor]:
        """
        Compute Reward 3 from a joint rollout:
          generated future video -> MUSIQ(mid frame + last frame) / 2
        """
        future_video = rollout["generated_future_video"]
        if not torch.is_tensor(future_video):
            raise TypeError("rollout must contain tensor entry `generated_future_video`")

        future_video = future_video.to(self.device)
        return compute_musiq_reward(
            future_video=future_video,
            musiq_metric=self._get_musiq_metric(),
            reduction=reduction,
        )

    @torch.inference_mode()
    def compute_joint_reward_bundle(
        self,
        features: Dict[str, torch.Tensor],
        idm_model: torch.nn.Module,
        sample_time: int = 8,
        seed: Optional[int] = None,
        lambda_pos: float = 1.0,
        lambda_yaw: float = 0.2,
    ) -> Dict[str, object]:
        """
        High-level training entry helper:
          batched features -> joint rollout -> Reward2/Reward3 tensors ready for GRPO.
        """
        rollout = self.sample_joint_rollouts_batched(
            features=features,
            sample_time=sample_time,
            seed=seed,
        )
        idm_reward_dict = self.compute_idm_reward(
            idm_model=idm_model,
            rollout=rollout,
            lambda_pos=lambda_pos,
            lambda_yaw=lambda_yaw,
            reduction="none",
        )
        musiq_reward_dict = self.compute_musiq_reward(
            rollout=rollout,
            reduction="none",
        )
        return {
            "rollout": rollout,
            "idm_reward_dict": idm_reward_dict,
            "musiq_reward_dict": musiq_reward_dict,
            "idm_rewards": idm_reward_dict["reward"],
            "musiq_rewards": musiq_reward_dict["reward"],
        }

    @torch.inference_mode()
    def forward_test(self, features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        rollout = self.sample_joint_rollouts(features, sample_time=1)
        self.last_inference_timing = rollout["timing"]
        result: Dict[str, object] = {
            "actions": rollout["actions"][0],
            "actions_tensor": rollout["actions_tensor"][0],
            "generated_future_video": rollout["generated_future_video"][0:1],
            "generated_future_video_bcthw": rollout["generated_future_video_bcthw"][0:1],
            "idm_clip": rollout["idm_clip"][0:1],
            "idm_meta": rollout["idm_meta"],
        }
        return result

    def compute_trajectory(self, agent_input: AgentInput,scene) -> Trajectory:
        """
        Computes the ego vehicle trajectory.
        :param current_input: Dataclass with agent inputs.
        :return: Trajectory representing the predicted ego's position in future
        """
        self.eval()
        
        
        timing = {}
        
        
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        features: Dict[str, torch.Tensor] = {}
        # build features
        for builder in self.get_feature_builders():
            features.update(builder.compute_features(agent_input))
        for builder in self.get_target_builders():
            data_dict = builder.compute_targets(scene)
        torch.cuda.synchronize()
        timing['feature_building'] = time.perf_counter() - t0

        # add batch dimension
        features = {k: v.unsqueeze(0) for k, v in features.items()}

        # forward pass
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        with torch.no_grad():
            predictions = self.forward(features)
            poses = predictions["actions"]
        torch.cuda.synchronize()
        timing['forward_pass'] = time.perf_counter() - t1
        
        
        if hasattr(self, 'last_inference_timing'):
            timing.update(self.last_inference_timing)
        
        
        self.last_inference_timing = timing
        
        # print_pred_gt_together(poses, data_dict["trajectory"])
        # exit(0)
        # extract trajectory
        return Trajectory(poses)

    def compute_trajectory_vis(self, agent_input: AgentInput) -> Trajectory:
        self.eval()

        features: Dict[str, torch.Tensor] = {}
        # build features
        for builder in self.get_feature_builders():
            features.update(builder.compute_features(agent_input))

        # add batch dimension
        features = {k: v.unsqueeze(0) for k, v in features.items()}

        with torch.no_grad():
            predictions = self.forward(features)
            poses = predictions["actions"]
        return Trajectory(poses)


    def compute_loss(self, features: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor], predictions: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self.training and self.grpo:
            return predictions
        elif self.training:
            return predictions.loss
        else:
            return torch.nn.functional.l1_loss(predictions["pred_traj"], targets["trajectory"])

    def get_optimizers(self) -> Union[Optimizer, Dict[str, LRScheduler]]:
        optimizer_cfg = DictConfig(dict(type="AdamW", lr=self._lr, weight_decay=1e-4, betas=(0.9, 0.95)))
        optimizer = build_from_configs(optim, optimizer_cfg, params=self.action_head.parameters())
        
        if self.grpo:
            scheduler = WarmupCosLR(optimizer=optimizer, lr=self._lr, min_lr=0.0, epochs=10, warmup_epochs=0)
        else:
            scheduler = WarmupCosLR(optimizer=optimizer, lr=self._lr, min_lr=1e-6, epochs=200, warmup_epochs=3)
            
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}

    @staticmethod
    def _decode_paths_from_tensor(path_tensor: torch.Tensor) -> List[str]:
        """
        Decodes a batch of path tensors back into a list of file path strings.
        
        Args:
            path_tensor (torch.Tensor): A 2D tensor of shape 
                (batch_size, max_path_length) from the collate_fn.
        
        Returns:
            List[str]: A list of decoded file path strings.
        """
        decoded_paths = []
        for single_path_tensor in path_tensor:
            chars = []
            for code in single_path_tensor:
                code_item = code.item()
                if code_item == 0: 
                    break
                chars.append(chr(code_item))
            decoded_paths.append("".join(chars))
        return decoded_paths

    @staticmethod
    def _compute_rel_from_T44(T: torch.Tensor):
            if not torch.is_tensor(T):
                T = torch.from_numpy(T)
            T = T.to(dtype=torch.float32)
            device = T.device
            n = T.shape[0]
            poses = torch.zeros(n, 2, dtype=T.dtype, device=device)
            yaws  = torch.zeros(n,     dtype=T.dtype, device=device)

            for i in range(1, n):
                T_prev_inv = torch.linalg.inv(T[i-1])
                T_rel = T_prev_inv @ T[i]          # (4,4)
                poses[i] = T_rel[:2, 3]
                R = T_rel[:3, :3]
                yaws[i] = torch.atan2(R[1, 0], R[0, 0])
            return poses, yaws

    @staticmethod
    def _tensor_video_to_pils(video: torch.Tensor):
        """
        video: (T,C,H,W) in [-1,1] 或 [0,1]
        return: List[PIL.Image]
        """
        x = video.detach().cpu()
        if x.min() < 0:
            x = (x + 1.0) / 2.0
        x = x.clamp(0, 1)
        pils = []
        T = x.shape[0]
        for t in range(T):
            arr = (x[t].permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
            pils.append(Image.fromarray(arr))
        return pils

def pil_frames_to_video(pil_frames, output_path, fps):
    try:
        export_to_video(pil_frames, output_path, fps=fps)
        return True
    except Exception as e:
        print(f"❌ Error saving video {output_path}: {e}")
        return False

def denorm_odo(normalized_trajectory: torch.Tensor) -> torch.Tensor:
    """Denormalizes trajectory from [-1, 1] back to original coordinate space."""
    x = (normalized_trajectory[..., 0:1] + 1) / 2 * 66.74 - 1.57
    y = (normalized_trajectory[..., 1:2] + 1) / 2 * 42 - 19.68
    heading = (normalized_trajectory[..., 2:3] + 1) / 2 * 3.53 - 1.67
    return torch.cat([x, y, heading], dim=-1)

def _to_np(x):
    import numpy as np
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x)

def print_pred_gt_together(poses, gt, precision=3, max_rows=None):
    import numpy as np
    p = _to_np(poses)
    g = _to_np(gt)

    if p.ndim == 1: p = p[:, None]
    if g.ndim == 1: g = g[:, None]

    T = min(len(p), len(g))
    if T == 0:
        print("No overlapping timesteps (T=0).")
        return

    p, g = p[:T], g[:T]
    D = min(p.shape[1], g.shape[1])
    p, g = p[:, :D], g[:, :D]

    err = np.linalg.norm(p - g, axis=1)
    np.set_printoptions(precision=precision, suppress=True)

    print(f"{'t':>3} | {'pred':>20} | {'gt':>20} | {'L2':>8}")
    print("-" * 64)
    rows = T if max_rows is None else min(max_rows, T)
    for i in range(rows):
        print(f"{i:>3} | {p[i]} | {g[i]} | {err[i]:>8.{precision}f}")
    if rows < T:
        print(f"... ({T-rows} more rows)")
    print("-" * 64)
    print(f"Avg L2 over T={T}: {err.mean():.{precision}f}")
