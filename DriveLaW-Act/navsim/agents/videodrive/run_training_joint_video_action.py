"""Adapt DriveLaW-Act video generation while preserving trajectory planning."""

import json
import os

import hydra
import torch
import torch.nn.functional as F
from accelerate import DistributedType
from einops import rearrange
from hydra.utils import get_original_cwd
from omegaconf import DictConfig

from navsim.agents.videodrive.run_training_videodrive import (
    CONFIG_NAME,
    CONFIG_PATH,
    UnifiedTrainer,
    free_memory,
    gen_noise_from_condition_frame_latent,
    get_latents,
    get_memory_statistics,
    get_text_conditions,
    logger,
    randn_tensor,
    unwrap_model,
)


class JointVideoActionTrainer(UnifiedTrainer):
    """Joint video/action trainer with real future-video supervision."""

    def _prepare_video_latents(self, batch, weight_dtype):
        accel = self.state.accelerator
        history = batch["video"].to(accel.device, dtype=weight_dtype).contiguous()
        batch_size, _, views, history_frames, height, width = history.shape
        if views != 1 or history_frames != 4:
            raise ValueError(
                f"Expected one view and four history frames, got {tuple(history.shape)}"
            )
        history = rearrange(history, "b c v t h w -> (b v) c t h w")

        future = batch["future_video"].to(accel.device, dtype=weight_dtype).contiguous()
        future = rearrange(future, "b c v t h w -> (b v) c t h w")
        expected_future_frames = int(self.args.future_video_frames)
        if future.shape[2] != expected_future_frames:
            raise ValueError(
                f"Expected {expected_future_frames} real future frames, got {future.shape[2]}"
            )

        # The future VAE clip starts at the current camera frame. At 2 Hz,
        # current + eight NAVSIM targets is a 9-frame, 4-second clip.
        target_clip = torch.cat([history[:, :, -1:], future], dim=2)
        if target_clip.shape[2] != 9:
            raise ValueError(f"Expected a 9-frame target clip, got {target_clip.shape[2]}")

        memory_latents, future_latents = get_latents(
            self.vae,
            history,
            target_clip,
            device=accel.device,
            dtype=weight_dtype,
        )
        latent_height = height // self.SPATIAL_DOWN_RATIO
        latent_width = width // self.SPATIAL_DOWN_RATIO

        memory_latents = rearrange(
            memory_latents,
            "(b m) (h w) c -> b c m h w",
            b=batch_size,
            m=history_frames,
            h=latent_height,
            w=latent_width,
        )
        future_latents = rearrange(
            future_latents,
            "b (f h w) c -> b c f h w",
            b=batch_size,
            h=latent_height,
            w=latent_width,
        )
        if future_latents.shape[2] != 2:
            raise ValueError(
                "A 9-frame target clip must encode to two temporal latents; "
                f"got {future_latents.shape[2]}"
            )

        clean_latents = torch.cat([memory_latents, future_latents], dim=2)
        latent_frames = clean_latents.shape[2]
        clean_latents = rearrange(
            clean_latents, "b c f h w -> b (f h w) c"
        ).to(dtype=weight_dtype)

        condition_tokens = history_frames * latent_height * latent_width
        conditioning_mask = torch.zeros(
            batch_size,
            clean_latents.shape[1],
            dtype=torch.bool,
            device=clean_latents.device,
        )
        conditioning_mask[:, :condition_tokens] = True
        return clean_latents, conditioning_mask, latent_frames, latent_height, latent_width

    def _video_flow_batch(self, clean_latents, conditioning_mask):
        sigmas = self._timestep_sampler.sample_for(clean_latents).clamp(0.0, 1.0)
        sigmas = sigmas.to(device=clean_latents.device, dtype=clean_latents.dtype)
        noise = torch.randn_like(clean_latents)
        sigma = sigmas.view(-1, 1, 1)
        noisy_latents = (1.0 - sigma) * clean_latents + sigma * noise
        noisy_latents = torch.where(
            conditioning_mask.unsqueeze(-1), clean_latents, noisy_latents
        )
        timesteps = self._create_timesteps_from_conditioning_mask(
            conditioning_mask, torch.round(sigmas * 1000.0).long()
        )
        return noisy_latents, timesteps, noise - clean_latents

    @staticmethod
    def _masked_video_loss(prediction, target, conditioning_mask):
        prediction = prediction.float()
        target = target.float()
        target_mask = (~conditioning_mask).unsqueeze(-1).to(prediction.dtype)
        denominator = target_mask.sum() * prediction.shape[-1]
        return ((prediction - target).pow(2) * target_mask).sum() / denominator.clamp_min(1)

    def _paper_stage3_action_loss(self, batch, weight_dtype):
        """Run the official Stage-3 Action forward and flow-matching target unchanged.

        The additional video loss is evaluated in a separate forward pass. Keeping
        this branch independent preserves the official Action noise construction,
        conditioning layout, timestep sampling, and target ``noise - actions``.
        """
        accel = self.state.accelerator

        video = batch["video"].to(accel.device, dtype=weight_dtype).contiguous()
        batch_size, _, views, history_frames, height, width = video.shape
        if views != 1 or history_frames != 4:
            raise ValueError(
                "Official Stage-3 Action branch expects one view and four history frames, "
                f"got {tuple(video.shape)}"
            )
        memory = rearrange(video, "b c v t h w -> (b v) c t h w")

        # This is intentionally the same Action-conditioning construction as
        # the author's Stage-3 runner: one future frame repeated to form the
        # 9-frame VAE clip used only to establish the latent layout.
        future_video = batch["future_video"].to(accel.device, dtype=weight_dtype).contiguous()
        future_video = rearrange(future_video, "b c v t h w -> (b v) c t h w")
        future_clip = future_video[:, :, :1].repeat(1, 1, 9, 1, 1)
        memory_latents, future_latents = get_latents(self.vae, memory, future_clip)

        latent_frames = 4 + 9 // self.TEMPORAL_DOWN_RATIO + 1
        latent_height = height // self.SPATIAL_DOWN_RATIO
        latent_width = width // self.SPATIAL_DOWN_RATIO
        memory_latents = rearrange(
            memory_latents,
            "(b m) (h w) c -> b c m h w",
            b=batch_size,
            m=4,
            h=latent_height,
        )
        future_latents = rearrange(
            future_latents,
            "b (f h w) c -> b c f h w",
            b=batch_size,
            h=latent_height,
            w=latent_width,
        )
        latents = torch.cat((memory_latents, future_latents), dim=2)
        latents = rearrange(latents, "b c f h w -> b (f h w) c")

        noise, conditioning_mask, condition_indicator = gen_noise_from_condition_frame_latent(
            memory_latents,
            latent_frames,
            latent_height,
            latent_width,
            noise_to_condition_frames=0.0,
        )
        sigmas = self._timestep_sampler.sample_for(noise)
        timesteps = torch.round(sigmas * 1000.0).long()
        if self.args.pixel_wise_timestep:
            timesteps = timesteps.unsqueeze(-1) * (1 - conditioning_mask.long())
        else:
            timesteps = timesteps.unsqueeze(-1) * (1 - condition_indicator)

        sigma = sigmas.reshape(-1, 1, 1).repeat(1, 1, latents.size(-1))
        if self.args.return_action and self.args.noisy_video:
            sigma = torch.full_like(sigma, 1.0)
        noisy_video = (1.0 - sigma) * latents + sigma * noise

        actions = batch["actions"].to(accel.device, dtype=weight_dtype)
        action_noise = randn_tensor(actions.shape, device=accel.device, dtype=weight_dtype)
        action_sigmas = self._timestep_sampler.sample_for(actions).clamp(0.0, 1.0)
        noisy_actions = (
            (1.0 - action_sigmas.view(-1, 1, 1)) * actions
            + action_sigmas.view(-1, 1, 1) * action_noise
        )
        action_timesteps = torch.round(action_sigmas * 1000.0).long()
        action_timesteps = action_timesteps.unsqueeze(-1).expand(-1, actions.shape[1])

        context_dict = {
            "cmd_onehot": batch["driving_command"].to(accel.device, dtype=weight_dtype),
            "vel": batch["vel"].to(accel.device, dtype=weight_dtype),
            "acc": batch["acc"].to(accel.device, dtype=weight_dtype),
        }
        captions = batch["caption"]
        text_conditions = get_text_conditions(self.tokenizer, self.text_encoder, captions)
        prompt_embeds = text_conditions["prompt_embeds"]
        prompt_attention_mask = text_conditions["prompt_attention_mask"]
        dropout_factor = torch.rand(batch_size, device=accel.device, dtype=weight_dtype)
        dropout_mask = (dropout_factor < self.args.caption_dropout_p).unsqueeze(1).unsqueeze(2)
        prompt_embeds = self.uncond_prompt_embeds.repeat(batch_size, 1, 1) * dropout_mask + (
            prompt_embeds * ~dropout_mask
        )

        if noisy_video.dtype != weight_dtype or noisy_actions.dtype != weight_dtype:
            raise TypeError(
                "Action forward dtype mismatch: "
                f"video={noisy_video.dtype}, action={noisy_actions.dtype}, "
                f"expected={weight_dtype}"
            )
        prediction = self._forward_pass(
            latents=noisy_video,
            timesteps=video_timesteps,
            prompt_embeds=prompt_embeds,
            prompt_attention_mask=prompt_attention_mask,
            latent_frames=latent_frames,
            latent_height=latent_height,
            latent_width=latent_width,
            return_video=self.args.return_video or self.args.return_action,
            return_action=True,
            video_attention_mask=None,
            action_timestep=action_timesteps,
            action_states=noisy_actions,
            action_dim=actions.shape[-1],
            context_dict=context_dict,
        )["action"]
        return F.mse_loss(prediction.float(), (action_noise - actions).float())

    def train(self):
        logger.info(
            "Starting joint video/action training (train_mode=%s)",
            self.args.train_mode,
        )
        logger.info("Memory before: %s", json.dumps(get_memory_statistics(), indent=2))
        accel = self.state.accelerator
        weight_dtype = self.state.weight_dtype
        video_loss_scale = float(self.args.video_loss_scale)
        action_loss_scale = float(self.args.action_loss_scale)
        global_step = 0

        for epoch in range(self.state.train_epochs):
            self.diffusion_model.train()
            for step, batch in enumerate(self.train_dataloader):
                logs = {}
                with accel.accumulate(self.diffusion_model):
                    # Run this first so Action uses the same random-operation order
                    # and flow-matching construction as the official Stage-3 runner.
                    loss_action = self._paper_stage3_action_loss(batch, weight_dtype)
                    if not torch.isfinite(loss_action):
                        raise FloatingPointError(f"Non-finite action loss: {loss_action.item()}")

                    (
                        clean_latents,
                        conditioning_mask,
                        latent_frames,
                        latent_height,
                        latent_width,
                    ) = self._prepare_video_latents(batch, weight_dtype)

                    captions = batch["caption"]
                    text_conditions = get_text_conditions(
                        self.tokenizer, self.text_encoder, captions
                    )
                    prompt_embeds = text_conditions["prompt_embeds"].to(
                        accel.device, dtype=weight_dtype
                    )
                    prompt_attention_mask = text_conditions["prompt_attention_mask"].to(
                        accel.device
                    )
                    dropout = (
                        torch.rand(clean_latents.shape[0], device=accel.device)
                        < float(self.args.caption_dropout_p)
                    ).view(-1, 1, 1)
                    prompt_embeds = torch.where(
                        dropout,
                        self.uncond_prompt_embeds.to(
                            accel.device, dtype=weight_dtype
                        ).repeat(clean_latents.shape[0], 1, 1),
                        prompt_embeds,
                    )

                    video_input, video_timesteps, video_target = self._video_flow_batch(
                        clean_latents, conditioning_mask
                    )
                    if video_input.dtype != weight_dtype:
                        raise TypeError(
                            f"Video forward dtype is {video_input.dtype}, expected {weight_dtype}"
                        )
                    video_prediction = self._forward_pass(
                        latents=video_input,
                        timesteps=video_timesteps,
                        prompt_embeds=prompt_embeds,
                        prompt_attention_mask=prompt_attention_mask,
                        latent_frames=latent_frames,
                        latent_height=latent_height,
                        latent_width=latent_width,
                        return_video=True,
                        return_action=False,
                        video_attention_mask=None,
                    )["video"]
                    loss_video = self._masked_video_loss(
                        video_prediction, video_target, conditioning_mask
                    )
                    if not torch.isfinite(loss_video):
                        raise FloatingPointError(f"Non-finite video loss: {loss_video.item()}")

                    loss = video_loss_scale * loss_video + action_loss_scale * loss_action
                    if step == 0 and epoch == 0 and accel.is_main_process:
                        model_dtype = next(self.diffusion_model.parameters()).dtype
                        logger.info(
                            "Forward contract: latents=%s/%s, prompt=%s/%s, "
                            "video_pred=%s/%s, model_dtype=%s",
                            tuple(clean_latents.shape),
                            clean_latents.dtype,
                            tuple(prompt_embeds.shape),
                            prompt_embeds.dtype,
                            tuple(video_prediction.shape),
                            video_prediction.dtype,
                            model_dtype,
                        )
                    accel.backward(loss)

                    if accel.sync_gradients and accel.distributed_type != DistributedType.DEEPSPEED:
                        logs["grad_norm"] = accel.clip_grad_norm_(
                            self.diffusion_model.parameters(), self.args.max_grad_norm
                        )
                    self.optimizer.step()
                    self.lr_scheduler.step()
                    if self.use_ema and accel.sync_gradients:
                        base_model = unwrap_model(accel, self.diffusion_model)
                        self.ema.step([p for p in base_model.parameters() if p.requires_grad])
                    self.optimizer.zero_grad()

                reduced_video = accel.reduce(loss_video.detach(), reduction="mean")
                reduced_action = accel.reduce(loss_action.detach(), reduction="mean")
                reduced_total = video_loss_scale * reduced_video + action_loss_scale * reduced_action
                if accel.sync_gradients:
                    global_step += 1

                logs.update(
                    {
                        "loss": reduced_total.item(),
                        "loss_video": reduced_video.item(),
                        "loss_action": reduced_action.item(),
                        "lr": self.lr_scheduler.get_last_lr()[0],
                    }
                )
                if global_step % self.args.steps_to_log == 0 or step == 0:
                    accel.log(logs, step=global_step)
                    if accel.is_main_process:
                        logger.info(
                            "[step %d] total=%.5f video=%.5f action=%.5f lr=%.3g",
                            global_step,
                            logs["loss"],
                            logs["loss_video"],
                            logs["loss_action"],
                            logs["lr"],
                        )

                if (
                    global_step > 0
                    and global_step % self.args.steps_to_save == 0
                    and accel.is_main_process
                ):
                    model = unwrap_model(accel, self.diffusion_model)
                    model.save_pretrained(
                        os.path.join(self.save_folder, f"step_{global_step}"),
                        safe_serialization=True,
                    )

                if global_step >= self.state.train_steps:
                    break

            logger.info(
                "Epoch %d complete. Memory: %s",
                epoch + 1,
                json.dumps(get_memory_statistics(), indent=2),
            )
            if global_step >= self.state.train_steps:
                break

        accel.wait_for_everyone()
        if accel.is_main_process:
            model = unwrap_model(accel, self.diffusion_model)
            model.save_pretrained(
                os.path.join(self.save_folder, f"step_{global_step}"),
                safe_serialization=True,
            )
        del self.diffusion_model, self.scheduler
        free_memory()
        accel.end_training()


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig):
    config_file = cfg.agent.config_file
    if not os.path.isabs(config_file):
        config_file = os.path.join(get_original_cwd(), config_file)
    trainer = JointVideoActionTrainer(cfg, config_file)
    if trainer.args.train_mode != "video_action_adapt":
        raise ValueError("Joint trainer requires train_mode: video_action_adapt")
    trainer.prepare_dataset()
    if trainer.args.enable_val:
        trainer.prepare_val_dataset()
    trainer.prepare_models()
    trainer.prepare_trainable_parameters()
    trainer.prepare_for_training()
    trainer.prepare_trackers()
    trainer.train()


if __name__ == "__main__":
    main()
