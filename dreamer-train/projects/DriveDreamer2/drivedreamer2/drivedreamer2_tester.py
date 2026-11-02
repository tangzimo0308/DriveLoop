import copy
import os
import json
import hashlib
import math
import shutil
import time
import imageio
import numpy as np
import torch
import torch.utils.checkpoint
from diffusers import DDIMScheduler, UniPCMultistepScheduler, LMSDiscreteScheduler, EulerDiscreteScheduler,AutoencoderKLTemporalDecoder

from dreamer_datasets import DefaultCollator, load_dataset, DefaultSampler,CLIPTextTransform
from dreamer_models import DriveDreamer2Pipeline
from dreamer_train import Tester
from . import drivedreamer2_transforms
from .drivedreamer2_utils import GLIGEN_WEIGHT_NAME, VideoSampler, VideoCollator, draw_mv_video,draw_mv_video_v2

CAM_NAMES = ['CAM_FRONT',
'CAM_FRONT_LEFT',
'CAM_FRONT_RIGHT',
'CAM_BACK',
'CAM_BACK_LEFT',
'CAM_BACK_RIGHT']
from torchvision import transforms


def _driveloop_selected_video_indices(data_config, batch_skip, frame_num):
    from pathlib import Path
    from scripts.run_dd2_batch_sampler_audit import (
        candidate_camera_starts,
        load_records,
        selected_frame_indices,
    )

    if batch_skip < 0:
        raise ValueError(f"DRIVELOOP_DD2_BATCH_SKIP must be >= 0, got {batch_skip}")

    labels_path = Path(data_config.data_or_config) / "labels" / "data.pkl"
    records = load_records(labels_path)
    starts = candidate_camera_starts(
        records,
        frame_num=frame_num,
        hz_factor=data_config.hz_factor,
        video_split_rate=data_config.get('video_split_rate', 1),
        multiview=data_config.is_multiview,
    )

    if batch_skip >= len(starts):
        raise IndexError(
            f"DRIVELOOP_DD2_BATCH_SKIP={batch_skip} exceeds candidate count {len(starts)}"
        )

    return selected_frame_indices(
        starts[batch_skip],
        frame_num=frame_num,
        hz_factor=data_config.hz_factor,
    )

class DriveDreamer2_Tester(Tester):
    def get_dataloaders(self, data_config):
        self.data_config = data_config
        data_or_config_override = os.environ.get("DRIVELOOP_DD2_DATA_OR_CONFIG")
        if data_or_config_override:
            self.logger.info(
                'DRIVELOOP_DD2_DATA_OR_CONFIG=%s: override data_config.data_or_config',
                data_or_config_override,
            )
            data_config.data_or_config = data_or_config_override
        self.driveloop_data_or_config = str(data_config.data_or_config)
        dataset = load_dataset(data_config.data_or_config)
        transform = getattr(drivedreamer2_transforms, data_config.transform.pop('type'))(**data_config.transform)
        dataset.set_transform(transform)

        self.fps=data_config.fps
        self.cam_num = data_config.cam_num
        self.frame_num = data_config.frame_num
        batch_size_per_gpu = self.frame_num * self.cam_num
        cam_names = data_config.get('cam_names',None)
        self.dd2_sampler_selected_batch_index = None

        target_batch_skip_requested = "DRIVELOOP_DD2_BATCH_SKIP" in os.environ
        target_batch_skip = int(os.environ.get("DRIVELOOP_DD2_BATCH_SKIP", "0"))
        source_bound_requested = os.environ.get("DRIVELOOP_DD2_SOURCE_BOUND") == "1"
        if (
            target_batch_skip_requested
            and (target_batch_skip > 0 or source_bound_requested)
            and 'Video' in data_config.type
        ):
            selected_indices = _driveloop_selected_video_indices(
                data_config,
                batch_skip=target_batch_skip,
                frame_num=self.frame_num,
            )
            self.dd2_sampler_selected_batch_index = target_batch_skip
            self.logger.info(
                'DRIVELOOP_DD2_BATCH_SKIP=%s: use targeted dataset subset with %s records',
                target_batch_skip,
                len(selected_indices),
            )
            selected_dataset = torch.utils.data.Subset(dataset, selected_indices)
            return torch.utils.data.DataLoader(
                selected_dataset,
                collate_fn=VideoCollator(
                    frame_num=self.frame_num,
                    img_mask_type=data_config.img_mask_type,
                    img_mask_num=data_config.img_mask_num,
                ),
                batch_size=len(selected_indices),
                num_workers=0,
            )

        audit_batch_skip = target_batch_skip
        if os.environ.get("DRIVELOOP_DD2_AUDIT_ONLY") == "1" and audit_batch_skip == 0:
            self.logger.info('DRIVELOOP_DD2_AUDIT_ONLY=1: use first contiguous batch without VideoSampler')
            audit_size = min(len(dataset), batch_size_per_gpu)
            audit_dataset = torch.utils.data.Subset(dataset, list(range(audit_size)))
            return torch.utils.data.DataLoader(
                audit_dataset,
                collate_fn=VideoCollator(
                    frame_num=self.frame_num,
                    img_mask_type=data_config.img_mask_type,
                    img_mask_num=data_config.img_mask_num,
                ) if 'Video' in data_config.type else DefaultCollator(),
                batch_size=audit_size,
                num_workers=0,
            )
        if os.environ.get("DRIVELOOP_DD2_AUDIT_ONLY") == "1":
            self.logger.info('DRIVELOOP_DD2_AUDIT_ONLY=1 with batch skip: use VideoSampler to match generation path')
        
        dataloader = torch.utils.data.DataLoader(
            dataset,
            sampler=VideoSampler(
                    dataset, 
                    batch_size=batch_size_per_gpu, 
                    frame_num=self.frame_num, 
                    cam_num=self.cam_num,
                    video_split_rate=data_config.get('video_split_rate',1),
                    hz_factor=data_config.hz_factor,
                    mv_video=data_config.is_multiview, 
                    view=data_config.view,
                    shuffle=data_config.shuffle,
                    logger=self.logger,
                    resample_num_workers=data_config.get('resample_num_workers', 0),
                    resample_batch_size=data_config.get('resample_batch_size', 8)), 
            collate_fn=VideoCollator(
                frame_num=self.frame_num,
                img_mask_type=data_config.img_mask_type,
                img_mask_num=data_config.img_mask_num) 
                if 'Video' in data_config.type else DefaultCollator(),
                batch_size=batch_size_per_gpu,
                num_workers=data_config.num_workers,)
       
       
        return dataloader
    
    def get_models(self, model_config):
        local_files_only = model_config.get('local_files_only', True)
        if os.environ.get("DRIVELOOP_DD2_AUDIT_ONLY") == "1":
            text_encoder_pretrained = model_config.get('text_encoder_pretrained', None)
            if text_encoder_pretrained is None:
                assert False
            self.text_encoder = CLIPTextTransform(
                model_path=text_encoder_pretrained,
                device=self.device,
                dtype=self.dtype,
            )
            self.mode = model_config.get('mode', 'img_cond')
            assert self.mode in ['img_cond', 'video_cond', 'wo_img']
            self.num_inf_steps = model_config.get('num_inf_steps', 50)
            self.logger.info('DRIVELOOP_DD2_AUDIT_ONLY=1: skip DriveDreamer2Pipeline load')
            return None
        pipeline_name = model_config.pipeline_name
        text_encoder_pretrained = model_config.get('text_encoder_pretrained',None)
        variant = 'fp16' if self.mixed_precision == 'fp16' else None
        if pipeline_name == 'DriveDreamer2Pipeline':
            model=DriveDreamer2Pipeline.from_pretrained(
                model_config.pretrained,
                torch_dtype=self.dtype,
                variant=variant,
                local_files_only=local_files_only,
                safety_checker=None,
            )
            if text_encoder_pretrained is None:
                assert False
            self.text_encoder = CLIPTextTransform(
                model_path=text_encoder_pretrained,
                device=self.device,
                dtype=self.dtype,
            )
            setattr(model, 'frame_num',8)
            # setattr(model, 'cam_num', self.cam_num)
            # model.load_clipTextTransformer()
        else:
            assert False
        
        self.mode = model_config.get('mode','img_cond')
        
        assert self.mode in ['img_cond','video_cond','wo_img']

        self.num_inf_steps = model_config.get('num_inf_steps', 50)
        
        weight_path = model_config.get('weight_path', None)
        if weight_path is None:
            checkpoint = self.get_checkpoint()
            weight_path = os.path.join(checkpoint, GLIGEN_WEIGHT_NAME)
        elif os.path.isdir(weight_path):
            weight_path = os.path.join(weight_path, GLIGEN_WEIGHT_NAME)
        
        assert weight_path is not None
        self.logger.info('load from {}'.format(weight_path))
        model.load_weights(weight_path)
        model.to(self.device)
        return model

    def test(self):
        if self.is_main_process:
            save_dir = self.kwargs.get('save_dir', None)
            os.makedirs(save_dir,exist_ok=True)
            num_steps_override = os.environ.get("DRIVELOOP_DD2_NUM_INF_STEPS")
            if num_steps_override:
                self.num_inf_steps = int(num_steps_override)
                self.logger.info('DRIVELOOP_DD2_NUM_INF_STEPS=%s: override num_inf_steps', num_steps_override)
            for kwargs_key, env_key in (
                ('min_guidance_scale', 'DRIVELOOP_DD2_MIN_GUIDANCE'),
                ('max_guidance_scale', 'DRIVELOOP_DD2_MAX_GUIDANCE'),
            ):
                env_value = os.environ.get(env_key)
                if env_value:
                    self.kwargs[kwargs_key] = float(env_value)
                    self.logger.info('%s=%s: override %s', env_key, env_value, kwargs_key)
            generator = torch.Generator(device=self.device)
            seed_offset = int(os.environ.get("DRIVELOOP_DD2_SEED_OFFSET", "0"))
            if seed_offset:
                self.logger.info('DRIVELOOP_DD2_SEED_OFFSET=%s: reseed per attempt', seed_offset)
            generator.manual_seed(self.seed + seed_offset)
            idx = 0
            sampler_selected_batch_index = getattr(self, "dd2_sampler_selected_batch_index", None)
            batch_skip = 0 if sampler_selected_batch_index is not None else int(os.environ.get("DRIVELOOP_DD2_BATCH_SKIP", "0"))
            prompts = [
                ['realistic autonomous driving scene, panoramic videos from different perspectives.' ],
                ['rainy, realistic autonomous driving scene, panoramic videos from different perspectives.'],
                ['night, realistic autonomous driving scene, panoramic videos from different perspectives.'],
            ]
            for batch_i, batch_dict in enumerate(self.dataloader):
                if batch_i < batch_skip:
                    continue
                grounding_downsampler_input = batch_dict.get('grounding_downsampler_input', None)
                grounding_downsampler_input = grounding_downsampler_input.reshape(self.cam_num,self.frame_num,*grounding_downsampler_input.shape[1:]).permute(1,2,3,0,4).flatten(3,4)
                box_downsampler_input = batch_dict.get('box_downsampler_input',None)
                box_downsampler_input = box_downsampler_input.reshape(self.cam_num,self.frame_num,*box_downsampler_input.shape[1:]).permute(1,2,3,0,4).flatten(3,4)
                img_cond = batch_dict.get('input_image',None)
                img_cond = img_cond.reshape(self.cam_num,self.frame_num,*img_cond.shape[1:]).permute(1,2,3,0,4).flatten(3,4)
                input_dict = {
                    'grounding_downsampler_input': grounding_downsampler_input,
                    'box_downsampler_input': box_downsampler_input}
                motion_metadata = batch_dict.get('motion_metadata', None)
                
                if self.mode == 'img_cond':
                    input_dict.update({
                        'img_cond':img_cond[:1],
                    })
                elif self.mode =='video_cond':
                    input_dict.update({
                        'video_cond':img_cond,
                    })
                
                prompt_override = os.environ.get("DRIVELOOP_DD2_PROMPT")
                if prompt_override:
                    prompt_embed = self.text_encoder([prompt_override], mode='after_pool', to_numpy=False)[:, None]
                else:
                    prompt_embed = batch_dict.get('prompt_embeds', None)

                from driveloop.image_condition_probe import validate_scale
                image_condition_scale = validate_scale(os.environ.get("DRIVELOOP_DD2_IMAGE_CONDITION_SCALE", "1"))
                image_condition_audit_path = os.environ.get("DRIVELOOP_DD2_IMAGE_CONDITION_AUDIT_PATH")
                audit_path = os.environ.get("DRIVELOOP_DD2_AUDIT_PATH")
                if audit_path:
                    def tensor_summary(value):
                        if value is None:
                            return {"available": False}
                        item = value.detach().float().cpu() if hasattr(value, "detach") else torch.tensor(value).float()
                        array = item.numpy()
                        contiguous = np.ascontiguousarray(array)
                        return {
                            "available": True,
                            "shape": list(item.shape),
                            "dtype": str(array.dtype),
                            "sum": float(item.sum().item()),
                            "mean": float(item.mean().item()),
                            "std": float(item.std().item()) if item.numel() > 1 else 0.0,
                            "nonzero": int(np.count_nonzero(array)),
                            "sha256": hashlib.sha256(contiguous.tobytes()).hexdigest(),
                        }

                    def metadata_summary(value):
                        if value is None:
                            return {
                                "available": False,
                                "claim": "metadata_observed_only_not_runtime_control",
                            }

                        def as_list(item):
                            if hasattr(item, "detach"):
                                item = item.detach().cpu()
                            if hasattr(item, "tolist"):
                                item = item.tolist()
                            if isinstance(item, tuple):
                                item = list(item)
                            if isinstance(item, list):
                                return item
                            return [item]

                        def bool_any(item):
                            return any(bool(x) for x in as_list(item))

                        def bool_all(item):
                            values = as_list(item)
                            return bool(values) and all(bool(x) for x in values)

                        def shape_preview(item, limit=8):
                            values = as_list(item)
                            normalized = [as_list(value) for value in values]
                            if not normalized:
                                return []
                            if all(isinstance(dim_values, list) for dim_values in normalized):
                                same_length = all(len(dim_values) == len(normalized[0]) for dim_values in normalized)
                                looks_transposed = 1 < len(normalized) <= 4 and same_length and len(normalized[0]) > len(normalized)
                                if looks_transposed:
                                    return [list(shape) for shape in zip(*normalized)][:limit]
                                return normalized[:limit]
                            return normalized[:limit]
                        raw_velocities_available = value.get("velocities_available_in_batch")
                        raw_actor_identity = value.get("actor_identity_available_in_batch")
                        raw_actor_identity_fields = value.get("actor_identity_fields")
                        raw_per_frame_boxes = value.get("per_frame_actor_boxes3d_observed")
                        raw_boxes_available = value.get("boxes3d_available_in_batch")
                        raw_labels_available = value.get("actor_labels_available_in_batch")
                        raw_claim = value.get("claim", "metadata_observed_only_not_runtime_control")

                        return {
                            "available": True,
                            "batch_item_count": len(as_list(raw_velocities_available)),
                            "velocities_available_in_batch_any": bool_any(raw_velocities_available),
                            "velocities_available_in_batch_all": bool_all(raw_velocities_available),
                            "velocities_shape_preview": shape_preview(value.get("velocities_shape")),
                            "actor_labels_available_in_batch_any": bool_any(raw_labels_available),
                            "actor_label_count_preview": as_list(value.get("actor_label_count"))[:8],
                            "actor_identity_available_in_batch_any": bool_any(raw_actor_identity),
                            "actor_identity_fields_preview": as_list(raw_actor_identity_fields)[:8],
                            "boxes3d_available_in_batch_any": bool_any(raw_boxes_available),
                            "boxes3d_shape_preview": shape_preview(value.get("boxes3d_shape")),
                            "per_frame_actor_boxes3d_observed_any": bool_any(raw_per_frame_boxes),
                            "claim": as_list(raw_claim)[0] if as_list(raw_claim) else "metadata_observed_only_not_runtime_control",
                        }

                    audit = {
                        "schema_version": "dd2_runtime_input_audit.v0",
                        "audit_only": os.environ.get("DRIVELOOP_DD2_AUDIT_ONLY") == "1",
                        "data_or_config": getattr(self, "driveloop_data_or_config", None),
                        "data_or_config_override": os.environ.get("DRIVELOOP_DD2_DATA_OR_CONFIG"),
                        "prompt_override": prompt_override,
                        "batch_skip": batch_skip,
                        "selected_batch_index": sampler_selected_batch_index if sampler_selected_batch_index is not None else batch_i,
                        "prompt_embed": tensor_summary(prompt_embed),
                        "img_cond": tensor_summary(input_dict.get("img_cond")),
                        "image_condition_scale": image_condition_scale,
                        "grounding_downsampler_input": tensor_summary(input_dict.get("grounding_downsampler_input")),
                        "box_downsampler_input": tensor_summary(input_dict.get("box_downsampler_input")),
                        "motion_metadata": metadata_summary(motion_metadata),
                    }
                    with open(audit_path, "w", encoding="utf-8") as f:
                        json.dump(audit, f, indent=2)

                if os.environ.get("DRIVELOOP_DD2_AUDIT_ONLY") == "1":
                    self.logger.info('DRIVELOOP_DD2_AUDIT_ONLY=1: wrote runtime audit and skipped inference')
                    idx += 1
                    break

                if prompt_embed is None:
                    videos = []
                    for this_prompt in prompts:
                        this_prompt_embed = self.text_encoder(this_prompt,mode='after_pool',to_numpy=False)[:,None]
                        images = self.model(
                            this_prompt_embed,
                            scheduled_sampling_beta=1.0,
                            input_dict=copy.deepcopy(input_dict),
                            height=batch_dict['height'][0],
                            width=batch_dict['width'][0]*6,
                            generator=generator,
                            min_guidance_scale=self.kwargs.get('min_guidance_scale', 1),
                            max_guidance_scale=self.kwargs.get('max_guidance_scale', 7.5),
                            num_inference_steps=self.num_inf_steps,
                            num_frames=self.frame_num,
                            decode_chunk_size=1,
                            first_frame=True,
                            image_condition_scale=image_condition_scale,
                            image_condition_audit_path=image_condition_audit_path,
                        )
                        
                        images=images.frames[0]
                        videos.append(images)

                    if save_dir is not None:
                        images = draw_mv_video_v2(videos, batch_dict)
                        imageio.mimsave(os.path.join(save_dir, '{:06d}.mp4'.format(idx)), images, fps=self.fps)
                        idx += 1
                else:
                    images = self.model(
                                prompt_embed[0:1].half(),
                                scheduled_sampling_beta=1.0,
                                input_dict=copy.deepcopy(input_dict),
                                height=batch_dict['height'][0],
                                width=batch_dict['width'][0]*6,
                                generator=generator,
                                min_guidance_scale=self.kwargs.get('min_guidance_scale',1),
                                max_guidance_scale=self.kwargs.get('max_guidance_scale', 7.5),
                                num_inference_steps=self.num_inf_steps,
                                num_frames=self.frame_num,
                                decode_chunk_size=1,
                                first_frame=True,
                            image_condition_scale=image_condition_scale,
                            image_condition_audit_path=image_condition_audit_path,
                            )
                    images=images.frames[0]
                    if save_dir is not None:
                        images = draw_mv_video(images, batch_dict)
                        imageio.mimsave(os.path.join(save_dir, '{:06d}.mp4'.format(idx)), images, fps=self.fps)
                    idx += 1
                    break
        self.accelerator.wait_for_everyone()
