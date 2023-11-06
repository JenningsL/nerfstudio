# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
F2-NeRF
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Tuple, Type, Optional, Any
import math

import numpy as np
import torch
import nerfacc
from nerfstudio.cameras.camera_optimizers import (CameraOptimizer,
                                                  CameraOptimizerConfig)
from nerfstudio.cameras.rays import RayBundle, RaySamples
from nerfstudio.engine.callbacks import (TrainingCallback,
                                         TrainingCallbackAttributes,
                                         TrainingCallbackLocation)
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SceneContraction
from nerfstudio.fields.density_fields import HashMLPDensityField
from nerfstudio.fields.f2nerf_field import F2NeRFField
from nerfstudio.fields.nerfacto_field import NerfactoField
from nerfstudio.model_components.losses import (
    MSELoss, distortion_loss, interlevel_loss, orientation_loss,
    pred_normal_loss, scale_gradients_by_distance_squared)
from nerfstudio.model_components.ray_samplers import F2NeRFSampler, VolumetricSampler
from nerfstudio.model_components.renderers import (AccumulationRenderer,
                                                   DepthRenderer,
                                                   NormalsRenderer,
                                                   RGBRenderer)
from nerfstudio.model_components.scene_colliders import NearFarCollider
from nerfstudio.model_components.shaders import NormalsShader
from nerfstudio.models.base_model import Model, ModelConfig
from nerfstudio.utils import colormaps
from torch.nn import Parameter
from torchmetrics.functional import structural_similarity_index_measure
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

import py_f2nerf

@dataclass
class F2NeRFModelConfig(ModelConfig):
    """F2NeRF Model Config"""

    _target: Type = field(default_factory=lambda: F2NeRFModel)
    hidden_dim: int = 64
    """Dimension of hidden layers"""
    hidden_dim_color: int = 64
    """Dimension of hidden layers for color network"""
    enable_collider: bool = False
    """Whether to create a scene collider to filter rays."""
    collider_params: Optional[Dict[str, float]] = None
    """Instant NGP doesn't use a collider."""
    grid_resolution: int = 128
    """Resolution of the grid used for the field."""
    grid_levels: int = 4
    """Levels of the grid used for the field."""
    max_res: int = 2048
    """Maximum resolution of the hashmap for the base mlp."""
    log2_hashmap_size: int = 19
    """Size of the hashmap for the base mlp"""
    alpha_thre: float = 0.01
    """Threshold for opacity skipping."""
    cone_angle: float = 0.004
    """Should be set to 0.0 for blender scenes but 1./256 for real scenes."""
    render_step_size: Optional[float] = None
    """Minimum step size for rendering."""
    near_plane: float = 0.05
    """How far along ray to start sampling."""
    far_plane: float = 1e3
    """How far along ray to stop sampling."""
    use_appearance_embedding: bool = False
    """Whether to use an appearance embedding."""
    background_color: Literal["random", "black", "white"] = "random"
    """The color that is given to untrained areas."""
    appearance_embed_dim: int = 16
    """Dimension of the appearance embedding."""
    camera_optimizer: CameraOptimizerConfig = CameraOptimizerConfig(mode="SO3xR3")
    """Config of the camera optimizer to use"""
    implementation: Literal["tcnn", "torch"] = "tcnn"
    """Which implementation to use for the model."""
    use_average_appearance_embedding: bool = False
    """Whether to use average appearance embedding or zeros for inference."""
    edge_loss_weight: float = 0.1
    """Edge feature variance loss multiplier."""
    disparity_loss_weight: float = 0
    """Disparity loss multiplier."""
    # var_loss_weight: float = 0.01
    var_loss_weight: float = 0.
    var_loss_start: int = 5000
    var_loss_end: int = 10000
    """Weight variance loss."""
    gradient_scaling_start: int = 1000
    gradient_scaling_end: int = 5000
    """Gradient scaling."""

class F2NeRFModel(Model):
    """F2NeRF model

    Args:
        config: F2NeRF configuration to instantiate model
    """

    config: F2NeRFModelConfig

    def populate_modules(self):
        """Set the fields and modules."""
        super().populate_modules()

        # create global data pool from camera poses
        cameras_train = self.kwargs['metadata']['cameras']
        bounds_train = self.kwargs['metadata']['ray_bounds']
        c2w_train = cameras_train.camera_to_worlds.cuda() # TODO: device
        bounds_train = bounds_train.to(c2w_train.device)
        w2c_R = c2w_train[:,:3,:3].transpose(1,2)
        w2c_t = -w2c_R @ c2w_train[:,:3,3:4]
        w2c_train = torch.cat([w2c_R, w2c_t], 2)
        intrinsic_train = cameras_train.get_intrinsics_matrices()
        # bounds_train = torch.Tensor([self.config.near_plane, self.config.far_plane])[None]
        # bounds_train = torch.Tensor([0.01, 5])[None] # TODO: compute from sparse points
        # bounds_train = torch.tile(bounds_train, [len(cameras_train), 1]).to(c2w_train.device)
        # Relax bounds
        bounds_train[:,0] *= 0.5
        bounds_train[:,1] *= 4
        # bounds_train[:,0] = 0.01
        # bounds_train[:,1] = 5
        bounds_train.clamp_(1e-2, 1e9)
        # print("bounds_train: ", bounds_train)
        global_data_pool = py_f2nerf.CreateGlobalDataPool(
            "configs/f2nerf.yaml", "exp/f2-nerf", 
            c2w_train, w2c_train, 
            intrinsic_train, bounds_train)
        self.global_data_pool = global_data_pool

        # RaySampler
        self.sampler = F2NeRFSampler(global_data_pool, self.density_fn)
        # sampler must be initialize before field to get correct n_volumes
        # print("n_volumes: ", self.global_data_pool.n_volumes)

        # Fields
        self.field = F2NeRFField(
            global_data_pool,
            aabb=self.scene_box.aabb,
            num_images=self.num_train_data,
            hidden_dim=self.config.hidden_dim,
            use_average_appearance_embedding=self.config.use_average_appearance_embedding,
            appearance_embedding_dim=self.config.appearance_embed_dim,
            implementation=self.config.implementation,
        )

        # # DEBUG
        # self.field = NerfactoField(
        #     aabb=self.scene_box.aabb,
        #     appearance_embedding_dim=self.config.appearance_embed_dim,
        #     num_images=self.num_train_data,
        #     log2_hashmap_size=self.config.log2_hashmap_size,
        #     max_res=self.config.max_res,
        #     spatial_distortion=SceneContraction(order=float("inf")),
        # )
        # ##########################

        self.camera_optimizer: CameraOptimizer = self.config.camera_optimizer.setup(
            num_cameras=self.num_train_data, device="cpu"
        )

        # renderers
        self.renderer_rgb = RGBRenderer(background_color=self.config.background_color)
        self.renderer_accumulation = AccumulationRenderer()
        self.renderer_depth = DepthRenderer(method="expected")
        # self.renderer_expected_depth = DepthRenderer(method="expected")
        self.renderer_normals = NormalsRenderer()

        # shaders
        self.normals_shader = NormalsShader()

        # losses
        self.rgb_loss = MSELoss()
        self.step = 0
        # metrics
        self.psnr = PeakSignalNoiseRatio(data_range=1.0)
        self.ssim = structural_similarity_index_measure
        self.lpips = LearnedPerceptualImagePatchSimilarity(normalize=True)
        self.step = 0

        self.gradient_scaling_progress = 1

    # def field_fn(self, ray_samples, ray_indices, num_rays):
    #     # for evaluating density
    #     field_outputs = self.field(ray_samples)
    #     # accumulation
    #     packed_info = nerfacc.pack_info(ray_indices, num_rays)
    #     weights, transmittance, alphas = nerfacc.render_weight_from_density(
    #         t_starts=ray_samples.frustums.starts[..., 0],
    #         t_ends=ray_samples.frustums.ends[..., 0],
    #         sigmas=field_outputs[FieldHeadNames.DENSITY][..., 0],
    #         packed_info=packed_info,
    #     )
    #     return weights, transmittance, alphas

    def density_fn(self, positions_warp, anchors):
        return self.field.density_fn(positions_warp, anchors)

    def get_param_groups(self) -> Dict[str, List[Parameter]]:
        param_groups = {}
        # param_groups["proposal_networks"] = list(self.proposal_networks.parameters())
        param_groups["fields"] = list(self.field.parameters())
        self.camera_optimizer.get_param_groups(param_groups=param_groups)
        return param_groups

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        callbacks = []

        def on_step(step):
            self.global_data_pool.iter_step = step
            # Update ray march fineness
            if step >= self.global_data_pool.ray_march_fineness_decay_end_iter:
                self.global_data_pool.ray_march_fineness = 1.
            else:
                progress = min(1, step / self.global_data_pool.ray_march_fineness_decay_end_iter)
                self.global_data_pool.ray_march_fineness = math.exp(
                    math.log(1.) * progress + math.log(self.global_data_pool.ray_march_init_fineness) * (1 - progress))
            # print(f"step: {step} ray_march_fineness: {self.global_data_pool.ray_march_fineness}")
            # self.global_data_pool.ray_march_fineness = 5 # DEBUG
            self.sampler.ray_march_fineness = torch.tensor(self.global_data_pool.ray_march_fineness)
            self.sampler.update_octree()

            # Update gradient scaling ratio
            progress = 1
            grad_scaling_start = self.config.gradient_scaling_start
            grad_scaling_end = self.config.gradient_scaling_end
            if step < grad_scaling_end:
                progress = max(0.,
                    (step - grad_scaling_start) / (grad_scaling_end - grad_scaling_start + 1e-9))
            self.gradient_scaling_progress = progress
            

        callbacks.append(
            TrainingCallback(
                where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
                update_every_num_iters=1,
                func=on_step,
            )
        )
            
        return callbacks

    def get_outputs(self, ray_bundle: RayBundle):
        assert self.field is not None
        num_rays = len(ray_bundle)

        self.global_data_pool.mode = py_f2nerf.RunningMode.TRAIN if self.training else py_f2nerf.RunningMode.VALIDATE

        with torch.no_grad():
            ray_samples, ray_indices = self.sampler(
                ray_bundle=ray_bundle,
            )

        # for tv loss
        n_edge_pts = 8192
        edge_pts, edge_anchors = self.sampler.sampler_.GetEdgeSamples(n_edge_pts)
        self.field.edge_pts = edge_pts.view(n_edge_pts * 2, 3)
        self.field.edge_anchors = edge_anchors.view(n_edge_pts * 2)

        field_outputs = self.field(ray_samples)
        # accumulation
        packed_info = nerfacc.pack_info(ray_indices, num_rays)
        # weights, transmittance, alphas = nerfacc.render_weight_from_density(
        #     t_starts=ray_samples.frustums.starts[..., 0],
        #     t_ends=ray_samples.frustums.ends[..., 0],
        #     sigmas=field_outputs[FieldHeadNames.DENSITY][..., 0],
        #     packed_info=packed_info,
        # )
        # weights = weights[..., None]
        idx_start_end = torch.stack([packed_info[:,0], packed_info[:,0]+packed_info[:,1]], dim=1).int()
        sec_density = field_outputs[FieldHeadNames.DENSITY][..., 0] * ray_samples.metadata["dt_warp"][...,0] # original impl
        alphas = 1. - torch.exp(-sec_density)
        acc_density = py_f2nerf.AccumulateSum(sec_density, idx_start_end, False)
        transmittance = torch.exp(-acc_density)
        weights = transmittance * alphas
        weights = weights[..., None]

        # gradient scaling
        if self.training and self.gradient_scaling_progress < 1.:
            field_outputs[FieldHeadNames.DENSITY] = \
                py_f2nerf.GradientScaling(field_outputs[FieldHeadNames.DENSITY], idx_start_end,
                                            self.gradient_scaling_progress)
            field_outputs[FieldHeadNames.RGB] = \
                py_f2nerf.GradientScaling(field_outputs[FieldHeadNames.RGB], idx_start_end,
                                            self.gradient_scaling_progress)

        rgb = self.renderer_rgb(
            rgb=field_outputs[FieldHeadNames.RGB],
            weights=weights,
            ray_indices=ray_indices,
            num_rays=num_rays,
        )
        depth = self.renderer_depth(
            weights=weights, ray_samples=ray_samples, ray_indices=ray_indices, num_rays=num_rays
        )
        accumulation = self.renderer_accumulation(weights=weights, ray_indices=ray_indices, num_rays=num_rays)

        # disparity_loss
        sample_t = (ray_samples.frustums.starts[..., 0] + ray_samples.frustums.ends[..., 0]) / 2 + 1e-2
        disparity = nerfacc.accumulate_along_rays(weights=weights[:,0], values=1./sample_t[:,None], ray_indices=ray_indices, n_rays=num_rays)

        # weights variance loss
        weights_variance = py_f2nerf.WeightVar(weights[:,0], idx_start_end)

        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,
            "num_samples_per_ray": packed_info[:, 1],
            "disparity": disparity,
            "weights_variance": weights_variance
        }
        return outputs

    def get_metrics_dict(self, outputs, batch):
        image = batch["image"].to(self.device)
        image = self.renderer_rgb.blend_background(image)
        metrics_dict = {}
        metrics_dict["psnr"] = self.psnr(outputs["rgb"], image)
        metrics_dict["num_samples_per_ray"] = outputs["num_samples_per_ray"].float().mean()
        metrics_dict["num_samples_per_batch"] = outputs["num_samples_per_ray"].sum()
        metrics_dict["sampled_oct_per_ray"] = self.global_data_pool.sampled_oct_per_ray
        return metrics_dict

    def compute_edge_loss(self):
        # Feature variation loss
        edge_feats = self.field.edge_feats
        tv_loss = (edge_feats[:,0] - edge_feats[:,1]).square().mean()
        return self.config.edge_loss_weight * tv_loss
    
    def compute_weights_variance_loss(self, outputs):
        var_loss = (outputs["weights_variance"] + 1e-2).sqrt().mean()
        iter_step_ = self.global_data_pool.iter_step
        var_loss_end_ = self.config.var_loss_end
        var_loss_start_ = self.config.var_loss_start
        if iter_step_ > var_loss_end_:
            var_loss_weight = self.config.var_loss_weight
        elif iter_step_ > var_loss_start_:
            var_loss_weight = float(iter_step_ - var_loss_start_) / float(var_loss_end_ - var_loss_start_) * self.config.var_loss_weight
        else:
            var_loss_weight = 0
        return var_loss_weight * var_loss

    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        image = batch["image"][..., :3].to(self.device)
        pred_rgb, image = self.renderer_rgb.blend_background_for_loss_computation(
            pred_image=outputs["rgb"],
            pred_accumulation=outputs["accumulation"],
            gt_image=image,
        )
        # rgb_loss = self.rgb_loss(image, pred_rgb)
        rgb_loss = torch.sqrt((pred_rgb - image).square() + 1e-4).mean()
        loss_dict = {"rgb_loss": rgb_loss}

        loss_dict["edge_loss"] = self.compute_edge_loss()
        loss_dict["disparity_loss"] = self.config.disparity_loss_weight * outputs["disparity"].square().mean()
        loss_dict["weight_var_loss"] = self.compute_weights_variance_loss(outputs)
        return loss_dict

    def get_image_metrics_and_images(
        self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
        image = batch["image"].to(self.device)
        image = self.renderer_rgb.blend_background(image)
        rgb = outputs["rgb"]
        acc = colormaps.apply_colormap(outputs["accumulation"])
        depth = colormaps.apply_depth_colormap(
            outputs["depth"],
            accumulation=outputs["accumulation"],
        )

        combined_rgb = torch.cat([image, rgb], dim=1)
        combined_acc = torch.cat([acc], dim=1)
        combined_depth = torch.cat([depth], dim=1)

        # Switch images from [H, W, C] to [1, C, H, W] for metrics computations
        image = torch.moveaxis(image, -1, 0)[None, ...]
        rgb = torch.moveaxis(rgb, -1, 0)[None, ...]

        psnr = self.psnr(image, rgb)
        ssim = self.ssim(image, rgb)
        lpips = self.lpips(image, rgb)

        # all of these metrics will be logged as scalars
        metrics_dict = {"psnr": float(psnr.item()), "ssim": float(ssim), "lpips": float(lpips)}  # type: ignore
        # TODO(ethan): return an image dictionary

        images_dict = {
            "img": combined_rgb,
            "accumulation": combined_acc,
            "depth": combined_depth,
        }

        return metrics_dict, images_dict

    def load_state_dict(self, model_state, strict=True):
        # size of these buffers will change
        self.sampler.tree_nodes = model_state['sampler.tree_nodes']
        self.sampler.pers_trans = model_state['sampler.pers_trans']
        self.sampler.tree_visit_cnt = model_state['sampler.tree_visit_cnt']
        self.sampler.milestones_ts = model_state['sampler.milestones_ts']
        super().load_state_dict(model_state, strict=strict)
        # set f2nerf states
        self.sampler.update_f2nerf_states()
        # from IPython import embed;embed()
        # self.field.update_f2nerf_states()