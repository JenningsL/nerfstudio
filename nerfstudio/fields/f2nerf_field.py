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


from typing import Dict, Literal, Optional, Tuple

import torch
from torch import Tensor, nn
from nerfstudio.cameras.rays import RaySamples
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.field_components.activations import _TruncExp
from nerfstudio.field_components.embedding import Embedding
from nerfstudio.field_components.encodings import (HashEncoding, NeRFEncoding,
                                                   SHEncoding)
from nerfstudio.field_components.field_heads import (FieldHeadNames,
                                                     PredNormalsFieldHead,
                                                     SemanticFieldHead,
                                                     TransientDensityFieldHead,
                                                     TransientRGBFieldHead,
                                                     UncertaintyFieldHead)
from nerfstudio.field_components.mlp import MLP
from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.fields.base_field import Field, get_normalized_directions

import py_f2nerf

import types

def state_dict(self, *args, destination=None, prefix='', keep_vars=False):
    states = {prefix+k: v.data for k,v in self.named_parameters().items()}
    states.update({prefix+k: v.data for k,v in self.named_buffers()})
    states.update({prefix+'n_volumes_': self.States()[-2]})
    destination.update(states)
    return destination

def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
    if "field.anchored_field.feat_pool_" in state_dict:
        self.LoadStates([
            state_dict["field.anchored_field.feat_pool_"],
            state_dict["field.anchored_field.prim_pool_"],
            state_dict["field.anchored_field.bias_pool_"],
            state_dict["field.anchored_field.n_volumes_"],
            state_dict["field.anchored_field.mlp_params_"]
        ], 0)

from torch.cuda.amp import custom_bwd
class CustomTruncExp(_TruncExp):
    @staticmethod
    @custom_bwd
    def backward(ctx, g):
        x = ctx.saved_tensors[0]
        return g * torch.exp(x.clamp(-100., 5.))
    
def trunc_exp(x):
    shift = 3.
    return CustomTruncExp.apply(x - shift)

class F2NeRFField(Field):
    """

    Args:
        aabb: parameters of scene aabb bounds
    """

    aabb: Tensor

    def __init__(
        self,
        global_data_pool,
        aabb: Tensor,
        num_images: int,
        num_layers: int = 2,
        hidden_dim: int = 64,
        geo_feat_dim: int = 15,
        num_layers_color: int = 3,
        hidden_dim_color: int = 64,
        appearance_embedding_dim: int = 32,
        use_average_appearance_embedding: bool = False,
        implementation: Literal["tcnn", "torch"] = "tcnn",
    ) -> None:
        super().__init__()

        self.register_buffer("aabb", aabb)

        self.num_images = num_images
        self.appearance_embedding_dim = appearance_embedding_dim
        self.embedding_appearance = Embedding(self.num_images, self.appearance_embedding_dim)
        # init embedding_appearance manually 
        self.embedding_appearance.embedding.weight.data = torch.randn_like(self.embedding_appearance.embedding.weight.data) * 0.1
        
        self.use_average_appearance_embedding = use_average_appearance_embedding
        self.geo_feat_dim = geo_feat_dim

        # # conf = 'exp/ngp_fox/test/record/runtime_config.yaml'
        # data_pool = py_f2nerf.GlobalDataPool(config_path)
        # data_pool.base_exp_dir = 'outputs/poster/f2-nerf'
        # dataset = py_f2nerf.Dataset(data_pool)

        self.anchored_field = py_f2nerf.Hash3DAnchoredField(global_data_pool)
        self.anchored_field.state_dict = types.MethodType(state_dict, self.anchored_field)
        self.anchored_field._load_from_state_dict = types.MethodType(_load_from_state_dict, self.anchored_field)

        # feat_pool_, prim_pool_, bias_pool_, n_volumes_, mlp_params_ = self.anchored_field.States()
        # self.register_buffer("init_feat_pool", feat_pool_.clone())
        # self.register_buffer("init_mlp_params", mlp_params_.clone())
        # self.register_buffer("hash_grid_prim_pool", prim_pool_.clone())
        # self.register_buffer("hash_grid_bias_pool", bias_pool_.clone())
        # self.register_buffer("hash_grid_n_volumes", n_volumes_.clone())
        # self.register_parameter("hash_grid_feat_pool", nn.Parameter(feat_pool_.clone(), requires_grad=True))
        # self.hash_grid_feat_pool.data.uniform_(-1e-2, 1e-2)
        # self.register_parameter("hash_grid_mlp", nn.Parameter(mlp_params_, requires_grad=True))

        # self.mlp_base = MLP(
        #     in_dim=self.anchored_field.feat_dim,
        #     num_layers=num_layers,
        #     layer_width=hidden_dim,
        #     out_dim=1 + self.geo_feat_dim,
        #     activation=nn.ReLU(),
        #     out_activation=None,
        #     implementation=implementation,
        # )

        self.direction_encoding = SHEncoding(
            levels=4,
            implementation=implementation,
        )

        self.mlp_head = MLP(
            in_dim=self.direction_encoding.get_out_dim() + self.geo_feat_dim + self.appearance_embedding_dim,
            num_layers=num_layers_color,
            layer_width=hidden_dim_color,
            out_dim=3,
            activation=nn.ReLU(),
            out_activation=nn.Sigmoid(),
            implementation=implementation,
        )

        self.edge_pts = None
        self.edge_anchors = None
        self.edge_feats = None

    def update_f2nerf_states(self):
        self.anchored_field.LoadStates([
            # self.hash_grid_feat_pool.data,
            self.init_feat_pool,
            self.hash_grid_prim_pool,
            self.hash_grid_bias_pool,
            self.hash_grid_n_volumes,
            self.init_mlp_params
        ], 0)

    def density_fn(self, positions_warp, anchors):
        # feats = self.anchored_field.QueryFeature(positions_warp, anchors[:,0], self.hash_grid_feat_pool)
        feats = self.anchored_field(positions_warp, anchors[:,0])
        # h = self.mlp_base(feats).view(feats.shape[0], -1)
        h = feats
        density_before_activation, base_mlp_out = torch.split(h, [1, self.geo_feat_dim], dim=-1)
        # Rectifying the density with an exponential is much more stable than a ReLU or
        # softplus, because it enables high post-activation (float32) density outputs
        # from smaller internal (float16) parameters.
        density = trunc_exp(density_before_activation.to(positions_warp))
        return density

    def get_density(self, ray_samples: RaySamples) -> Tuple[Tensor, Tensor]:
        """Computes and returns the densities."""
        positions = ray_samples.frustums.get_positions()
        # positions = SceneBox.get_normalized_positions(ray_samples.frustums.get_positions(), self.aabb)
        metadata = ray_samples.metadata
        anchors = metadata["anchors"][:,0]
        positions_warp = metadata["pts_warp"]
        # positions_flat = positions.view(-1, 3)

        # self.anchored_field.QueryFeature has state inside for backwarding
        if self.edge_pts is not None and self.edge_anchors is not None:
            query_pts = torch.cat([positions_warp, self.edge_pts])
            query_anchors = torch.cat([anchors, self.edge_anchors])
        else:
            query_pts = positions_warp
            query_anchors = anchors
        # feats = self.anchored_field.QueryFeature(query_pts, query_anchors, self.hash_grid_feat_pool)
        feats = self.anchored_field(query_pts, query_anchors)
        # h = self.mlp_base(feats)
        h = feats

        if self.edge_pts is not None and self.edge_anchors is not None:
            n_edge_pts = self.edge_pts.shape[0] // 2
            n_scene_pts = positions_warp.shape[0]
            h, edge_feats = torch.split(h, [n_scene_pts, n_edge_pts * 2], dim=0)
            self.edge_feats = edge_feats.view(n_edge_pts, 2, -1)
            self.edge_pts = None
            self.edge_anchors = None

        h = h.view(*ray_samples.frustums.shape, -1)
        density_before_activation, base_mlp_out = torch.split(h, [1, self.geo_feat_dim], dim=-1)
        self._density_before_activation = density_before_activation

        # Rectifying the density with an exponential is much more stable than a ReLU or
        # softplus, because it enables high post-activation (float32) density outputs
        # from smaller internal (float16) parameters.
        density = trunc_exp(density_before_activation.to(positions))
        return density, base_mlp_out

    def get_outputs(
        self, ray_samples: RaySamples, density_embedding: Optional[Tensor] = None
    ) -> Dict[FieldHeadNames, Tensor]:
        assert density_embedding is not None
        outputs = {}
        if ray_samples.camera_indices is None:
            raise AttributeError("Camera indices are not provided.")
        camera_indices = ray_samples.camera_indices.squeeze()
        directions = get_normalized_directions(ray_samples.frustums.directions)
        directions_flat = directions.view(-1, 3)
        d = self.direction_encoding(directions_flat)

        outputs_shape = ray_samples.frustums.directions.shape[:-1]
        
        # appearance
        if self.training:
            embedded_appearance = self.embedding_appearance(camera_indices)
        else:
            if self.use_average_appearance_embedding:
                embedded_appearance = torch.ones(
                    (*directions.shape[:-1], self.appearance_embedding_dim), device=directions.device
                ) * self.embedding_appearance.mean(dim=0)
            else:
                embedded_appearance = torch.zeros(
                    (*directions.shape[:-1], self.appearance_embedding_dim), device=directions.device
                )

        h = torch.cat(
            [
                d,
                density_embedding.view(-1, self.geo_feat_dim),
                embedded_appearance.view(-1, self.appearance_embedding_dim),
            ],
            dim=-1,
        )
        rgb = self.mlp_head(h).view(*outputs_shape, -1).to(directions)
        outputs.update({FieldHeadNames.RGB: rgb})

        return outputs
