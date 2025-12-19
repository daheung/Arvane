"""
Point Transformer - V3 Mode2 - Sonata
Pointcept detached version

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

# Copyright (c) Meta Platforms, Inc. and affiliates.
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


import os
import torch
import spconv.pytorch as spconv
import numpy as np

from typing import Any, Dict, Tuple, Optional, Callable
from torch.nn.init import trunc_normal_
from huggingface_hub import PyTorchModelHubMixin
from numpy.typing import NDArray
from open3d.geometry import TriangleMesh

from source.utils import np_to_torch_dtype
from source.predictor.ptv3.defines import CLASS_COLOR_20
from source.predictor.ptv3.transform import default
from source.predictor.ptv3.utils import (
    extract_point_from_file,
    extract_point_from_glb,
    load_ckpt, 
    load, 
)
from source.predictor.ptv3.module import (
    PointModule,
    Embedding,
    PointSequential,
    GridPooling,
    Block,
    GridUnpooling,
    Point    
)

class PointTransformerV3(PointModule, PyTorchModelHubMixin):
    def __init__(
        self,
        in_channels=6,
        order=("z", "z-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(3, 3, 3, 12, 3),
        enc_channels=(48, 96, 192, 384, 512),
        enc_num_head=(3, 6, 12, 24, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        dec_depths=(3, 3, 3, 3),
        dec_channels=(96, 96, 192, 384),
        dec_num_head=(6, 6, 12, 32),
        dec_patch_size=(1024, 1024, 1024, 1024),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        layer_scale=None,
        pre_norm=True,
        shuffle_orders=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        traceable=False,
        mask_token=False,
        enc_mode=False,
        freeze_encoder=False,
    ):
        super().__init__()
        self.num_stages = len(enc_depths)
        self.order = [order] if isinstance(order, str) else order
        self.enc_mode = enc_mode
        self.shuffle_orders = shuffle_orders
        self.freeze_encoder = freeze_encoder

        assert self.num_stages == len(stride) + 1
        assert self.num_stages == len(enc_depths)
        assert self.num_stages == len(enc_channels)
        assert self.num_stages == len(enc_num_head)
        assert self.num_stages == len(enc_patch_size)
        assert self.enc_mode or self.num_stages == len(dec_depths) + 1
        assert self.enc_mode or self.num_stages == len(dec_channels) + 1
        assert self.enc_mode or self.num_stages == len(dec_num_head) + 1
        assert self.enc_mode or self.num_stages == len(dec_patch_size) + 1

        # normalization layer
        ln_layer = torch.nn.LayerNorm
        # activation layers
        act_layer = torch.nn.GELU

        self.embedding = Embedding(
            in_channels=in_channels,
            embed_channels=enc_channels[0],
            norm_layer=ln_layer,
            act_layer=act_layer,
            mask_token=mask_token,
        )

        # encoder
        enc_drop_path = [
            x.item() for x in torch.linspace(0, drop_path, sum(enc_depths))
        ]
        self.enc = PointSequential()
        for s in range(self.num_stages):
            enc_drop_path_ = enc_drop_path[
                sum(enc_depths[:s]) : sum(enc_depths[: s + 1])
            ]
            enc = PointSequential()
            if s > 0:
                enc.add(
                    GridPooling(
                        in_channels=enc_channels[s - 1],
                        out_channels=enc_channels[s],
                        stride=stride[s - 1],
                        norm_layer=ln_layer,
                        act_layer=act_layer,
                    ),
                    name="down",
                )
            for i in range(enc_depths[s]):
                enc.add(
                    Block(
                        channels=enc_channels[s],
                        num_heads=enc_num_head[s],
                        patch_size=enc_patch_size[s],
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        attn_drop=attn_drop,
                        proj_drop=proj_drop,
                        drop_path=enc_drop_path_[i],
                        layer_scale=layer_scale,
                        norm_layer=ln_layer,
                        act_layer=act_layer,
                        pre_norm=pre_norm,
                        order_index=i % len(self.order),
                        cpe_indice_key=f"stage{s}",
                        enable_rpe=enable_rpe,
                        enable_flash=enable_flash,
                        upcast_attention=upcast_attention,
                        upcast_softmax=upcast_softmax,
                    ),
                    name=f"block{i}",
                )
            if len(enc) != 0:
                self.enc.add(module=enc, name=f"enc{s}")

        # decoder
        if not self.enc_mode:
            dec_drop_path = [
                x.item() for x in torch.linspace(0, drop_path, sum(dec_depths))
            ]
            self.dec = PointSequential()
            dec_channels = list(dec_channels) + [enc_channels[-1]]
            for s in reversed(range(self.num_stages - 1)):
                dec_drop_path_ = dec_drop_path[
                    sum(dec_depths[:s]) : sum(dec_depths[: s + 1])
                ]
                dec_drop_path_.reverse()
                dec = PointSequential()
                dec.add(
                    GridUnpooling(
                        in_channels=dec_channels[s + 1],
                        skip_channels=enc_channels[s],
                        out_channels=dec_channels[s],
                        norm_layer=ln_layer,
                        act_layer=act_layer,
                        traceable=traceable,
                    ),
                    name="up",
                )
                for i in range(dec_depths[s]):
                    dec.add(
                        Block(
                            channels=dec_channels[s],
                            num_heads=dec_num_head[s],
                            patch_size=dec_patch_size[s],
                            mlp_ratio=mlp_ratio,
                            qkv_bias=qkv_bias,
                            qk_scale=qk_scale,
                            attn_drop=attn_drop,
                            proj_drop=proj_drop,
                            drop_path=dec_drop_path_[i],
                            layer_scale=layer_scale,
                            norm_layer=ln_layer,
                            act_layer=act_layer,
                            pre_norm=pre_norm,
                            order_index=i % len(self.order),
                            cpe_indice_key=f"stage{s}",
                            enable_rpe=enable_rpe,
                            enable_flash=enable_flash,
                            upcast_attention=upcast_attention,
                            upcast_softmax=upcast_softmax,
                        ),
                        name=f"block{i}",
                    )
                self.dec.add(module=dec, name=f"dec{s}")
        if self.freeze_encoder:
            for p in self.embedding.parameters():
                p.requires_grad = False
            for p in self.enc.parameters():
                p.requires_grad = False
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, torch.nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, spconv.SubMConv3d):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, data_dict):
        point = Point(data_dict)
        point = self.embedding(point)

        point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
        point.sparsify()

        point = self.enc(point)
        if not self.enc_mode:
            point = self.dec(point)
        return point

class SegHead(torch.nn.Module):
    def __init__(self, backbone_out_channels, num_classes):
        super(SegHead, self).__init__()
        self.seg_head = torch.nn.Linear(backbone_out_channels, num_classes)

    def forward(self, x):
        return self.seg_head(x)
    
class ExtractPredictor:
    def __init__(
        self, 
        device: torch.device,
        model_name: str = "sonata"
    ):
        self.device = device
        
        self.model = self._load_predictor(name=model_name, device=device)
        self.model.eval()

        self.ckpt = load_ckpt(
            "sonata_linear_prob_head_sc", repo_id="facebook/sonata"
        )

        # Load default data transform pipeline    
        self.transform = default(infer_mode="train", grid_size=0.02)

        self.seg_head = SegHead(**self.ckpt["config"])
        self.seg_head.load_state_dict(self.ckpt["state_dict"])
        self.seg_head.eval()
        self.seg_head.to(device=device)

    def init(self):
        ...

    def infer(
        self, 
        glb_bytes_or_file: TriangleMesh | str, 
        return_color: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # extract_point_considering_param: Callable[[TriangleMesh | str, Optional[Tuple[int, int, int]]], Dict[str, NDArray]] = extract_point_from_file if isinstance(glb_bytes_or_file, str) else extract_point_from_glb
        extract_point_considering_param = extract_point_from_file if isinstance(glb_bytes_or_file, str) else extract_point_from_glb
        points = extract_point_considering_param(
            glb_bytes_or_file, [255, 255, 255]
        )

        import pdb; pdb.set_trace()
        points = self.transform(points)
        
        with torch.inference_mode():
            for key in points.keys():
                if isinstance(points[key], torch.Tensor):
                    points[key] = points[key].to(device=self.device)

            # model forward:
            points = self.model(points)
            while "pooling_parent" in points.keys():
                assert "pooling_inverse" in points.keys()
                parent = points.pop("pooling_parent")
                inverse = points.pop("pooling_inverse")
                parent.feat = torch.cat([parent.feat, points.feat[inverse]], dim=-1)
                points = parent

            feat = points.feat
            seg_logits = self.seg_head(feat)
            pred = seg_logits.argmax(dim=-1).data.cpu()
            color: Optional[torch.Tensor] = torch.tensor(CLASS_COLOR_20)[pred] if return_color else None

        ret_value: Tuple[torch.Tensor, Optional[torch.Tensor]] = (
            torch.tensor(points.coord, dtype=points.coord.dtype), 
            color
        )
        return ret_value
    
    def _load_predictor(
        self, 
        name: str,
        device: torch.device,
        repo_id="facebook/sonata",
        download_root: str = None,
        custom_config: dict = None,
    ) -> Any | PointTransformerV3:
        return load(
            name=name,
            repo_id=repo_id,
            download_root=download_root,
            custom_config=custom_config,
        ).to(device=device)