from __future__ import annotations

import torch
import torchvision

import math
from typing import Iterable, Optional

import torch
import torch.nn.functional as F


class FOVNetwork(torch.nn.Module):
    """Field of View estimation network."""

    def __init__(
        self,
        num_features: int,
        fov_encoder: Optional[torch.nn.Module] = None,
    ):
        """Initialize the Field of View estimation block.

        Args:
        ----
            num_features: Number of features used.
            fov_encoder: Optional encoder to bring additional network capacity.

        """
        super().__init__()

        # Create FOV head.
        fov_head0 = [
            torch.nn.Conv2d(
                num_features, num_features // 2, kernel_size=3, stride=2, padding=1
            ),  # 128 x 24 x 24
            torch.nn.ReLU(True),
        ]
        fov_head = [
            torch.nn.Conv2d(
                num_features // 2, num_features // 4, kernel_size=3, stride=2, padding=1
            ),  # 64 x 12 x 12
            torch.nn.ReLU(True),
            torch.nn.Conv2d(
                num_features // 4, num_features // 8, kernel_size=3, stride=2, padding=1
            ),  # 32 x 6 x 6
            torch.nn.ReLU(True),
            torch.nn.Conv2d(num_features // 8, 1, kernel_size=6, stride=1, padding=0),
        ]
        if fov_encoder is not None:
            self.encoder = torch.nn.Sequential(
                fov_encoder, torch.nn.Linear(fov_encoder.embed_dim, num_features // 2)
            )
            self.downsample = torch.nn.Sequential(*fov_head0)
        else:
            fov_head = fov_head0 + fov_head
        self.head = torch.nn.Sequential(*fov_head)

    def forward(self, x: torch.Tensor, lowres_feature: torch.Tensor) -> torch.Tensor:
        """Forward the fov network.

        Args:
        ----
            x (torch.Tensor): Input image.
            lowres_feature (torch.Tensor): Low resolution feature.

        Returns:
            The field of view tensor.

        -------
        """
        if hasattr(self, "encoder"):
            x = F.interpolate(
                x,
                size=None,
                scale_factor=0.25,
                mode="bilinear",
                align_corners=False,
            )
            x = self.encoder(x)[:, 1:].permute(0, 2, 1)
            lowres_feature = self.downsample(lowres_feature)
            x = x.reshape_as(lowres_feature) + lowres_feature
        else:
            x = lowres_feature
        return self.head(x)


class DepthProEncoder(torch.nn.Module):
    """DepthPro Encoder.

    An encoder aimed at creating multi-resolution encodings from Vision Transformers.
    """

    def __init__(
        self,
        dims_encoder: Iterable[int],
        patch_encoder: torch.nn.Module,
        image_encoder: torch.nn.Module,
        hook_block_ids: Iterable[int],
        decoder_features: int,
    ):
        """Initialize DepthProEncoder.

        The framework
            1. creates an image pyramid,
            2. generates overlapping patches with a sliding window at each pyramid level,
            3. creates batched encodings via vision transformer backbones,
            4. produces multi-resolution encodings.

        Args:
        ----
            img_size: Backbone image resolution.
            dims_encoder: Dimensions of the encoder at different layers.
            patch_encoder: Backbone used for patches.
            image_encoder: Backbone used for global image encoder.
            hook_block_ids: Hooks to obtain intermediate features for the patch encoder model.
            decoder_features: Number of feature output in the decoder.

        """
        super().__init__()

        self.dims_encoder = list(dims_encoder)
        self.patch_encoder = patch_encoder
        self.image_encoder = image_encoder
        self.hook_block_ids = list(hook_block_ids)

        patch_encoder_embed_dim = patch_encoder.embed_dim
        image_encoder_embed_dim = image_encoder.embed_dim

        self.out_size = int(
            patch_encoder.patch_embed.img_size[0] // patch_encoder.patch_embed.patch_size[0]
        )

        def _create_project_upsample_block(
            dim_in: int,
            dim_out: int,
            upsample_layers: int,
            dim_int: Optional[int] = None,
        ) -> torch.nn.Module:
            if dim_int is None:
                dim_int = dim_out
            # Projection.
            blocks = [
                torch.nn.Conv2d(
                    in_channels=dim_in,
                    out_channels=dim_int,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False,
                )
            ]

            # Upsampling.
            blocks += [
                torch.nn.ConvTranspose2d(
                    in_channels=dim_int if i == 0 else dim_out,
                    out_channels=dim_out,
                    kernel_size=2,
                    stride=2,
                    padding=0,
                    bias=False,
                )
                for i in range(upsample_layers)
            ]

            return torch.nn.Sequential(*blocks)

        self.upsample_latent0 = _create_project_upsample_block(
            dim_in=patch_encoder_embed_dim,
            dim_int=self.dims_encoder[0],
            dim_out=decoder_features,
            upsample_layers=3,
        )
        self.upsample_latent1 = _create_project_upsample_block(
            dim_in=patch_encoder_embed_dim, dim_out=self.dims_encoder[0], upsample_layers=2
        )

        self.upsample0 = _create_project_upsample_block(
            dim_in=patch_encoder_embed_dim, dim_out=self.dims_encoder[1], upsample_layers=1
        )
        self.upsample1 = _create_project_upsample_block(
            dim_in=patch_encoder_embed_dim, dim_out=self.dims_encoder[2], upsample_layers=1
        )
        self.upsample2 = _create_project_upsample_block(
            dim_in=patch_encoder_embed_dim, dim_out=self.dims_encoder[3], upsample_layers=1
        )

        self.upsample_lowres = torch.nn.ConvTranspose2d(
            in_channels=image_encoder_embed_dim,
            out_channels=self.dims_encoder[3],
            kernel_size=2,
            stride=2,
            padding=0,
            bias=True,
        )
        self.fuse_lowres = torch.nn.Conv2d(
            in_channels=(self.dims_encoder[3] + self.dims_encoder[3]),
            out_channels=self.dims_encoder[3],
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        # Obtain intermediate outputs of the blocks.
        self.patch_encoder.blocks[self.hook_block_ids[0]].register_forward_hook(
            self._hook0
        )
        self.patch_encoder.blocks[self.hook_block_ids[1]].register_forward_hook(
            self._hook1
        )

    def _hook0(self, model, input, output):
        self.backbone_highres_hook0 = output

    def _hook1(self, model, input, output):
        self.backbone_highres_hook1 = output

    @property
    def img_size(self) -> int:
        """Return the full image size of the SPN network."""
        return self.patch_encoder.patch_embed.img_size[0] * 4

    def _create_pyramid(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Create a 3-level image pyramid."""
        # Original resolution: 1536 by default.
        x0 = x

        # Middle resolution: 768 by default.
        x1 = F.interpolate(
            x, size=None, scale_factor=0.5, mode="bilinear", align_corners=False
        )

        # Low resolution: 384 by default, corresponding to the backbone resolution.
        x2 = F.interpolate(
            x, size=None, scale_factor=0.25, mode="bilinear", align_corners=False
        )

        return x0, x1, x2

    def split(self, x: torch.Tensor, overlap_ratio: float = 0.25) -> torch.Tensor:
        """Split the input into small patches with sliding window."""
        patch_size = 384
        patch_stride = int(patch_size * (1 - overlap_ratio))

        image_size = x.shape[-1]
        steps = int(math.ceil((image_size - patch_size) / patch_stride)) + 1

        x_patch_list = []
        for j in range(steps):
            j0 = j * patch_stride
            j1 = j0 + patch_size

            for i in range(steps):
                i0 = i * patch_stride
                i1 = i0 + patch_size
                x_patch_list.append(x[..., j0:j1, i0:i1])

        return torch.cat(x_patch_list, dim=0)

    def merge(self, x: torch.Tensor, batch_size: int, padding: int = 3) -> torch.Tensor:
        """Merge the patched input into a image with sliding window."""
        steps = int(math.sqrt(x.shape[0] // batch_size))

        idx = 0

        output_list = []
        for j in range(steps):
            output_row_list = []
            for i in range(steps):
                output = x[batch_size * idx : batch_size * (idx + 1)]

                if j != 0:
                    output = output[..., padding:, :]
                if i != 0:
                    output = output[..., :, padding:]
                if j != steps - 1:
                    output = output[..., :-padding, :]
                if i != steps - 1:
                    output = output[..., :, :-padding]

                output_row_list.append(output)
                idx += 1

            output_row = torch.cat(output_row_list, dim=-1)
            output_list.append(output_row)
        output = torch.cat(output_list, dim=-2)
        return output

    def reshape_feature(
        self, embeddings: torch.Tensor, width, height, cls_token_offset=1
    ):
        """Discard class token and reshape 1D feature map to a 2D grid."""
        b, hw, c = embeddings.shape

        # Remove class token.
        if cls_token_offset > 0:
            embeddings = embeddings[:, cls_token_offset:, :]

        # Shape: (batch, height, width, dim) -> (batch, dim, height, width)
        embeddings = embeddings.reshape(b, height, width, c).permute(0, 3, 1, 2)
        return embeddings

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Encode input at multiple resolutions.

        Args:
        ----
            x (torch.Tensor): Input image.

        Returns:
        -------
            Multi resolution encoded features.

        """
        batch_size = x.shape[0]

        # Step 0: create a 3-level image pyramid.
        x0, x1, x2 = self._create_pyramid(x)

        # Step 1: split to create batched overlapped mini-images at the backbone (BeiT/ViT/Dino)
        # resolution.
        # 5x5 @ 384x384 at the highest resolution (1536x1536).
        x0_patches = self.split(x0, overlap_ratio=0.25)
        # 3x3 @ 384x384 at the middle resolution (768x768).
        x1_patches = self.split(x1, overlap_ratio=0.5)
        # 1x1 # 384x384 at the lowest resolution (384x384).
        x2_patches = x2

        # Concatenate all the sliding window patches and form a batch of size (35=5x5+3x3+1x1).
        x_pyramid_patches = torch.cat(
            (x0_patches, x1_patches, x2_patches),
            dim=0,
        )

        # Step 2: Run the backbone (BeiT) model and get the result of large batch size.
        x_pyramid_encodings = self.patch_encoder(x_pyramid_patches)
        x_pyramid_encodings = self.reshape_feature(
            x_pyramid_encodings, self.out_size, self.out_size
        )
        
        # Step 3: merging.
        # Merge highres latent encoding.
        x_latent0_encodings = self.reshape_feature(
            self.backbone_highres_hook0,
            self.out_size,
            self.out_size,
        )
        x_latent0_features = self.merge(
            x_latent0_encodings[: batch_size * 5 * 5], batch_size=batch_size, padding=3
        )

        x_latent1_encodings = self.reshape_feature(
            self.backbone_highres_hook1,
            self.out_size,
            self.out_size,
        )
        x_latent1_features = self.merge(
            x_latent1_encodings[: batch_size * 5 * 5], batch_size=batch_size, padding=3
        )

        # Split the 35 batch size from pyramid encoding back into 5x5+3x3+1x1.
        x0_encodings, x1_encodings, x2_encodings = torch.split(
            x_pyramid_encodings,
            [len(x0_patches), len(x1_patches), len(x2_patches)],
            dim=0,
        )

        # 96x96 feature maps by merging 5x5 @ 24x24 patches with overlaps.
        x0_features = self.merge(x0_encodings, batch_size=batch_size, padding=3)

        # 48x84 feature maps by merging 3x3 @ 24x24 patches with overlaps.
        x1_features = self.merge(x1_encodings, batch_size=batch_size, padding=6)

        # 24x24 feature maps.
        x2_features = x2_encodings

        # Apply the image encoder model.
        x_global_features = self.image_encoder(x2_patches)
        x_global_features = self.reshape_feature(
            x_global_features, self.out_size, self.out_size
        )
        
        # Upsample feature maps.
        x_latent0_features = self.upsample_latent0(x_latent0_features)
        x_latent1_features = self.upsample_latent1(x_latent1_features)

        x0_features = self.upsample0(x0_features)
        x1_features = self.upsample1(x1_features)
        x2_features = self.upsample2(x2_features)

        x_global_features = self.upsample_lowres(x_global_features)
        x_global_features = self.fuse_lowres(
            torch.cat((x2_features, x_global_features), dim=1)
        )

        return [
            x_latent0_features,
            x_latent1_features,
            x0_features,
            x1_features,
            x_global_features,
        ]


class MultiresConvDecoder(torch.nn.Module):
    """Decoder for multi-resolution encodings."""

    def __init__(
        self,
        dims_encoder: Iterable[int],
        dim_decoder: int,
    ):
        """Initialize multiresolution convolutional decoder.

        Args:
        ----
            dims_encoder: Expected dims at each level from the encoder.
            dim_decoder: Dim of decoder features.

        """
        super().__init__()
        self.dims_encoder = list(dims_encoder)
        self.dim_decoder = dim_decoder
        self.dim_out = dim_decoder

        num_encoders = len(self.dims_encoder)

        # At the highest resolution, i.e. level 0, we apply projection w/ 1x1 convolution
        # when the dimensions mismatch. Otherwise we do not do anything, which is
        # the default behavior of monodepth.
        conv0 = (
            torch.nn.Conv2d(self.dims_encoder[0], dim_decoder, kernel_size=1, bias=False)
            if self.dims_encoder[0] != dim_decoder
            else torch.nn.Identity()
        )

        convs = [conv0]
        for i in range(1, num_encoders):
            convs.append(
                torch.nn.Conv2d(
                    self.dims_encoder[i],
                    dim_decoder,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                )
            )

        self.convs = torch.nn.ModuleList(convs)

        fusions = []
        for i in range(num_encoders):
            fusions.append(
                FeatureFusionBlock2d(
                    num_features=dim_decoder,
                    deconv=(i != 0),
                    batch_norm=False,
                )
            )
        self.fusions = torch.nn.ModuleList(fusions)

    def forward(self, encodings: torch.Tensor) -> torch.Tensor:
        """Decode the multi-resolution encodings."""
        num_levels = len(encodings)
        num_encoders = len(self.dims_encoder)

        if num_levels != num_encoders:
            raise ValueError(
                f"Got encoder output levels={num_levels}, expected levels={num_encoders+1}."
            )

        # Project features of different encoder dims to the same decoder dim.
        # Fuse features from the lowest resolution (num_levels-1)
        # to the highest (0).
        features = self.convs[-1](encodings[-1])
        lowres_features = features
        features = self.fusions[-1](features)
        for i in range(num_levels - 2, -1, -1):
            features_i = self.convs[i](encodings[i])
            features = self.fusions[i](features, features_i)
        return features, lowres_features


class ResidualBlock(torch.nn.Module):
    """Generic implementation of residual blocks.

    This implements a generic residual block from
        He et al. - Identity Mappings in Deep Residual Networks (2016),
        https://arxiv.org/abs/1603.05027
    which can be further customized via factory functions.
    """

    def __init__(self, residual: torch.nn.Module, shortcut: torch.nn.Module | None = None) -> None:
        """Initialize ResidualBlock."""
        super().__init__()
        self.residual = residual
        self.shortcut = shortcut

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply residual block."""
        delta_x = self.residual(x)

        if self.shortcut is not None:
            x = self.shortcut(x)

        return x + delta_x


class FeatureFusionBlock2d(torch.nn.Module):
    """Feature fusion for DPT."""

    def __init__(
        self,
        num_features: int,
        deconv: bool = False,
        batch_norm: bool = False,
    ):
        """Initialize feature fusion block.

        Args:
        ----
            num_features: Input and output dimensions.
            deconv: Whether to use deconv before the final output conv.
            batch_norm: Whether to use batch normalization in resnet blocks.

        """
        super().__init__()

        self.resnet1 = self._residual_block(num_features, batch_norm)
        self.resnet2 = self._residual_block(num_features, batch_norm)

        self.use_deconv = deconv
        if deconv:
            self.deconv = torch.nn.ConvTranspose2d(
                in_channels=num_features,
                out_channels=num_features,
                kernel_size=2,
                stride=2,
                padding=0,
                bias=False,
            )

        self.out_conv = torch.nn.Conv2d(
            num_features,
            num_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
        )

        self.skip_add = torch.nn.quantized.FloatFunctional()

    def forward(self, x0: torch.Tensor, x1: torch.Tensor | None = None) -> torch.Tensor:
        """Process and fuse input features."""
        x = x0

        if x1 is not None:
            res = self.resnet1(x1)
            x = self.skip_add.add(x, res)

        x = self.resnet2(x)

        if self.use_deconv:
            x = self.deconv(x)
        x = self.out_conv(x)

        return x

    @staticmethod
    def _residual_block(num_features: int, batch_norm: bool):
        """Create a residual block."""

        def _create_block(dim: int, batch_norm: bool) -> list[torch.nn.Module]:
            layers = [
                torch.nn.ReLU(False),
                torch.nn.Conv2d(
                    num_features,
                    num_features,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=not batch_norm,
                ),
            ]
            if batch_norm:
                layers.append(nn.BatchNorm2d(dim))
            return layers

        residual = torch.nn.Sequential(
            *_create_block(dim=num_features, batch_norm=batch_norm),
            *_create_block(dim=num_features, batch_norm=batch_norm),
        )
        return ResidualBlock(residual)


class Cnn3d(torch.nn.Module):
    def __init__(self, in_c):
        super().__init__()

        channels = [64, 64, 128, 64, 64]

        self.stem = torch.nn.Sequential(
            ConvBnRelu3d(in_c, channels[0], ks=1, padding=0),
            ResBlock3d(channels[0]),
        )
        self.conv1x1_1 = ConvBnRelu3d(channels[0], channels[1])
        self.down1 = torch.nn.Sequential(
            ResBlock3d(channels[1]),
            ResBlock3d(channels[1]),
        )
        self.conv1x1_2 = ConvBnRelu3d(channels[1], channels[2])
        self.down2 = torch.nn.Sequential(
            ResBlock3d(channels[2]),
            ResBlock3d(channels[2]),
        )
        self.up1 = torch.nn.Sequential(
            ConvBnRelu3d(channels[2] + channels[1], channels[3]),
            ResBlock3d(channels[3]),
            ResBlock3d(channels[3]),
        )
        self.up2 = torch.nn.Sequential(
            ConvBnRelu3d(channels[3] + channels[0], channels[4]),
            ResBlock3d(channels[4]),
            ResBlock3d(channels[4]),
        )
        self.up3 = torch.nn.Sequential(
            ConvBnRelu3d(channels[4] + in_c, channels[4]),
            ResBlock3d(channels[4]),
            ResBlock3d(channels[4]),
        )
        self.out_c = channels[4]

    def forward(self, x, _):
        x0 = self.stem(x)
        x1 = torch.nn.functional.max_pool3d(self.conv1x1_1(x0), 2)
        x1 = self.down1(x1)
        out = torch.nn.functional.max_pool3d(self.conv1x1_2(x1), 2)
        out = self.down2(out)
        out = torch.nn.functional.interpolate(out, scale_factor=2, mode="nearest")
        out = torch.cat((out, x1), dim=1)
        out = self.up1(out)
        out = torch.nn.functional.interpolate(out, scale_factor=2, mode="nearest")
        out = torch.cat((out, x0), dim=1)
        out = self.up2(out)
        out = torch.cat((out, x), dim=1)
        out = self.up3(out)
        return out


class Cnn2d(torch.nn.Module):
    def __init__(self, out_dim=64):
        super().__init__()

        channel_mean = [0.485, 0.456, 0.406]
        channel_std = [0.229, 0.224, 0.225]
        self.normalize = torchvision.transforms.Normalize(channel_mean, channel_std)

        weights = torchvision.models.EfficientNet_V2_S_Weights.DEFAULT
        backbone = torchvision.models.efficientnet_v2_s(weights=weights, progress=True)

        self.conv0 = backbone.features[:3]
        self.conv1 = backbone.features[3]
        self.conv2 = backbone.features[4]

        self.out0 = torch.nn.Sequential(
            torch.nn.Conv2d(48, out_dim, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(out_dim),
            torch.nn.LeakyReLU(True),
        )

        self.out1 = torch.nn.Sequential(
            torch.nn.Conv2d(64, out_dim, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(out_dim),
            torch.nn.LeakyReLU(True),
        )

        self.out2 = torch.nn.Sequential(
            torch.nn.Conv2d(128, out_dim, 3, padding=1, bias=False),
            torch.nn.BatchNorm2d(out_dim),
            torch.nn.LeakyReLU(True),
        )

        self.out3 = ResBlock2d(out_dim)

    def forward(self, x):
        x = self.normalize(x)

        x = self.conv0(x)
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)

        x = self.out0(x)
        conv1 = self.out1(conv1)
        conv2 = self.out2(conv2)

        conv1 = torch.nn.functional.interpolate(
            conv1, scale_factor=2, mode="bilinear", align_corners=False
        )
        conv2 = torch.nn.functional.interpolate(
            conv2, scale_factor=4, mode="bilinear", align_corners=False
        )

        x += conv1
        x += conv2

        return self.out3(x)


class FeatureFusion(torch.nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.out_c = in_c
        self.bn = torch.nn.BatchNorm3d(self.out_c)

    def forward(self, x, valid):
        counts = torch.sum(valid, dim=1, keepdim=True)
        counts.masked_fill_(counts == 0, 1)
        x.masked_fill_(~valid[:, :, None], 0)
        x /= counts[:, :, None]
        mean = x.sum(dim=1)

        return self.bn(mean)


class ResBlock(torch.nn.Module):
    def forward(self, x):
        out = self.net(x)
        out += x
        torch.nn.functional.leaky_relu_(out)
        return out


class ResBlock3d(ResBlock):
    def __init__(self, c, ks=3, padding=1):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv3d(c, c, ks, bias=False, padding=padding),
            torch.nn.BatchNorm3d(c),
            torch.nn.LeakyReLU(True),
            torch.nn.Conv3d(c, c, ks, bias=False, padding=padding),
            torch.nn.BatchNorm3d(c),
        )


class ResBlock2d(ResBlock):
    def __init__(self, c, ksize=3, padding=1):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(c, c, ksize, bias=False, padding=padding),
            torch.nn.BatchNorm2d(c),
            torch.nn.LeakyReLU(True),
            torch.nn.Conv2d(c, c, ksize, bias=False, padding=padding),
            torch.nn.BatchNorm2d(c),
        )


class ResBlock1d(ResBlock):
    def __init__(self, c):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv1d(c, c, 1, bias=False),
            torch.nn.BatchNorm1d(c),
            torch.nn.LeakyReLU(True),
            torch.nn.Conv1d(c, c, 1, bias=False),
            torch.nn.BatchNorm1d(c),
        )


class ConvBnRelu3d(torch.nn.Module):
    def __init__(self, in_c, out_c, ks=3, padding=1):
        super().__init__()

        self.net = torch.nn.Sequential(
            torch.nn.Conv3d(in_c, out_c, ks, padding=padding, bias=False),
            torch.nn.BatchNorm3d(out_c),
            torch.nn.LeakyReLU(True),
        )

    def forward(self, x):
        return self.net(x)
