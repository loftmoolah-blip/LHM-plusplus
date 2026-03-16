# -*- coding: utf-8 -*-
# @Organization  : Tongyi Lab, Alibaba
# @Author        : Lingteng Qiu
# @Email         : 220019047@link.cuhk.edu.cn
# @Time          : 2025-07-17 23:07:39
# @Function      : efficient point-image transformer block using merge and unmerge strategy

from typing import Any, Dict

import torch
import torch.nn as nn
from diffusers.utils import is_torch_version
from einops import rearrange

from core.models.encoders.sonata import transform
from core.models.encoders.sonata.structure import Point
from core.models.ImageToME.token_graph import (
    bipartite_soft_matching_random2d,
    init_generator,
)
from core.models.PI_transformer import (
    PITransformerA4OBase,
    PITransformerA4OE2EEncoderDecoder,
)
from core.models.transformer_block.transformer_dit import SD3PMMJointTransformerBlock
from core.structures.slidewindows import SlidingAverageTimer


class EfficientPITransformerA4OE2EEncoderDecoderDense(
    PITransformerA4OE2EEncoderDecoder
):
    """
    EfficientPITransformerA4OE2EEncoderDecoderDense implements an efficient variant of the point-image (PI) transformer,
    optimized for dense prediction tasks using merge and unmerge strategies.
    Inherits from PITransformerA4OE2EEncoderDecoder
    and supports configurable point and image backbones, with flexible transformer stacking. Designed for processing
    multi-modality inputs, it allows efficient feature aggregation and dense correspondence between point clouds and images.
    """

    def __init__(
        self,
        point_backbone: Dict[str, Any] = None,
        image_backbone: Dict[str, Any] = None,
        gradient_checkpointing=False,
        merge_ratio=0.5,
        **kwargs,
    ):
        """
        Initializes the EfficientPITransformerA4OE2EEncoderDecoderDense module.

        Args:
            point_backbone (Dict[str, Any], optional): Configuration dictionary for the point backbone model.
            image_backbone (Dict[str, Any], optional): Configuration dictionary for the image backbone model.
            gradient_checkpointing (bool, optional): If True, enables gradient checkpointing for memory savings. Default is False.
            merge_ratio (float, optional): Ratio used for merge strategy in dense PI-Transformer. Default is 0.5.
            **kwargs: Additional keyword arguments, typically including transformer configuration details
                      such as 'img_dim', 'dim', and 'num_heads'.
        """

        super(PITransformerA4OBase, self).__init__()

        self.point_processing = transform.default()
        # 1. build point transformer
        self.build_point_transformer(point_backbone)
        # 2. image transformer
        self.build_image_transformer(image_backbone)

        depth = len(self.point_backbone.enc)
        assert depth - 1 == self.image_backbone.aa_block_num

        # 3. build P-I transformer
        # point-image multi-modality transformer
        self.pi_transformer_enc = nn.ModuleList()
        self.pi_transformer_dec = nn.ModuleList()

        enc_channels = point_backbone["enc_channels"]
        self.enc_patch_size = point_backbone["enc_patch_size"]
        aa_order = image_backbone["aa_order"]

        # enc
        for i in range(1, depth):
            self.pi_transformer_enc.append(
                SD3PMMJointTransformerBlock(
                    pv_dim=enc_channels[i],
                    img_dim=kwargs["img_dim"],
                    dim=kwargs["dim"],
                    num_heads=kwargs["num_heads"],
                    eps=1e-5,
                    context_pre_only=False,
                    qk_norm="rms_norm",
                    mlp_ratio=4.0,
                    require_mapping=True if len(aa_order) == 2 else False,
                )
            )
        # dec
        self.build_decoder(point_backbone, image_backbone, **kwargs)
        self.gradient_checkpointing = gradient_checkpointing

        # freeze model
        freeze_point = kwargs.get("freeze_point", True)
        freeze_image = kwargs.get("freeze_image", True)
        self.only_global_attention(freeze_point, freeze_image)

        # accumulate methods
        self.accumulate = kwargs.get("accumulate", "concat")

        self.random_generator = None
        self.ratio = merge_ratio

    def forward(
        self,
        query_points: dict,
        cond_imgs: torch.Tensor = None,
        motion_emb: torch.Tensor = None,
        **kwargs,
    ) -> dict:
        """
        Forward pass of the transformer model.
        Args:
            query_points (torch.Tensor): Input tensor of shape [N, L, D].
            cond_imgs (torch.Tensor, optional):  [B S P C] from pretrained_models
            motion_emb (torch.Tensor, optional): Modulation tensor of shape [N, D_mod] or None. Defaults to None.  # For SD3_MM_Cond, temb means MotionCLIP
        Returns:
            torch.Tensor: Output tensor of shape [N, L, D].
        """

        def sample_batch(query_points, batch):
            batch_sample = dict()
            for key in query_points.keys():
                if key == "mesh_meta":
                    continue
                batch_sample[key] = query_points[key][batch : batch + 1]

            return batch_sample

        batch_size = cond_imgs.shape[0]
        mesh_meta = query_points["mesh_meta"]

        query_feats_list = []
        img_feats_list = []
        motion_emb_list = []
        pos_list = []

        for batch in range(batch_size):
            sample = sample_batch(query_points, batch)
            sample["mesh_meta"] = mesh_meta
            # ref_imgs_bool
            batch_ref_imgs_bool = kwargs["ref_imgs_bool"][batch]
            batch_cond_imgs = cond_imgs[batch : batch + 1]
            batch_motion_emb = motion_emb[batch : batch + 1]

            valid_cond_imgs = batch_cond_imgs[:, batch_ref_imgs_bool]
            valid_cond_motion_emb = batch_motion_emb[:, batch_ref_imgs_bool].mean(dim=1)

            query_feats, img_feats, update_motion_emb, pos = self._forward_per_batch(
                sample, valid_cond_imgs, valid_cond_motion_emb, **kwargs
            )

            query_feats_list.append(query_feats)
            img_feats_list.append(img_feats)
            motion_emb_list.append(update_motion_emb)
            pos_list.append(pos)

        return dict(
            query_feats=torch.cat(query_feats_list, dim=0),
            img_feats=img_feats_list,
            motion_embs=motion_emb_list,
            pos_embs=pos_list,
        )

    def build_decoder(self, point_backbone, image_backbone, **kwargs):
        """
        Adds decoder transformer blocks for point-image decoding.

        Args:
            point_backbone (dict): Point transformer backbone configuration.
            image_backbone (dict): Image transformer backbone configuration.
            **kwargs: Additional arguments, e.g. 'img_dim', 'dim', 'num_heads'.

        Initializes a stack of SD3PMMJointTransformerBlock decoder layers,
        one per decoder depth stage, using the reversed decoder channels.
        Block configuration is determined by aa_order and input kwargs.
        """

        dec_channels = list(point_backbone["dec_channels"])[::-1]
        depth = len(self.point_backbone.dec)
        aa_order = image_backbone["aa_order"]

        for i in range(depth):
            self.pi_transformer_dec.append(
                SD3PMMJointTransformerBlock(
                    pv_dim=dec_channels[i],
                    img_dim=kwargs["img_dim"],
                    dim=kwargs["dim"],
                    num_heads=kwargs["num_heads"],
                    eps=1e-5,
                    context_pre_only=False if i != depth - 1 else True,
                    qk_norm="rms_norm",
                    mlp_ratio=4.0,
                    require_mapping=True if len(aa_order) == 2 else False,
                )
            )

        self.outnorm = nn.LayerNorm(dec_channels[-1])

    def _build_tokens_graph(self, tokens, tokens_metric, hight, width):
        """
        Builds a token graph for merging and unmerging operations based on the provided token metrics.

        Args:
            tokens (torch.Tensor):
                The input token tensor of shape (batch_size, num_tokens, feature_dim).
                Represents the features for each token in the batch.
            tokens_metric (torch.Tensor):
                Token metric tensor of shape (batch_size, num_views, num_tokens_per_view, metric_dim),
                where metric_dim should be >= 5. This tensor is used to calculate the merging/unmerging relations.
            hight (int):
                The height of the multi-view token's spatial arrangement.
            width (int):
                The width of the multi-view token's spatial arrangement.

        Returns:
            tuple:
                - _merge (callable): function to merge tokens based on soft bipartite matching.
                - _unmerge (callable): function to reverse the merging of tokens.
                - merge_tokens (torch.Tensor): merged token tensor of shape (batch_size, merged_token_count, feature_dim).
        """

        tokens_metric = tokens_metric[:, :, 5:]  # do not choose register tokens
        views = tokens_metric.shape[1]
        tokens_metric = rearrange(tokens_metric, "b s p c -> b (s p) c")

        num_tokens = tokens.shape[1]
        if self.random_generator is None:
            self.random_generator = init_generator(tokens.device)
        elif self.random_generator.device != tokens.device:
            self.random_generator = init_generator(
                tokens.device, fallback=self.random_generator
            )

        _merge, _unmerge = bipartite_soft_matching_random2d(
            tokens_metric,
            int(width * views),
            hight,
            2,
            2,
            int(self.ratio * num_tokens),
            no_rand=False,
            generator=self.random_generator,
        )

        merge_tokens = _merge(tokens)

        return _merge, _unmerge, merge_tokens

    def multi_modality_layer(
        self,
        mv_tokens,
        query_points,
        motion_emb,
        metric,
        depth_i,
        mode="enc",
        use_auto_patch=True,
        size=None,
    ):
        """Process multi-modality features with configurable encoder/decoder transformer.
        Args:
            mv_tokens:          Multi-view tokens [B, S, P, C]
            query_points:       Query points structure with features
            motion_emb:         Motion embeddings
            metric:             metric to tokens graph.
            depth_i:            Current depth index
            mode:               Processing mode: "enc" for encoder or "dec" for decoder

        Returns:
            mv_tokens:          Processed multi-view tokens [B, S, P, C]
            query_points:       Updated query points with new features
        """

        if motion_emb is None:
            raise ValueError("motion_emb cannot be None for multi-modality processing")
        assert size is not None
        hight, width = size

        # 1. Reshape multi-view tokens
        mv_shape = mv_tokens.shape

        mv_p = mv_shape[-2]
        mv_views = mv_shape[1]

        register_tokens, merge_mv_tokens = self._merge_mv(mv_tokens, mode, depth_i)

        # _merge_tokens
        _merge_func, _unmerge_func, merge_mv_tokens = self._build_tokens_graph(
            merge_mv_tokens, metric, hight, width
        )

        # 2. Prepare point ordering and padding
        offset = query_points.offset

        # 3. Prepare point features
        # TODO fix to 10
        auto_patch_size = 10 * mv_p

        if (
            use_auto_patch and offset.item() > auto_patch_size
        ):  # 10 is the expected value when sampling from [1, 2, 4, 8 ,10, 12, 14, 16]
            # TODO fix a bug for Encoder-Decoder framework, we find if use auto patch, when using E-D framework, the graident cannot pass, always leads to nan error.
            pad, unpad, _, patch_size = self.get_padding_and_inverse(
                query_points, auto_patch_size
            )
            order = query_points.mm_order[pad]
            inverse = unpad[query_points.mm_inverse]
            order_feat = query_points.feat[order]
            point_latents = order_feat.view(-1, patch_size, order_feat.shape[-1])
        else:
            order = query_points.mm_order
            inverse = query_points.mm_inverse
            order_feat = query_points.feat[order]
            point_latents = order_feat.unsqueeze(0)

        # 4. Select appropriate transformer
        transformer = (
            self.pi_transformer_enc[depth_i - 1]
            if mode == "enc"
            else self.pi_transformer_dec[depth_i - 1]
        )

        # 5. Process through transformer
        point_latents, merge_mv_tokens = self.forward_multi_modality_layer(
            transformer,
            point_latents,
            merge_mv_tokens,
            motion_emb=motion_emb,
        )

        # 6. demerge tokens
        if merge_mv_tokens is not None:
            mv_tokens = self._demerge_mv(
                merge_mv_tokens, register_tokens, mv_views, _unmerge_func
            )
        else:
            mv_tokens = None

        # 7. Update query points
        points_latents = point_latents.view(-1, order_feat.shape[-1])[inverse]
        query_points.feat = points_latents
        query_points.sparse_conv_feat = query_points.sparse_conv_feat.replace_feature(
            points_latents
        )

        return mv_tokens, query_points

    def _forward_per_batch(self, query_points, cond_imgs, motion_emb=None, **kwargs):
        """
        Forward pass of the transformer model.
        Args:
            query_points (torch.Tensor): Input tensor of shape [N, L, D].
            cond_imgs (torch.Tensor, optional):  [B S P C] from pretrained_models
            motion_emb (torch.Tensor, optional): Modulation tensor of shape [N, D_mod] or None. Defaults to None.  # For SD3_MM_Cond, temb means MotionCLIP
        Returns:
            torch.Tensor: Output tensor of shape [N, L, D].
        """

        # x: [N, L, D]
        # cond: [B S, L_cond, D_cond] or None
        # mod: [N, D_mod] or None

        query_points = self.point_processing(query_points)
        query_points = Point(query_points)
        query_points.cuda()
        query_points = self.point_backbone.serialization_sparisy(query_points)

        # encode first
        query_points = self.point_backbone.enc[0](query_points)

        # prepare for image tokens
        B, S, P, C = cond_imgs.shape

        img_tokens, pos = self.image_backbone.prepare_image_tokens(
            cond_imgs, latent_size=kwargs["latent_size"]
        )

        _, P, C = img_tokens.shape
        hight, width = kwargs["latent_size"]

        frame_idx = 0
        global_idx = 0

        depth = len(self.point_backbone.enc)

        # encode
        for depth_i in range(1, depth):
            # 1. point transformer
            query_points = self.point_backbone.enc[depth_i](query_points)

            # 2. vggt transformer
            for attn_type in self.image_backbone.aa_order:

                if attn_type == "frame":
                    _, frame_idx, frame_intermediates, metric_intermediates = (
                        self.image_backbone._process_frame_attention(
                            img_tokens, B, S, P, C, frame_idx, pos=pos
                        )
                    )
                elif attn_type == "global":
                    _, global_idx, global_intermediates = (
                        self.image_backbone._process_global_attention(
                            img_tokens, B, S, P, C, global_idx, pos=pos
                        )
                    )
                else:
                    raise ValueError(f"Unknown attention type: {attn_type}")

            # 3. modality transformer
            if self.image_backbone.aa_block_num == 2:
                mv_tokens = torch.cat(
                    [frame_intermediates[-1], global_intermediates[-1]], dim=-1
                )
            else:
                mv_tokens = frame_intermediates[-1]
                mv_metric = metric_intermediates[-1]

            mv_tokens, query_points = self.multi_modality_layer(
                mv_tokens,
                query_points,
                motion_emb,
                mv_metric,
                depth_i,
                mode="enc",
                use_auto_patch=False,
                size=(hight, width),
            )
            img_tokens = mv_tokens

        # decode
        depth = len(self.point_backbone.dec)

        for depth_i in range(depth):
            # 1. point transformer
            query_points = self.point_backbone.dec[depth_i](query_points)
            # 2. mm_transformer
            mv_tokens, query_points = self.multi_modality_layer(
                img_tokens,
                query_points,
                motion_emb,
                mv_metric,
                depth_i + 1,
                mode="dec",
                use_auto_patch=False,
                size=(hight, width),
            )

            if mv_tokens is not None:
                img_tokens = mv_tokens

        # inverse to original size
        query_points.feat = self.outnorm(query_points.feat[query_points.inverse])

        return query_points.feat.unsqueeze(0), img_tokens, motion_emb, pos

    def _merge_mv(self, mv_tokens, mode="enc", depth_i=0):
        """
        Merge multi-view tokens by concatenating the sequence and patch dimensions
        when in encoder mode and accumulation strategy is 'concat'.

        Args:
            mv_tokens (Tensor): Input tokens of shape (b, s, p, c)
            img_tokens_size (Optional): Ignored, kept for interface compatibility
            mode (str): Operation mode, e.g., 'enc' for encoder
            depth_i (int): Depth index, used for potential future logic

        Returns:
            Tuple[Tensor, Tensor]:
                - Merged tokens of shape (b, s*p, c)
                - Merge size mask of shape (b, s*p, 1), all ones
        """
        # Only 'concat' accumulation is supported
        if self.accumulate != "concat":
            raise NotImplementedError("Only 'concat' accumulation is supported")

        # In encoder mode, merge the sequence and patch dimensions
        register_tokens, mv_tokens = mv_tokens[:, :, :5], mv_tokens[:, :, 5:]
        mv_tokens = rearrange(mv_tokens, "b s p c -> b (s p) c")

        return register_tokens, mv_tokens

    def _demerge_mv(
        self, merge_mv_tokens, register_tokens, mv_views, unmerge_func, mode="enc"
    ):
        """
        Reverse the merging of multi-view tokens by reshaping back to original
        multi-view layout when in encoder mode and using 'concat' accumulation.

        Args:
            mv_tokens (Tensor): Placeholder for output (not used when merging)
            merge_mv_tokens (Tensor): Merged tokens of shape (b, s*p, c)
            mv_p (int): Number of patches per view (used to unflatten)
            img_tokens_size (Optional): Ignored, kept for interface compatibility
            mode (str): Operation mode, e.g., 'enc' for encoder

        Returns:
            Tensor: Demerged tokens of shape (b, s, p, c)
        """
        # Only 'concat' accumulation is supported
        if self.accumulate != "concat":
            raise NotImplementedError("Only 'concat' accumulation is supported")

        merge_mv_tokens = unmerge_func(merge_mv_tokens)

        # Only demerge in encoder mode; otherwise, keep as-is
        mv_tokens = rearrange(merge_mv_tokens, "b (s p) c -> b s p c", s=mv_views)

        # merge mv tokens with register_tokens
        return torch.cat([register_tokens, mv_tokens], dim=2)

    def forward_multi_modality_layer(self, layer, x, cond, motion_emb=None):
        """
        Runs one multi-modality transformer layer, supporting optional motion embedding and
        gradient checkpointing for memory-efficient training.

        Args:
            layer (nn.Module): The transformer block or layer to apply.
            x (Tensor): Input tokens/features for the point/PI branch.
            cond (Tensor): Conditioning tokens/features (may be image, patch, etc.).
            motion_emb (Tensor, optional): Optional motion embedding for the block.

            Returns:
            Tuple[Tensor, Tensor]: Output tokens/features for point/PI and conditioning branches.
        """

        if self.training and self.gradient_checkpointing:

            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward

            ckpt_kwargs: Dict[str, Any] = (
                {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
            )

            x, cond = torch.utils.checkpoint.checkpoint(
                create_custom_forward(layer),
                x,
                cond,
                motion_emb,
                **ckpt_kwargs,
            )
        else:
            x, cond = layer(
                x,
                cond,
                motion_emb,
            )

        return x, cond

    def only_global_attention(self, freeze_point=True, freeze_image=True):
        """
        Freezes the parameters of the image and/or point backbones so that only global attention modules
        are trainable (i.e., disables gradient computation for the specified components).

        Args:
            freeze_point (bool, optional): If True, freezes all parameters in the point backbone. Default is True.
            freeze_image (bool, optional): If True, freezes all parameters in the image backbone. Default is True.
        """

        # freeze image backbone
        if freeze_image:
            for param in self.image_backbone.parameters():
                param.requires_grad = False

        # freeze point backbone
        if freeze_point:
            for param in self.point_backbone.parameters():
                param.requires_grad = False


if __name__ == "__main__":
    point_backbone = dict(
        type="pv3",
        in_channels=6,  # xyz, normal
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(1, 2),
        enc_depths=(4, 4, 4),
        enc_channels=(64, 256, 512),
        enc_num_head=(4, 8, 16),
        enc_patch_size=(4096, 4096, 2048),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.0,
        shuffle_orders=False,  # for patch interaction.
        pre_norm=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        traceable=True,
        mask_token=False,
        enc_mode=True,
        freeze_encoder=False,
        decoder_concat_stop_feat=2,
    )

    image_backbone = dict(
        type="vggt",
        depth=2,  # depends on the number of point_backbone: (the size of stride)
        aa_order=["frame"],  # only frame attention
    )

    point_backbone = dict(
        type="pv3",
        in_channels=6,  # xyz, normal
        order=("z", "z-trans", "hilbert", "hilbert-trans"),
        stride=(1, 2, 2),
        enc_depths=(4, 4, 4, 4),
        enc_channels=(64, 128, 256, 512),
        enc_num_head=(4, 8, 16, 16),
        enc_patch_size=(4096, 4096, 2048, 1024),
        dec_depths=[4, 4, 4],
        dec_channels=[128, 256, 512],
        dec_num_head=[16, 16, 16],
        dec_patch_size=[4096, 4096, 2048],
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.0,
        shuffle_orders=False,  # for patch interaction.
        pre_norm=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        traceable=True,
        mask_token=False,
        enc_mode=False,
        freeze_encoder=False,
        decoder_concat_stop_feat=2,
    )

    image_backbone = dict(
        type="vggt",
        depth=3,  # depends on the number of point_backbone: (the size of stride)
        aa_order=["frame"],  # only frame attention
    )

    pi_model = EfficientPITransformerA4OE2EEncoderDecoderDense(
        point_backbone=point_backbone,
        image_backbone=image_backbone,
        img_dim=1024,
        dim=1024,
        num_heads=16,
        gradient_checkpointing=True,
        freeze_point=True,
        accumulate="concat",
        merge_ratio=0.5,
    )

    pi_model.cuda()

    from core.models.encoders.sonata import transform
    from core.models.rendering.skinnings.smplx_voxel_skinning import get_query_points

    points = get_query_points()
    view = 16

    img_latents = torch.randn(1, view, 71 * 47, 1024).float().cuda()

    slide_window = SlidingAverageTimer(window_size=10)

    for i in range(100):
        with torch.no_grad():
            slide_window.start()
            out = pi_model(
                points,
                img_latents,
                motion_emb=torch.randn(1, view, 1024).cuda(),
                latent_size=(71, 47),
                ref_imgs_bool=torch.ones(1, view).cuda() > 0,
            )
            torch.cuda.synchronize()
            slide_window.record()
            print(f"{slide_window}")
