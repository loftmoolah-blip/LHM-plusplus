import pdb
import sys

sys.path.append("./")
import numpy as np
import torch
import torch.nn as nn

from lib.pointops.functions import pointops
from core.models.encoders.pointtransformer.pointtransformer_seg import (
    PointTransformerBlock,
    TransitionDown,
    TransitionUp,
)


class PointEmbed(nn.Module):
    def __init__(self, hidden_dim=48):
        super().__init__()

        assert hidden_dim % 6 == 0

        self.embedding_dim = hidden_dim
        e = torch.pow(2, torch.arange(self.embedding_dim // 6)).float() * np.pi
        e = torch.stack(
            [
                torch.cat(
                    [
                        e,
                        torch.zeros(self.embedding_dim // 6),
                        torch.zeros(self.embedding_dim // 6),
                    ]
                ),
                torch.cat(
                    [
                        torch.zeros(self.embedding_dim // 6),
                        e,
                        torch.zeros(self.embedding_dim // 6),
                    ]
                ),
                torch.cat(
                    [
                        torch.zeros(self.embedding_dim // 6),
                        torch.zeros(self.embedding_dim // 6),
                        e,
                    ]
                ),
            ]
        )

        self.register_buffer("basis", e)  # 3 x 16

        self.output_channel = 3 + 3 * 16

    @staticmethod
    def embed(input, basis):
        projections = torch.einsum("bnd,de->bne", input, basis)
        embeddings = torch.cat([projections.sin(), projections.cos()], dim=2)

        return embeddings

    def forward(self, input):
        # input: B x N x 3
        embed = self.embed(input, self.basis)
        return input, embed


class PointTransformerEncoder(nn.Module):
    def __init__(
        self,
        c=3,
        planes=[128, 256, 1024],
        strides=[1, 4, 4],
        nsamples=[8, 16, 16],
        block=PointTransformerBlock,
        blocks=[2, 3, 4],
    ):
        """Point Transformer V1."""
        super().__init__()

        self.pointembed = PointEmbed()
        self.c = self.pointembed.output_channel
        self.in_planes, planes = self.c, planes
        share_planes = 8
        self.encoder_nets = nn.ModuleList()

        for plane, block_size, stride, nsample in zip(
            planes, blocks, strides, nsamples
        ):
            self.encoder_nets.append(
                self._make_enc(
                    block,
                    plane,
                    block_size,
                    share_planes,
                    stride=stride,
                    nsample=nsample,
                )
            )

    def _make_enc(self, block, planes, blocks, share_planes=8, stride=1, nsample=16):
        layers = []
        layers.append(
            TransitionDown(self.in_planes, planes * block.expansion, stride, nsample)
        )
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.in_planes, self.in_planes, share_planes, nsample=nsample)
            )
        return nn.Sequential(*layers)

    def _forward(self, pxo):
        p0, x0, o0 = pxo  # (n, 3), (n, c), (b)
        x0 = p0 if self.c == 3 else torch.cat((p0, x0), 1)

        output_list = []
        layer_size = len(self.encoder_nets)

        inputs = [p0, x0, o0]

        for layer_i in range(layer_size):
            inputs = self.encoder_nets[layer_i](inputs)
            output_list.append(inputs)

        return output_list

    def forward(self, x):
        x, pcl_embed = self.pointembed(x)
        x = x.squeeze(0).contiguous()
        pcl_embed = pcl_embed.squeeze(0).contiguous()
        N = x.shape[0]
        o = torch.IntTensor([N]).to(x.device)

        output_list = self._forward([x, pcl_embed, o])

        return output_list


class PointTransformerDecoder(nn.Module):
    def __init__(
        self, planes=[128, 256, 1024], nsample=[8, 16, 16], block=PointTransformerBlock
    ):
        """Point Transformer V1."""
        super().__init__()

        self.in_planes, planes = planes[-1], planes
        share_planes = 8

        self.decoder_nets = nn.ModuleList()

        self.layer_size = len(planes)

        for layer_i, (plane_dim, sample_size) in enumerate(
            zip(planes[::-1], nsample[::-1])
        ):
            self.decoder_nets.append(
                self._make_dec(
                    block,
                    plane_dim,
                    2,
                    share_planes,
                    nsample=sample_size,
                    is_head=(layer_i == 0),
                )
            )
        self.decoder_nets = self.decoder_nets[::-1]

    def _make_dec(
        self, block, planes, blocks, share_planes=8, nsample=16, is_head=False
    ):
        layers = []
        layers.append(
            TransitionUp(self.in_planes, None if is_head else planes * block.expansion)
        )
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(self.in_planes, self.in_planes, share_planes, nsample=nsample)
            )
        return nn.Sequential(*layers)

    def forward(self, feats, encoder_feats):

        decoder_feat = None

        for layer_idx, (encoder_feat, layer) in enumerate(
            zip(encoder_feats[::-1], self.decoder_nets[::-1])
        ):
            if layer_idx == 0:
                p, _, o = encoder_feat
                feats = layer[1:]([p, layer[0]([p, feats, o]), o])[1]
                decoder_feat = [p, feats, o]
            else:
                p, _, o = encoder_feat
                feats = layer[1:]([p, layer[0](encoder_feat, decoder_feat), o])[1]
                decoder_feat = [p, feats, o]

        return feats


if __name__ == "__main__":

    encoder_model = PointTransformerEncoder()
    decoder_model = PointTransformerDecoder()

    x = torch.randn(1, 20000, 3).cuda()

    encoder_model = encoder_model.cuda()
    decoder_model = decoder_model.cuda()

    out = encoder_model(x)

    ret_feat = decoder_model(out[-1][1], out)
    print(ret_feat.shape)
