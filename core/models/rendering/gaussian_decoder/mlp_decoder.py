# -*- coding: utf-8 -*-
# @Organization  : Tongyi Lab, Alibaba
# @Author        : Lingteng Qiu
# @Email         : 220019047@link.cuhk.edu.cn
# @Time          : 2025-06-04 20:43:18
# @Function      : MLP Gaussian Decoder

import omegaconf
import torch
import torch.nn as nn

from core.models.rendering.utils.utils import trunc_exp
from core.models.utils import LinerParameterTuner, StaticParameterTuner
from core.outputs.output import GaussianAppOutput
from core.structures.gaussian_model import inverse_sigmoid


class GSMLPDecoder(nn.Module):
    """
    GSMLPDecoder is a Multi-Layer Perceptron-based decoder for Gaussian Splatting.

    This class generates Gaussian parameters (such as spatial offsets, scaling, opacity, rotation, and SH coefficients)
    from latent feature representations. It can optionally predict RGB color, supports spherical harmonics (SH) degrees,
    and allows for additional fine features. The decoder includes parameter tuners for scaling and supports methods
    for applying non-linearities and normalization during output construction.

    Attributes:
        use_rgb (bool): Indicates if RGB output is enabled.
        restrict_offset (bool): If True, restricts predicted XYZ offsets.
        xyz_offset_max_step (float or None): Maximum step for restrict_offset.
        fix_opacity (bool): If True, opacity is not learned.
        fix_rotation (bool): If True, rotation is not learned.
        use_fine_feat (bool): Enables use of additional fine features.
        attr_dict (dict): Output dimensionality dictionary for each predicted attribute.
    """

    def setup_functions(self):
        self.scaling_activation = trunc_exp  # proposed by torch-ngp
        self.scaling_inverse_activation = torch.log

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid
        self.rotation_activation = torch.nn.functional.normalize
        self.rgb_activation = torch.sigmoid

    def __init__(
        self,
        in_channels,
        use_rgb,
        clip_scaling=0.2,
        init_scaling=-6.0,
        init_density=0.1,
        sh_degree=None,
        xyz_offset=True,
        restrict_offset=True,
        xyz_offset_max_step=None,
        fix_opacity=False,
        fix_rotation=False,
        use_fine_feat=False,
    ):
        """
        Initializes the GSMLPDecoder.

        Args:
            in_channels (int): Number of input feature channels.
            use_rgb (bool): Whether to produce RGB color predictions using sigmoid activation.
            clip_scaling (float or list): Initial or time-dependent scaling clipping parameter. Can be a float, list, or OmegaConf ListConfig.
            init_scaling (float): Initial scaling value for the scaling output.
            init_density (float): Initial density value for the opacity output.
            sh_degree (int, optional): Degree for Spherical Harmonics (SH); defines SH output dimensionality.
            xyz_offset (bool): Whether to predict XYZ offsets.
            restrict_offset (bool): Whether to restrict XYZ offsets to a given range.
            xyz_offset_max_step (float, optional): Max step value for restricted XYZ offset prediction.
            fix_opacity (bool): Whether to keep opacity fixed (not learned).
            fix_rotation (bool): Whether to keep rotation fixed (not learned).
            use_fine_feat (bool): Whether to use extra fine features for enhanced output (e.g., fine SHS).
        """

        super().__init__()
        self.setup_functions()

        if isinstance(clip_scaling, omegaconf.listconfig.ListConfig) or isinstance(
            clip_scaling, list
        ):
            self.clip_scaling_pruner = LinerParameterTuner(*clip_scaling)
        else:
            self.clip_scaling_pruner = StaticParameterTuner(clip_scaling)
        self.clip_scaling = self.clip_scaling_pruner.get_value(0)

        self.use_rgb = use_rgb
        self.restrict_offset = restrict_offset
        self.xyz_offset = xyz_offset
        self.xyz_offset_max_step = xyz_offset_max_step  # 1.2 / 32
        self.fix_opacity = fix_opacity
        self.fix_rotation = fix_rotation
        self.use_fine_feat = use_fine_feat

        self.attr_dict = {
            "shs": (sh_degree + 1) ** 2 * 3,
            "scaling": 3,
            "xyz": 3,
            "opacity": None,
            "rotation": None,
        }
        if not self.fix_opacity:
            self.attr_dict["opacity"] = 1
        if not self.fix_rotation:
            self.attr_dict["rotation"] = 4

        self.out_layers = nn.ModuleDict()
        for key, out_ch in self.attr_dict.items():
            if out_ch is None:
                layer = nn.Identity()
            else:
                if key == "shs":
                    out_ch = 3 if use_rgb else out_ch
                    self.shs_out_ch = out_ch
                layer = nn.Linear(in_channels, out_ch)
                self._initialize_layer(layer, key, init_scaling, init_density)
            self.out_layers[key] = layer

        if self.use_fine_feat:
            fine_shs_layer = nn.Linear(in_channels, self.shs_out_ch)
            nn.init.constant_(fine_shs_layer.weight, 0)
            nn.init.constant_(fine_shs_layer.bias, 0)
            self.out_layers["fine_shs"] = fine_shs_layer

    def _initialize_layer(self, layer, key, init_scaling, init_density):
        if key == "shs" and self.use_rgb:
            return

        nn.init.constant_(layer.weight, 0)
        nn.init.constant_(layer.bias, 0)

        if key == "scaling":
            nn.init.constant_(layer.bias, init_scaling)
        elif key == "rotation":
            nn.init.constant_(layer.bias[0], 1.0)
        elif key == "opacity":
            nn.init.constant_(layer.bias, inverse_sigmoid(init_density))

    def hyper_step(self, step):
        self.clip_scaling = self.clip_scaling_pruner.get_value(step)

    def inference_forward(self, ret, constrain_dict, threshold=0.02):
        """
        Post-processes the output dictionary `ret` during inference.

        Applies constraints on the 'scaling' parameter for points identified as upper body
        by the constrain_dict['is_upper_body'] mask. Any scaling value above `threshold`
        for these points is clamped down to the threshold. This is used to restrict the
        effect of upper body scaling during inference, typically for avatar/character rendering.

        Args:
            ret (dict): Output dictionary from the network; must contain key 'scaling'.
            constrain_dict (dict): Constraint mask dictionary, must contain 'is_upper_body'.
            threshold (float, optional): Maximum allowed scaling value for points where the mask is true.
        Returns:
            dict: Post-processed output dictionary with 'scaling' values constrained for upper body.
        """

        mask = constrain_dict["is_upper_body"]
        scaling = ret["scaling"]
        scaling[mask] = scaling[mask].clamp(max=threshold)
        ret["scaling"] = scaling

        return ret

    def training_forward(self, ret, constrain_dict, threshold=0.02):
        """
        Post-processes the output dictionary `ret` during training.

        Applies constraints on the 'scaling' parameter for points identified as upper body
        by the constrain_dict['is_upper_body'] mask. Any scaling value above `threshold`
        for these points is clamped down to the threshold. This can help regulate the
        effect of upper body scaling during training, for example in avatar/character rendering.

        Args:
            ret (dict): Output dictionary from the network; must contain key 'scaling'.
            constrain_dict (dict): Constraint mask dictionary, must contain 'is_upper_body'.
            threshold (float, optional): Maximum allowed scaling value for points where the mask is true.
        Returns:
            dict: Post-processed output dictionary with 'scaling' values constrained for upper body.
        """

        mask = constrain_dict["is_upper_body"]
        scaling = ret["scaling"]
        scaling[mask] = scaling[mask].clamp(max=threshold)
        ret["scaling"] = scaling

        return ret

    def constrain_forward(self, ret, constrain_dict):
        # constrain_setting
        if not self.training:
            return self.inference_forward(ret, constrain_dict)
        else:
            return self.training_forward(ret, constrain_dict)

    def forward(self, x, pts, x_fine=None, constrain_dict=None):
        """
        Forward pass for the GSMLPDecoder.

        Processes input feature tensor `x` to generate Gaussian Splatting parameters such as spatial offsets,
        scaling, opacity, rotation, and SH coefficients through the corresponding output layers. Optionally,
        applies fine feature enhancement and constrains scaling values based on the supplied mask dictionary.

        Args:
            x (torch.Tensor): Input feature tensor of shape (N, C), where N is the number of points and C the feature dimension.
            pts (torch.Tensor): 3D points tensor (shape used for positional context; not directly used in base implementation).
            x_fine (torch.Tensor, optional): Additional (fine-grained) features for SH/color output; used if fine feature support is enabled.
            constrain_dict (dict, optional): Dictionary of masks for applying constraints (e.g., 'is_upper_body', 'is_hands') to the output.

        Returns:
            GaussianAppOutput: Output structure with predicted Gaussian parameters (offsets, scaling, rotation, opacity, SH, etc.).
        """

        assert len(x.shape) == 2
        ret = {}

        for k, layer in self.out_layers.items():
            if k == "fine_shs":
                continue

            v = layer(x)

            if k == "rotation":
                v = self._process_rotation(v)
            elif k == "scaling":
                v = self._process_scaling(v)
            elif k == "opacity":
                v = self._process_opacity(v, constrain_dict)
            elif k == "shs":
                v = self._process_shs(v, x_fine)
            elif k == "xyz":
                v, k = self._process_xyz(v)

            ret[k] = v

        ret["use_rgb"] = self.use_rgb

        if constrain_dict is not None:
            ret = self.constrain_forward(ret, constrain_dict)

        return GaussianAppOutput(**ret)

    ############################## Implicit Function ###########################################
    def _process_rotation(self, v):
        if self.fix_rotation:
            return matrix_to_quaternion(
                torch.eye(3, device=v.device).expand(v.size(0), 3, 3)
            )
        return self.rotation_activation(v)

    def _process_scaling(self, v):
        v = self.scaling_activation(v)
        return v.clamp(min=0, max=self.clip_scaling) if self.clip_scaling else v

    def _process_opacity(self, v, constrain_dict):
        if self.fix_opacity:
            v = 0.95 * torch.ones(v.size(0), 1, device=v.device)
            v = self.inverse_opacity_activation(v)
        else:
            if "is_hands" in constrain_dict:
                is_hands = constrain_dict["is_hands"]
                fix_values = self.inverse_opacity_activation(0.95)
                v[is_hands] = fix_values

        return self.opacity_activation(v)

    def _process_shs(self, v, x_fine):
        if self.use_rgb:
            v = self.rgb_activation(v)
        if self.use_fine_feat and x_fine is not None:
            v_fine = torch.tanh(self.out_layers["fine_shs"](x_fine))
            v = v + v_fine
        return v.reshape(v.size(0), -1, 3)

    def _process_xyz(self, v):
        if self.restrict_offset and self.xyz_offset_max_step:
            v = (torch.sigmoid(v) - 0.5) * self.xyz_offset_max_step
        else:
            raise NotImplementedError

        return v, "offset_xyz"

    ############################## Implicit Function ###########################################


class GSDeformableMLPDecoder(GSMLPDecoder):
    """
    GSDeformableMLPDecoder extends GSMLPDecoder by incorporating a deformable head for enhanced
    prediction of Gaussian parameters. It supports both attention-based and MLP-based deformation heads
    for flexible spatial adjustment of point offsets and related features.

    Attributes:
        use_deform (bool): Indicates if the deformable head is used.
        deform_head (nn.Module): The deformable head module (either DeformXYZHead or DeformXYZMLP).
    """

    def __init__(
        self,
        in_channels,
        use_rgb,
        clip_scaling=0.2,
        init_scaling=-5.0,
        init_density=0.1,
        sh_degree=None,
        xyz_offset=True,
        restrict_offset=True,
        xyz_offset_max_step=None,
        fix_opacity=False,
        fix_rotation=False,
        use_fine_feat=False,
        use_deform=True,  # head deform
        deform_head_type="mlp",
    ):
        """
        GSDeformableMLPDecoder extends GSMLPDecoder by incorporating a deformable head for enhanced
        prediction of Gaussian parameters. This class allows the decoder to learn spatial adjustments
        of Gaussian offsets and related features using either attention-based or MLP-based deformation heads.

        Args:
            in_channels (int): Number of input feature channels.
            use_rgb (bool): Whether to produce RGB color predictions using sigmoid activation.
            clip_scaling (float or list): Scaling clipping parameter.
            init_scaling (float): Initial scaling value for the scaling output.
            init_density (float): Initial density value for the opacity output.
            sh_degree (int, optional): Degree for Spherical Harmonics (SH).
            xyz_offset (bool): Whether to predict XYZ offsets.
            restrict_offset (bool): Whether to restrict XYZ offsets to a given range.
            xyz_offset_max_step (float, optional): Max step value for restricted XYZ offset prediction.
            fix_opacity (bool): Whether to keep opacity fixed (not learned).
            fix_rotation (bool): Whether to keep rotation fixed (not learned).
            use_fine_feat (bool): Whether to use extra fine features for enhanced output.
            use_deform (bool): Whether to enable the deformable head for predicting spatial adjustments.
            deform_head_type (str): Type of deformation head to use ('mlp' or 'attn').

        Attributes:
            use_deform (bool): Indicates if the deformable head is used.
            deform_head (nn.Module): The deformable head module (either DeformXYZHead or DeformXYZMLP).
        """

        super(GSDeformableMLPDecoder, self).__init__(
            in_channels,
            use_rgb,
            clip_scaling,
            init_scaling,
            init_density,
            sh_degree,
            xyz_offset,
            restrict_offset,
            xyz_offset_max_step,
            fix_opacity,
            fix_rotation,
            use_fine_feat,
        )

        self.use_deform = use_deform

        # ADDING IMPLEMENTATION in future.
        from core.models.heads.deform_heads import DeformXYZHead, DeformXYZMLP

        if self.use_deform:
            head_class = DeformXYZHead if deform_head_type == "attn" else DeformXYZMLP
            self.deform_head = head_class(
                dim_in=in_channels,
                img_dim_in=1024,
                sh_degree=sh_degree,
                xyz_offset=xyz_offset,
                restrict_offset=restrict_offset,
                xyz_offset_max_step=xyz_offset,
                fix_opacity=fix_opacity,
                fix_rotation=fix_rotation,
                use_rgb=True,
            )

    def forward(self, x, pts, x_fine=None, constrain_dict=None, image_feats=None):
        """
        Forward pass of the GSDeformableMLPDecoder.

        Args:
            x (torch.Tensor): Input feature tensor of shape (N, in_channels).
            pts (torch.Tensor): Input point coordinates.
            x_fine (torch.Tensor, optional): Additional fine features if used.
            constrain_dict (dict, optional): Dictionary containing constraints such as 'is_face'.
            image_feats (torch.Tensor, optional): Image features used by the deformable head for deformation prediction.

        Returns:
            dict: Dictionary containing predicted Gaussian parameters and optionally deformed offsets.
        """

        assert len(x.shape) == 2
        ret = super().forward(x, pts, x_fine, constrain_dict)

        if self.use_deform and constrain_dict is not None:
            is_face = constrain_dict["is_face"]
            offset = self.deform_head(x[is_face].unsqueeze(0), image_feats.unsqueeze(0))
            ret["offset_xyz"][is_face] += offset["xyz"]
        return ret
