# Adapted from JonasSchult/Mask3D.
from enum import Enum
from collections.abc import Sequence
import torch
import torch.nn as nn
from torch_scatter import scatter
import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiOps as me
from MinkowskiEngine import MinkowskiReLU

from mmengine.model import BaseModule
from mmdet3d.registry import MODELS


class NormType(Enum):
    BATCH_NORM = 0
    INSTANCE_NORM = 1
    INSTANCE_BATCH_NORM = 2


def get_norm(norm_type, n_channels, D, bn_momentum=0.1):
    if norm_type == NormType.BATCH_NORM:
        return ME.MinkowskiBatchNorm(n_channels, momentum=bn_momentum)
    elif norm_type == NormType.INSTANCE_NORM:
        return ME.MinkowskiInstanceNorm(n_channels)
    elif norm_type == NormType.INSTANCE_BATCH_NORM:
        return nn.Sequential(
            ME.MinkowskiInstanceNorm(n_channels),
            ME.MinkowskiBatchNorm(n_channels, momentum=bn_momentum))
    else:
        raise ValueError(f"Norm type: {norm_type} not supported")


class ConvType(Enum):
    """
    Define the kernel region type
    """

    HYPERCUBE = 0, "HYPERCUBE"
    SPATIAL_HYPERCUBE = 1, "SPATIAL_HYPERCUBE"
    SPATIO_TEMPORAL_HYPERCUBE = 2, "SPATIO_TEMPORAL_HYPERCUBE"
    HYPERCROSS = 3, "HYPERCROSS"
    SPATIAL_HYPERCROSS = 4, "SPATIAL_HYPERCROSS"
    SPATIO_TEMPORAL_HYPERCROSS = 5, "SPATIO_TEMPORAL_HYPERCROSS"
    SPATIAL_HYPERCUBE_TEMPORAL_HYPERCROSS = (
        6,
        "SPATIAL_HYPERCUBE_TEMPORAL_HYPERCROSS")

    def __new__(cls, value, name):
        member = object.__new__(cls)
        member._value_ = value
        member.fullname = name
        return member

    def __int__(self):
        return self.value


# Convert the ConvType var to a RegionType var
conv_to_region_type = {
    # kernel_size = [k, k, k, 1]
    ConvType.HYPERCUBE: ME.RegionType.HYPER_CUBE,
    ConvType.SPATIAL_HYPERCUBE: ME.RegionType.HYPER_CUBE,
    ConvType.SPATIO_TEMPORAL_HYPERCUBE: ME.RegionType.HYPER_CUBE,
    ConvType.HYPERCROSS: ME.RegionType.HYPER_CROSS,
    ConvType.SPATIAL_HYPERCROSS: ME.RegionType.HYPER_CROSS,
    ConvType.SPATIO_TEMPORAL_HYPERCROSS: ME.RegionType.HYPER_CROSS,
    ConvType.SPATIAL_HYPERCUBE_TEMPORAL_HYPERCROSS: ME.RegionType.HYPER_CUBE
}

# int_to_region_type = {m.value: m for m in ME.RegionType}
int_to_region_type = {m: ME.RegionType(m) for m in range(3)}


def convert_region_type(region_type):
    """Convert the integer region_type to the corresponding
    RegionType enum object.
    """
    return int_to_region_type[region_type]


def convert_conv_type(conv_type, kernel_size, D):
    assert isinstance(conv_type, ConvType), "conv_type must be of ConvType"
    region_type = conv_to_region_type[conv_type]
    axis_types = None
    if conv_type == ConvType.SPATIAL_HYPERCUBE:
        # No temporal convolution
        if isinstance(kernel_size, Sequence):
            kernel_size = kernel_size[:3]
        else:
            kernel_size = [
                kernel_size,
            ] * 3
        if D == 4:
            kernel_size.append(1)
    elif conv_type == ConvType.SPATIO_TEMPORAL_HYPERCUBE:
        # conv_type conversion already handled
        assert D == 4
    elif conv_type == ConvType.HYPERCUBE:
        # conv_type conversion already handled
        pass
    elif conv_type == ConvType.SPATIAL_HYPERCROSS:
        if isinstance(kernel_size, Sequence):
            kernel_size = kernel_size[:3]
        else:
            kernel_size = [
                kernel_size,
            ] * 3
        if D == 4:
            kernel_size.append(1)
    elif conv_type == ConvType.HYPERCROSS:
        # conv_type conversion already handled
        pass
    elif conv_type == ConvType.SPATIO_TEMPORAL_HYPERCROSS:
        # conv_type conversion already handled
        assert D == 4
    elif conv_type == ConvType.SPATIAL_HYPERCUBE_TEMPORAL_HYPERCROSS:
        # Define the CUBIC conv kernel for spatial dims
        # and CROSS conv for temp dim
        axis_types = [
            ME.RegionType.HYPER_CUBE,
        ] * 3
        if D == 4:
            axis_types.append(ME.RegionType.HYPER_CROSS)
    return region_type, axis_types, kernel_size


def conv(in_planes,
         out_planes,
         kernel_size,
         stride=1,
         dilation=1,
         bias=False,
         conv_type=ConvType.HYPERCUBE,
         D=-1):
    assert D > 0, "Dimension must be a positive integer"
    region_type, axis_types, kernel_size = convert_conv_type(
        conv_type, kernel_size, D)
    kernel_generator = ME.KernelGenerator(
        kernel_size,
        stride,
        dilation,
        region_type=region_type,
        axis_types=None,  # axis_types JONAS
        dimension=D)

    return ME.MinkowskiConvolution(
        in_channels=in_planes,
        out_channels=out_planes,
        kernel_size=kernel_size,
        stride=stride,
        dilation=dilation,
        bias=bias,
        kernel_generator=kernel_generator,
        dimension=D)


def conv_tr(in_planes,
            out_planes,
            kernel_size,
            upsample_stride=1,
            dilation=1,
            bias=False,
            conv_type=ConvType.HYPERCUBE,
            D=-1):
    assert D > 0, "Dimension must be a positive integer"
    region_type, axis_types, kernel_size = convert_conv_type(
        conv_type, kernel_size, D)
    kernel_generator = ME.KernelGenerator(
        kernel_size,
        upsample_stride,
        dilation,
        region_type=region_type,
        axis_types=axis_types,
        dimension=D)

    return ME.MinkowskiConvolutionTranspose(
        in_channels=in_planes,
        out_channels=out_planes,
        kernel_size=kernel_size,
        stride=upsample_stride,
        dilation=dilation,
        bias=bias,
        kernel_generator=kernel_generator,
        dimension=D)


class BasicBlockBase(nn.Module):
    expansion = 1
    NORM_TYPE = NormType.BATCH_NORM

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 conv_type=ConvType.HYPERCUBE,
                 bn_momentum=0.1,
                 D=3):
        super().__init__()

        self.conv1 = conv(
            inplanes,
            planes,
            kernel_size=3,
            stride=stride,
            dilation=dilation,
            conv_type=conv_type,
            D=D)
        self.norm1 = get_norm(
            self.NORM_TYPE, planes, D, bn_momentum=bn_momentum)
        self.conv2 = conv(
            planes,
            planes,
            kernel_size=3,
            stride=1,
            dilation=dilation,
            bias=False,
            conv_type=conv_type,
            D=D)
        self.norm2 = get_norm(
            self.NORM_TYPE, planes, D, bn_momentum=bn_momentum)
        self.relu = MinkowskiReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BasicBlock(BasicBlockBase):
    NORM_TYPE = NormType.BATCH_NORM


class Res16UNetBase(BaseModule):
    """Base class for Minkowski U-Net.

    Args:
        in_channels (int): Number of input channels.
        out_channles (int): Number of output channels.
        config (dict): Extra parameters including
            `dilations`, `conv1_kernel_size`, `bn_momentum`.
        D (int): Conv dimension.
    """
    BLOCK = None
    PLANES = (32, 64, 128, 256, 256, 256, 256, 256)
    DILATIONS = (1, 1, 1, 1, 1, 1, 1, 1)
    LAYERS = (2, 2, 2, 2, 2, 2, 2, 2)
    INIT_DIM = 32
    OUT_PIXEL_DIST = 1
    NORM_TYPE = NormType.BATCH_NORM
    NON_BLOCK_CONV_TYPE = ConvType.SPATIAL_HYPERCUBE
    CONV_TYPE = ConvType.SPATIAL_HYPERCUBE_TEMPORAL_HYPERCROSS

    def __init__(self,
                 in_channels,
                 out_channels,
                 config,
                 D=3,
                 dino_dim=None,  # 可选：DINO 通道数，提供则启用 2D 注入
                 dual_output=False,
                 **kwargs):
        self.D = D
        self.dino_dim = dino_dim
        self._dual_output = dual_output
        super().__init__()
        self.network_initialization(in_channels, out_channels, config, D)
        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)

    def _make_layer(self,
                    block,
                    planes,
                    blocks,
                    stride=1,
                    dilation=1,
                    norm_type=NormType.BATCH_NORM,
                    bn_momentum=0.1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                    D=self.D),
                get_norm(
                    norm_type,
                    planes * block.expansion,
                    D=self.D,
                    bn_momentum=bn_momentum))
        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride=stride,
                dilation=dilation,
                downsample=downsample,
                conv_type=self.CONV_TYPE,
                D=self.D))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    stride=1,
                    dilation=dilation,
                    conv_type=self.CONV_TYPE,
                    D=self.D))

        return nn.Sequential(*layers)

    def network_initialization(self, in_channels, out_channels, config, D):
        # Setup net_metadata
        dilations = self.DILATIONS
        bn_momentum = config.bn_momentum

        def space_n_time_m(n, m):
            return n if D == 3 else [n, n, n, m]

        if D == 4:
            self.OUT_PIXEL_DIST = space_n_time_m(self.OUT_PIXEL_DIST, 1)

        # Output of the first conv concated to conv6
        self.inplanes = self.INIT_DIM
        self.conv0p1s1 = conv(
            in_channels,
            self.inplanes,
            kernel_size=space_n_time_m(config.conv1_kernel_size, 1),
            stride=1,
            dilation=1,
            conv_type=self.NON_BLOCK_CONV_TYPE,
            D=D)

        self.bn0 = get_norm(
            self.NORM_TYPE, self.inplanes, D, bn_momentum=bn_momentum)

        self.conv1p1s2 = conv(
            self.inplanes,
            self.inplanes,
            kernel_size=space_n_time_m(2, 1),
            stride=space_n_time_m(2, 1),
            dilation=1,
            conv_type=self.NON_BLOCK_CONV_TYPE,
            D=D)
        self.bn1 = get_norm(
            self.NORM_TYPE, self.inplanes, D, bn_momentum=bn_momentum)
        self.block1 = self._make_layer(
            self.BLOCK,
            self.PLANES[0],
            self.LAYERS[0],
            dilation=dilations[0],
            norm_type=self.NORM_TYPE,
            bn_momentum=bn_momentum)

        self.conv2p2s2 = conv(
            self.inplanes,
            self.inplanes,
            kernel_size=space_n_time_m(2, 1),
            stride=space_n_time_m(2, 1),
            dilation=1,
            conv_type=self.NON_BLOCK_CONV_TYPE,
            D=D)
        self.bn2 = get_norm(
            self.NORM_TYPE, self.inplanes, D, bn_momentum=bn_momentum)
        self.block2 = self._make_layer(
            self.BLOCK,
            self.PLANES[1],
            self.LAYERS[1],
            dilation=dilations[1],
            norm_type=self.NORM_TYPE,
            bn_momentum=bn_momentum)

        self.conv3p4s2 = conv(
            self.inplanes,
            self.inplanes,
            kernel_size=space_n_time_m(2, 1),
            stride=space_n_time_m(2, 1),
            dilation=1,
            conv_type=self.NON_BLOCK_CONV_TYPE,
            D=D)
        self.bn3 = get_norm(
            self.NORM_TYPE, self.inplanes, D, bn_momentum=bn_momentum)
        self.block3 = self._make_layer(
            self.BLOCK,
            self.PLANES[2],
            self.LAYERS[2],
            dilation=dilations[2],
            norm_type=self.NORM_TYPE,
            bn_momentum=bn_momentum)

        self.conv4p8s2 = conv(
            self.inplanes,
            self.inplanes,
            kernel_size=space_n_time_m(2, 1),
            stride=space_n_time_m(2, 1),
            dilation=1,
            conv_type=self.NON_BLOCK_CONV_TYPE,
            D=D)
        self.bn4 = get_norm(
            self.NORM_TYPE, self.inplanes, D, bn_momentum=bn_momentum)
        self.block4 = self._make_layer(
            self.BLOCK,
            self.PLANES[3],
            self.LAYERS[3],
            dilation=dilations[3],
            norm_type=self.NORM_TYPE,
            bn_momentum=bn_momentum)
        self.convtr4p16s2 = conv_tr(
            self.inplanes,
            self.PLANES[4],
            kernel_size=space_n_time_m(2, 1),
            upsample_stride=space_n_time_m(2, 1),
            dilation=1,
            bias=False,
            conv_type=self.NON_BLOCK_CONV_TYPE,
            D=D)
        self.bntr4 = get_norm(
            self.NORM_TYPE, self.PLANES[4], D, bn_momentum=bn_momentum)

        self.inplanes = self.PLANES[4] + self.PLANES[2] * self.BLOCK.expansion
        self.block5 = self._make_layer(
            self.BLOCK,
            self.PLANES[4],
            self.LAYERS[4],
            dilation=dilations[4],
            norm_type=self.NORM_TYPE,
            bn_momentum=bn_momentum)
        self.convtr5p8s2 = conv_tr(
            self.inplanes,
            self.PLANES[5],
            kernel_size=space_n_time_m(2, 1),
            upsample_stride=space_n_time_m(2, 1),
            dilation=1,
            bias=False,
            conv_type=self.NON_BLOCK_CONV_TYPE,
            D=D)
        self.bntr5 = get_norm(
            self.NORM_TYPE, self.PLANES[5], D, bn_momentum=bn_momentum)

        self.inplanes = self.PLANES[5] + self.PLANES[1] * self.BLOCK.expansion
        self.block6 = self._make_layer(
            self.BLOCK,
            self.PLANES[5],
            self.LAYERS[5],
            dilation=dilations[5],
            norm_type=self.NORM_TYPE,
            bn_momentum=bn_momentum)
        self.convtr6p4s2 = conv_tr(
            self.inplanes,
            self.PLANES[6],
            kernel_size=space_n_time_m(2, 1),
            upsample_stride=space_n_time_m(2, 1),
            dilation=1,
            bias=False,
            conv_type=self.NON_BLOCK_CONV_TYPE,
            D=D)
        self.bntr6 = get_norm(
            self.NORM_TYPE, self.PLANES[6], D, bn_momentum=bn_momentum)

        self.inplanes = self.PLANES[6] + self.PLANES[0] * self.BLOCK.expansion
        self.block7 = self._make_layer(
            self.BLOCK,
            self.PLANES[6],
            self.LAYERS[6],
            dilation=dilations[6],
            norm_type=self.NORM_TYPE,
            bn_momentum=bn_momentum)
        self.convtr7p2s2 = conv_tr(
            self.inplanes,
            self.PLANES[7],
            kernel_size=space_n_time_m(2, 1),
            upsample_stride=space_n_time_m(2, 1),
            dilation=1,
            bias=False,
            conv_type=self.NON_BLOCK_CONV_TYPE,
            D=D)
        self.bntr7 = get_norm(
            self.NORM_TYPE, self.PLANES[7], D, bn_momentum=bn_momentum)

        self.inplanes = self.PLANES[7] + self.INIT_DIM
        self.block8 = self._make_layer(
            self.BLOCK,
            self.PLANES[7],
            self.LAYERS[7],
            dilation=dilations[7],
            norm_type=self.NORM_TYPE,
            bn_momentum=bn_momentum)

        # self.final = conv(
        #     self.PLANES[7],
        #     out_channels,
        #     kernel_size=1,
        #     stride=1,
        #     bias=True,
        #     D=D)
        self.relu = MinkowskiReLU(inplace=True)

        # 若提供 dino_dim，则为各解码层准备 1x1 投影（用于后续 2D 注入）
        self.use_dino = self.dino_dim is not None
        self._dino_point_fusion_enable = bool(
            getattr(config, 'dino_point_fusion_enable', True))
        if self.use_dino:
            # 最新一次 forward 中各尺度的命中率（用于训练期 sanity hook）
            self._last_dino_hit = {}
            # 最新一次 forward 中各尺度的融合尺度统计（用于训练期 sanity hook）
            self._last_dino_fuse = {}
            # 低命中保护：当 DINO sparse 坐标与 backbone 坐标不一致时，
            # features_at_coordinates 会大量 miss 并返回全 0，此时继续做融合会把 backbone 特征扰乱。
            # 这里提供一个轻量保护：命中率低于阈值则退化为原始 (up ⊕ skip)。
            self._dino_miss_warn_count = 0
            self._dino_min_hit_ratio = float(getattr(config, 'dino_min_hit_ratio', 0.05))
            self._dino_strict = bool(getattr(config, 'dino_strict', False))
            # 推荐：使用 residual 形式注入（base_cat + fuse(cat+ dino)），
            # 并将 fuse 的 1×1 conv 初始化为 0，确保加载旧 checkpoint（不含 DINO）时
            # 主干行为保持不变，避免 AP 断崖式下降。
            self._dino_residual = bool(getattr(config, 'dino_residual', True))

            # DITR 风格：三路特征（up、skip、dino）concat 后做 1×1 投影 + BN + GELU，
            # 输出通道数保持与原始 (up ⊕ skip) 相同，以避免重写后续 decoder blocks。
            def _fuse(in_ch: int, out_ch: int):
                return nn.Sequential(
                    ME.MinkowskiConvolution(
                        in_channels=in_ch,
                        out_channels=out_ch,
                        kernel_size=1,
                        stride=1,
                        bias=False,
                        dimension=D,
                    ),
                    ME.MinkowskiBatchNorm(out_ch, momentum=bn_momentum),
                    ME.MinkowskiGELU(),
                )

            # level stride=8/4/2/1
            self.dino_fuse_8 = _fuse(
                # up: PLANES[4], skip(out_b3p8): PLANES[2]
                self.PLANES[4] + self.PLANES[2] + self.dino_dim,
                self.PLANES[4] + self.PLANES[2],
            )
            self.dino_fuse_4 = _fuse(
                # up: PLANES[5], skip(out_b2p4): PLANES[1]
                self.PLANES[5] + self.PLANES[1] + self.dino_dim,
                self.PLANES[5] + self.PLANES[1],
            )
            self.dino_fuse_2 = _fuse(
                # up: PLANES[6], skip(out_b1p2): PLANES[0]
                self.PLANES[6] + self.PLANES[0] + self.dino_dim,
                self.PLANES[6] + self.PLANES[0],
            )
            self.dino_fuse_1 = _fuse(
                self.PLANES[7] + self.INIT_DIM + self.dino_dim,
                self.PLANES[7] + self.INIT_DIM,
            )

            # small-normal init (std=1e-4) to avoid gradient deadlock:
            # zero-init causes MinkowskiEngine sparse conv to produce 0 output,
            # which blocks backward gradient (same deadlock as W_c=0).
            if self._dino_residual:
                for m in (self.dino_fuse_8, self.dino_fuse_4, self.dino_fuse_2, self.dino_fuse_1):
                    try:
                        conv1x1 = m[0]
                        if hasattr(conv1x1, 'kernel') and conv1x1.kernel is not None:
                            conv1x1.kernel.data.normal_(mean=0.0, std=1e-4)
                    except Exception:
                        pass

        self._object_feedback_enable = bool(
            getattr(config, 'object_feedback_enable', False) or
            getattr(config, 'pre_memory_object_feedback', False) or
            getattr(config, 's2_object_feedback_enable', False))
        self._object_feedback_mode = str(getattr(config, 'object_feedback_mode', 'real'))
        self._object_feedback_placement = str(
            getattr(config, 'object_feedback_placement', 'pre_memory'))
        if self._object_feedback_placement not in ('pre_memory', 'post_memory'):
            raise RuntimeError(
                'object_feedback_placement must be pre_memory or post_memory, '
                f'got {self._object_feedback_placement}')
        self._object_feedback_include_dino = bool(
            getattr(config, 'object_feedback_include_dino', True) and
            self.dino_dim is not None)
        self._object_feedback_include_s2 = bool(
            getattr(config, 'object_feedback_include_s2', True))
        self._object_feedback_include_xyz = bool(getattr(config, 'object_feedback_include_xyz', True))
        self._object_feedback_oracle_purity = bool(
            getattr(config, 'object_feedback_oracle_purity', False))
        self._object_feedback_oracle_purity_threshold = float(
            getattr(config, 'object_feedback_oracle_purity_threshold', 0.8))
        if not 0.0 <= self._object_feedback_oracle_purity_threshold <= 1.0:
            raise RuntimeError(
                'object_feedback_oracle_purity_threshold must be in [0, 1], '
                f'got {self._object_feedback_oracle_purity_threshold}')
        self._object_feedback_persistent_oracle = bool(
            getattr(config, 'object_feedback_persistent_oracle', False))
        self._object_feedback_memory_momentum = float(
            getattr(config, 'object_feedback_memory_momentum', 0.8))
        if not 0.0 <= self._object_feedback_memory_momentum < 1.0:
            raise RuntimeError(
                'object_feedback_memory_momentum must be in [0, 1), '
                f'got {self._object_feedback_memory_momentum}')
        self._object_feedback_output_scale = float(
            getattr(config, 'object_feedback_output_scale', 1.0))
        if self._object_feedback_output_scale < 0.0:
            raise RuntimeError(
                'object_feedback_output_scale must be non-negative, '
                f'got {self._object_feedback_output_scale}')
        if self._object_feedback_persistent_oracle:
            if not self._object_feedback_oracle_purity:
                raise RuntimeError(
                    'persistent oracle feedback requires oracle purity labels')
            if (not self._object_feedback_include_dino or
                    self._object_feedback_include_s2 or
                    self._object_feedback_include_xyz):
                raise RuntimeError(
                    'persistent oracle feedback currently requires a DINO-only prototype')
        self._object_feedback_memory = []
        self._object_feedback_min_hit = float(getattr(config, 'object_feedback_min_hit', 0.01))
        self._object_feedback_eps = float(getattr(config, 'object_feedback_eps', 1e-6))
        self._last_object_feedback_stats = dict(
            enabled=False, placement=self._object_feedback_placement,
            fragments=0, gate_mean=0.0,
            delta_input_norm_ratio=0.0, dino_valid_rate=0.0,
            purity_mean=0.0, write_keep_rate=1.0,
            memory_hit_rate=0.0, memory_objects=0)
        self._last_object_feedback_repr = None
        if self._object_feedback_enable:
            s2_ch = self.PLANES[0]
            proto_ch = s2_ch if self._object_feedback_include_s2 else 0
            proto_ch += self.dino_dim if self._object_feedback_include_dino else 0
            if self._object_feedback_include_xyz:
                proto_ch += 3
            if proto_ch == 0:
                raise RuntimeError('object feedback requires at least one prototype source')
            hidden_ch = int(getattr(config, 'object_feedback_hidden_dim', max(64, s2_ch)))
            self.object_feedback_proto_norm = nn.LayerNorm(proto_ch)
            self.object_feedback_delta = nn.Sequential(
                nn.Linear(proto_ch, hidden_ch),
                nn.GELU(),
                nn.Linear(hidden_ch, s2_ch))
            self.object_feedback_point_gate = nn.Sequential(
                nn.Linear(s2_ch + proto_ch, hidden_ch),
                nn.GELU(),
                nn.Linear(hidden_ch, 1))
            self.object_feedback_voxel_gate = nn.Sequential(
                nn.Linear(s2_ch + proto_ch, hidden_ch),
                nn.GELU(),
                nn.Linear(hidden_ch, 1))
            self._init_object_feedback(float(getattr(config, 'object_feedback_init_gate', 1e-3)))

        # ── dual_output: semantic side branch projection ──
        if self._dual_output:
            # dino_fuse_1 outputs PLANES[7]+INIT_DIM channels (cat of up+skip),
            # project to out_channels to match the main path dimension.
            sem_in = self.PLANES[7] + self.INIT_DIM
            self.sem_out_proj = nn.Sequential(
                nn.Linear(sem_in, out_channels),
                nn.LayerNorm(out_channels),
                nn.ReLU(),
            )

    def _init_object_feedback(self, init_gate):
        init_gate = max(1e-8, min(float(init_gate), 1.0 - 1e-8))
        gate_each = init_gate ** 0.5
        gate_bias = torch.logit(torch.tensor(gate_each)).item()
        for module in (self.object_feedback_delta,
                       self.object_feedback_point_gate,
                       self.object_feedback_voxel_gate):
            final = module[-1]
            nn.init.normal_(final.weight, mean=0.0, std=1e-4)
            nn.init.constant_(final.bias, 0.0)
        nn.init.constant_(self.object_feedback_point_gate[-1].bias, gate_bias)
        nn.init.constant_(self.object_feedback_voxel_gate[-1].bias, gate_bias)

    def reset_object_feedback_memory(self):
        self._object_feedback_memory = []

    def _object_feedback_stats(self, **kwargs):
        stats = dict(
            enabled=bool(getattr(self, '_object_feedback_enable', False)),
            placement=getattr(self, '_object_feedback_placement', 'pre_memory'),
            fragments=0,
            gate_mean=0.0,
            delta_input_norm_ratio=0.0,
            dino_valid_rate=0.0,
            purity_mean=0.0,
            write_keep_rate=1.0,
            memory_hit_rate=0.0,
            memory_objects=0)
        stats.update(kwargs)
        self._last_object_feedback_stats = stats
        return stats

    def _flatten_sp_masks(self, sp_pts_mask, device):
        if sp_pts_mask is None:
            return None
        if torch.is_tensor(sp_pts_mask):
            return sp_pts_mask.to(device=device, dtype=torch.long).reshape(-1)
        if not isinstance(sp_pts_mask, (list, tuple)):
            return None
        flat, offset = [], 0
        for mask in sp_pts_mask:
            if mask is None:
                return None
            mask = mask.to(device=device, dtype=torch.long).reshape(-1)
            flat.append(mask + offset)
            offset += int(mask.max().item()) + 1 if mask.numel() else 0
        return torch.cat(flat, dim=0) if flat else None

    def _flatten_point_labels(self, labels, device):
        if labels is None:
            return None
        if torch.is_tensor(labels):
            return labels.to(device=device, dtype=torch.long).reshape(-1)
        if not isinstance(labels, (list, tuple)):
            return None
        flat = []
        for label in labels:
            if label is None:
                return None
            flat.append(label.to(device=device, dtype=torch.long).reshape(-1))
        return torch.cat(flat, dim=0) if flat else None

    def _point_to_out_voxel_index(self, out_b1p2, point_coords):
        coords = point_coords.to(device=out_b1p2.coordinates.device)
        if coords.dim() != 2 or coords.shape[1] != out_b1p2.coordinates.shape[1]:
            return None, None, 0.0
        coords = torch.floor(coords.float()).to(torch.int32)
        candidates = [coords]
        aligned_s2_coords = coords.clone()
        aligned_s2_coords[:, 1:] = (
            torch.div(aligned_s2_coords[:, 1:], 2, rounding_mode='floor') * 2)
        candidates.append(aligned_s2_coords)
        s2_coords = coords.clone()
        s2_coords[:, 1:] = torch.div(s2_coords[:, 1:], 2, rounding_mode='floor')
        candidates.append(s2_coords)
        out_coords = out_b1p2.coordinates.to(device=coords.device, dtype=torch.int32)
        n_out = out_coords.shape[0]
        best_coords, best_idx, best_hit = None, None, -1.0
        for cand in candidates:
            all_coords = torch.cat([out_coords, cand], dim=0)
            _, inverse = torch.unique(all_coords, dim=0, return_inverse=True)
            out_inverse = inverse[:n_out]
            point_inverse = inverse[n_out:]
            unique_to_out = torch.full(
                (int(inverse.max().item()) + 1,),
                -1,
                device=coords.device,
                dtype=torch.long)
            unique_to_out[out_inverse] = torch.arange(n_out, device=coords.device, dtype=torch.long)
            voxel_idx = unique_to_out[point_inverse]
            hit = float((voxel_idx >= 0).float().mean().item()) if voxel_idx.numel() else 0.0
            if hit > best_hit:
                best_coords, best_idx, best_hit = cand, voxel_idx, hit
        return best_coords, best_idx, max(float(best_hit), 0.0)

    def _apply_object_feedback_s2(self, out_b1p2, dino_s2=None,
                                  sp_pts_mask=None, point_coords=None,
                                  point_xyz=None, point_instance_ids=None):
        self._last_object_feedback_repr = None
        if not getattr(self, '_object_feedback_enable', False):
            self._object_feedback_stats(enabled=False)
            return out_b1p2
        mode = getattr(self, '_object_feedback_mode', 'real')
        if mode not in ('real', 'shuffled', 'stats-only'):
            raise RuntimeError(f'object_feedback_mode must be real/shuffled/stats-only, got {mode}')
        if sp_pts_mask is None or point_coords is None:
            self._object_feedback_stats(enabled=True, skipped='missing_inputs')
            return out_b1p2

        device = out_b1p2.features.device
        dtype = out_b1p2.features.dtype
        sp = self._flatten_sp_masks(sp_pts_mask, device)
        if sp is None or sp.numel() == 0 or sp.shape[0] != point_coords.shape[0]:
            self._object_feedback_stats(enabled=True, skipped='invalid_sp_mask')
            return out_b1p2
        point_s2_coords, voxel_idx, map_hit = self._point_to_out_voxel_index(out_b1p2, point_coords)
        if point_s2_coords is None or voxel_idx is None or map_hit < self._object_feedback_min_hit:
            self._object_feedback_stats(enabled=True, skipped='low_s2_hit', s2_hit=float(map_hit))
            return out_b1p2
        valid_point = voxel_idx >= 0
        if int(valid_point.sum().item()) == 0:
            self._object_feedback_stats(enabled=True, skipped='no_valid_voxels', s2_hit=float(map_hit))
            return out_b1p2

        point_feat = out_b1p2.features[voxel_idx[valid_point]]
        proto_parts = [point_feat] if self._object_feedback_include_s2 else []
        dino_valid_rate = 0.0
        if self._object_feedback_include_dino:
            if dino_s2 is None:
                dino_on_point = point_feat.new_zeros((point_feat.shape[0], self.dino_dim))
            else:
                try:
                    dino_on_point = dino_s2.features_at_coordinates(point_s2_coords[valid_point].float())
                    if dino_on_point.dim() != 2 or dino_on_point.shape[0] != point_feat.shape[0]:
                        dino_on_point = point_feat.new_zeros((point_feat.shape[0], self.dino_dim))
                    else:
                        dino_on_point = dino_on_point.to(device=device, dtype=dtype)
                except Exception:
                    dino_on_point = point_feat.new_zeros((point_feat.shape[0], self.dino_dim))
            dino_valid = dino_on_point.abs().sum(dim=1) > self._object_feedback_eps
            dino_valid_rate = float(dino_valid.float().mean().item()) if dino_valid.numel() else 0.0
            proto_parts.append(dino_on_point)
        if self._object_feedback_include_xyz:
            if point_xyz is None:
                xyz = point_feat.new_zeros((point_feat.shape[0], 3))
            else:
                xyz = point_xyz.to(device=device, dtype=dtype)[valid_point]
                if xyz.dim() != 2 or xyz.shape[1] != 3:
                    self._object_feedback_stats(enabled=True, skipped='invalid_xyz')
                    return out_b1p2
            proto_parts.append(xyz)

        valid_sp = sp[valid_point]
        n_frag = int(valid_sp.max().item()) + 1 if valid_sp.numel() else 0
        if n_frag <= 0:
            self._object_feedback_stats(enabled=True, skipped='no_fragments')
            return out_b1p2
        proto_in = torch.cat(proto_parts, dim=1)
        counts = scatter(torch.ones_like(valid_sp, dtype=dtype), valid_sp, dim=0, dim_size=n_frag, reduce='sum').clamp(min=1.0)
        current_proto = (
            scatter(proto_in, valid_sp, dim=0, dim_size=n_frag, reduce='sum') /
            counts[:, None])

        purity = None
        write_keep = None
        dominant_ids = None
        fragment_batch = None
        point_batch = point_coords[valid_point, 0].to(device=device, dtype=torch.long)
        if self._object_feedback_oracle_purity:
            instance_ids = self._flatten_point_labels(point_instance_ids, device)
            if instance_ids is None or instance_ids.shape[0] != sp.shape[0]:
                self._object_feedback_stats(
                    enabled=True, skipped='missing_oracle_labels',
                    fragments=int(n_frag), s2_hit=float(map_hit),
                    dino_valid_rate=float(dino_valid_rate))
                return out_b1p2
            instance_ids = instance_ids[valid_point]
            valid_instance = instance_ids >= 0
            dominant_counts = counts.new_zeros(n_frag)
            dominant_ids = torch.full(
                (n_frag,), -1, device=device, dtype=torch.long)
            if bool(valid_instance.any()):
                instance_width = int(instance_ids[valid_instance].max().item()) + 1
                pair_keys = (
                    valid_sp[valid_instance] * instance_width +
                    instance_ids[valid_instance])
                unique_pairs, pair_counts = torch.unique(
                    pair_keys, return_counts=True)
                pair_fragments = torch.div(
                    unique_pairs, instance_width, rounding_mode='floor')
                dominant_counts = scatter(
                    pair_counts.to(dtype), pair_fragments, dim=0,
                    dim_size=n_frag, reduce='max')
                pair_instances = unique_pairs.remainder(instance_width)
                dominant_candidates = torch.where(
                    pair_counts.to(dtype) == dominant_counts[pair_fragments],
                    pair_instances + 1,
                    torch.zeros_like(pair_instances))
                dominant_ids = scatter(
                    dominant_candidates, pair_fragments, dim=0,
                    dim_size=n_frag, reduce='max') - 1
            purity = dominant_counts / counts
            write_keep = (
                purity >= self._object_feedback_oracle_purity_threshold
            ).to(dtype)

        memory_hit = counts.new_zeros(n_frag)
        if self._object_feedback_persistent_oracle:
            fragment_batch = scatter(
                point_batch, valid_sp, dim=0, dim_size=n_frag, reduce='max')
            batch_size = int(point_batch.max().item()) + 1 if point_batch.numel() else 0
            while len(self._object_feedback_memory) < batch_size:
                self._object_feedback_memory.append({})

            historical_proto = current_proto.new_zeros(current_proto.shape)
            for fragment_idx in range(n_frag):
                if (write_keep is None or write_keep[fragment_idx] <= 0 or
                        dominant_ids[fragment_idx] < 0):
                    continue
                batch_idx = int(fragment_batch[fragment_idx].item())
                instance_idx = int(dominant_ids[fragment_idx].item())
                state = self._object_feedback_memory[batch_idx].get(instance_idx)
                if state is not None:
                    historical_proto[fragment_idx] = state['prototype'].to(
                        device=device, dtype=dtype)
                    memory_hit[fragment_idx] = 1.0

            if mode == 'shuffled':
                hit_indices = torch.nonzero(memory_hit > 0, as_tuple=True)[0]
                if hit_indices.numel() > 1:
                    permuted = hit_indices[
                        torch.randperm(hit_indices.numel(), device=device)]
                    historical_proto[hit_indices] = historical_proto[permuted]
            proto = self.object_feedback_proto_norm(historical_proto)
            write_keep = write_keep * memory_hit

            momentum = self._object_feedback_memory_momentum
            for batch_idx in range(batch_size):
                batch_mask = (point_batch == batch_idx) & valid_instance
                if not bool(batch_mask.any()):
                    continue
                for instance_idx in torch.unique(instance_ids[batch_mask]).tolist():
                    instance_mask = batch_mask & (instance_ids == instance_idx)
                    observation = proto_in[instance_mask].mean(dim=0).detach()
                    state = self._object_feedback_memory[batch_idx].get(instance_idx)
                    if state is None:
                        updated = observation
                        age = 1
                    else:
                        updated = (
                            momentum * state['prototype'].to(
                                device=device, dtype=dtype) +
                            (1.0 - momentum) * observation)
                        age = int(state['age']) + 1
                    self._object_feedback_memory[batch_idx][instance_idx] = {
                        'prototype': updated.detach(),
                        'age': age}
        else:
            proto = self.object_feedback_proto_norm(current_proto)
            if mode == 'shuffled' and n_frag > 1:
                proto = proto[torch.randperm(n_frag, device=device)]

        point_gate_in = torch.cat([point_feat, proto[valid_sp]], dim=1)
        point_gate = torch.sigmoid(self.object_feedback_point_gate(point_gate_in))
        point_delta = self.object_feedback_delta(proto[valid_sp])
        point_update = point_gate * point_delta
        if write_keep is not None:
            point_update = point_update * write_keep[valid_sp, None]

        valid_voxels, inverse = torch.unique(voxel_idx[valid_point], sorted=False, return_inverse=True)
        voxel_feat = out_b1p2.features[valid_voxels]
        voxel_proto = scatter(
            proto[valid_sp], inverse, dim=0,
            dim_size=valid_voxels.shape[0], reduce='mean')
        voxel_point_update = scatter(
            point_update, inverse, dim=0,
            dim_size=valid_voxels.shape[0], reduce='mean')
        voxel_gate = torch.sigmoid(self.object_feedback_voxel_gate(torch.cat([voxel_feat, voxel_proto], dim=1)))
        update = (
            self._object_feedback_output_scale * voxel_gate *
            voxel_point_update)

        if getattr(self, '_capture_object_feedback_repr', False):
            instance_ids = self._flatten_point_labels(point_instance_ids, device)
            valid_instance_ids = None
            if instance_ids is not None and instance_ids.shape[0] == sp.shape[0]:
                valid_instance_ids = instance_ids[valid_point]
            self._last_object_feedback_repr = dict(
                mode=mode,
                voxel_coords=out_b1p2.coordinates[valid_voxels].detach().cpu(),
                voxel_features_before=voxel_feat.detach().to(
                    dtype=torch.float16).cpu(),
                voxel_update=update.detach().to(dtype=torch.float16).cpu(),
                voxel_gate=voxel_gate.detach().to(dtype=torch.float16).cpu(),
                point_to_voxel=inverse.detach().to(dtype=torch.int32).cpu(),
                point_superpoint_ids=valid_sp.detach().to(
                    dtype=torch.int32).cpu(),
                point_xyz=(point_xyz[valid_point].detach().cpu()
                           if point_xyz is not None else None),
                point_instance_ids=(valid_instance_ids.detach().to(
                    dtype=torch.int32).cpu()
                    if valid_instance_ids is not None else None),
                fragment_tokens=proto.detach().to(dtype=torch.float16).cpu(),
                aligned_fragment_tokens=self.object_feedback_proto_norm(
                    current_proto).detach().to(dtype=torch.float16).cpu())

        input_norm = voxel_feat.norm(p=2, dim=1).mean().clamp(min=self._object_feedback_eps)
        ratio = float(update.detach().norm(p=2, dim=1).mean().div(input_norm.detach()).item())
        gate_mean = float((voxel_gate.detach().mean() * point_gate.detach().mean()).item()) if voxel_gate.numel() and point_gate.numel() else 0.0
        self._object_feedback_stats(
            enabled=True,
            mode=mode,
            fragments=int(n_frag),
            support_voxels=int(valid_voxels.shape[0]),
            s2_hit=float(map_hit),
            gate_mean=gate_mean,
            delta_input_norm_ratio=ratio,
            dino_valid_rate=float(dino_valid_rate),
            purity_mean=(float(purity.mean().item())
                         if purity is not None and purity.numel() else 0.0),
            write_keep_rate=(float(write_keep.mean().item())
                             if write_keep is not None and write_keep.numel()
                             else 1.0),
            memory_hit_rate=(float(memory_hit.mean().item())
                             if memory_hit.numel() else 0.0),
            memory_objects=sum(
                len(batch_memory)
                for batch_memory in self._object_feedback_memory))
        if mode == 'stats-only':
            return out_b1p2

        out_features = out_b1p2.features.clone()
        out_features[valid_voxels] = out_features[valid_voxels] + update
        return ME.SparseTensor(
            features=out_features,
            coordinate_map_key=out_b1p2.coordinate_map_key,
            tensor_stride=out_b1p2.tensor_stride,
            coordinate_manager=out_b1p2.coordinate_manager)

    def _fuse_with_dino(self, out_up, out_skip, dino_feat, fuse):
        """DITR 风格：在 decoder level 将三路特征 concat 后投影融合。

        - out_up: 当前 decoder 上采样分支特征（convtr+bn+relu 后）
        - out_skip: encoder skip 特征（同 stride）
        - dino_feat: 同 stride 的 DINO 稀疏特征（可为 None）
        - fuse: 1×1 conv + BN + GELU（输出通道与 out_up⊕out_skip 一致）
        """
        base_cat = me.cat(out_up, out_skip)

        if dino_feat is None:
            if getattr(self, '_dino_strict', False):
                raise RuntimeError(
                    f"DINO strict: missing dino_feat for tensor_stride={tuple(out_up.tensor_stride)}")
            return base_cat

        try:
            dino_on_out = dino_feat.features_at_coordinates(out_up.coordinates.float())
        except Exception as e:
            if getattr(self, '_dino_strict', False):
                raise RuntimeError(
                    f"DINO strict: features_at_coordinates failed for tensor_stride={tuple(out_up.tensor_stride)}: {repr(e)}")
            return base_cat

        if dino_on_out.dim() != 2 or dino_on_out.shape[0] != out_up.features.shape[0]:
            if getattr(self, '_dino_strict', False):
                raise RuntimeError(
                    "DINO strict: features_at_coordinates returned invalid shape "
                    f"got={tuple(getattr(dino_on_out, 'shape', []))}, expected=({out_up.features.shape[0]}, C)")
            return base_cat

        # 命中率检查（避免坐标 miss 时把 backbone 特征通过随机 fuse 破坏掉）
        try:
            hit = (dino_on_out.abs().sum(dim=1) > 1e-8).float().mean().item()
            # 记录到成员变量，供训练期 sanity hook 汇总
            try:
                s = int(getattr(out_up, 'tensor_stride', (0,))[0])
                key = {1: 's1', 2: 's2', 4: 's4', 8: 's8', 16: 's16'}.get(s, f's{s}')
                if hasattr(self, '_last_dino_hit') and isinstance(self._last_dino_hit, dict):
                    self._last_dino_hit[key] = float(hit)
            except Exception:
                pass
            if hit < getattr(self, '_dino_min_hit_ratio', 0.0):
                if getattr(self, '_dino_strict', False):
                    raise RuntimeError(
                        f"DINO strict: low hit_ratio={hit:.4f} (<{self._dino_min_hit_ratio:.4f}) "
                        f"at tensor_stride={tuple(out_up.tensor_stride)}; check elastic/flip/resize/intrinsics sync.")
                if getattr(self, '_dino_miss_warn_count', 0) < 3:
                    print(f"[DINO][warn] low hit_ratio={hit:.4f} (<{self._dino_min_hit_ratio:.4f}), fallback to 3D-only concat")
                    self._dino_miss_warn_count += 1
                return base_cat
        except Exception:
            # 若检查失败，宁可继续走融合（不影响正常训练）
            pass

        dino_on_out = dino_on_out.to(dtype=out_up.features.dtype)
        dino_on_out = ME.SparseTensor(
            features=dino_on_out,
            coordinate_map_key=out_up.coordinate_map_key,
            tensor_stride=out_up.tensor_stride,
            coordinate_manager=out_up.coordinate_manager,
        )
        fused_in = me.cat(out_up, out_skip, dino_on_out)
        fused_out = fuse(fused_in)
        # 记录融合尺度统计（仅用于训练期监控）
        try:
            s = int(getattr(out_up, 'tensor_stride', (0,))[0])
            key = {1: 's1', 2: 's2', 4: 's4', 8: 's8', 16: 's16'}.get(s, f's{s}')
            base_feat = base_cat.features
            fuse_feat = fused_out.features
            # 均值/方差
            dino_mean = float(dino_on_out.features.mean())
            dino_std = float(dino_on_out.features.std())
            base_mean = float(base_feat.mean())
            base_std = float(base_feat.std())
            # L2 norm 比例
            base_norm = float(base_feat.norm(p=2, dim=1).mean())
            fuse_norm = float(fuse_feat.norm(p=2, dim=1).mean())
            ratio = float(fuse_norm / (base_norm + 1e-6))
            if hasattr(self, '_last_dino_fuse') and isinstance(self._last_dino_fuse, dict):
                self._last_dino_fuse[key] = dict(
                    base_norm=base_norm,
                    fuse_norm=fuse_norm,
                    ratio=ratio,
                    dino_mean=dino_mean,
                    dino_std=dino_std,
                    base_mean=base_mean,
                    base_std=base_std,
                )
        except Exception:
            pass
        if getattr(self, '_dino_residual', False):
            return base_cat + fused_out
        return fused_out

    def forward(self, x, dino_feats=None, memory=None, sp_pts_mask=None,
                point_coords=None, point_xyz=None, point_instance_ids=None):
        """dino_feats: 可选 list/tuple，期望顺序 [s1, s2, s4, s8, s16]
        解码注入逻辑：
          - 仅在 self.use_dino 且提供 dino_feats 时生效；
          - 在各 decoder level，将 (up, skip, dino) 三路特征 concat 后做 1×1+BN+GELU 融合；
          - 输出通道与原始 (up ⊕ skip) 一致，保持 U-Net 主干结构与 block 输入不变。
        """
        if self.use_dino and hasattr(self, '_last_dino_hit') and isinstance(self._last_dino_hit, dict):
            self._last_dino_hit = {}
        if self.use_dino and hasattr(self, '_last_dino_fuse') and isinstance(self._last_dino_fuse, dict):
            self._last_dino_fuse = {}

        dino_s1 = dino_s2 = dino_s4 = dino_s8 = dino_s16 = None
        if self.use_dino and dino_feats is not None and len(dino_feats) >= 4:
            # build_sparse_fpn 返回 [s1, s2, s4, s8, s16]
            dino_s1 = dino_feats[0]
            dino_s2 = dino_feats[1]
            dino_s4 = dino_feats[2]
            dino_s8 = dino_feats[3]
            if len(dino_feats) >= 5:
                dino_s16 = dino_feats[4]

        out = self.conv0p1s1(x)
        out = self.bn0(out)
        out_p1 = self.relu(out)

        out = self.conv1p1s2(out_p1)
        out = self.bn1(out)
        out = self.relu(out)
        out_b1p2 = self.block1(out)
        if self._object_feedback_placement == 'pre_memory':
            out_b1p2 = self._apply_object_feedback_s2(
                out_b1p2,
                dino_s2=dino_s2,
                sp_pts_mask=sp_pts_mask,
                point_coords=point_coords,
                point_xyz=point_xyz,
                point_instance_ids=point_instance_ids)

        out = self.conv2p2s2(out_b1p2)
        out = self.bn2(out)
        out = self.relu(out)
        out_b2p4 = self.block2(out)

        out = self.conv3p4s2(out_b2p4)
        out = self.bn3(out)
        out = self.relu(out)
        out_b3p8 = self.block3(out)

        # pixel_dist=16
        out = self.conv4p8s2(out_b3p8)
        out = self.bn4(out)
        out = self.relu(out)
        out = self.block4(out)

        # stride=16 的坐标命中率统计（不参与融合，只用于 sanity/调试）
        if self.use_dino and dino_s16 is not None:
            try:
                d16 = dino_s16.features_at_coordinates(out.coordinates.float())
                hit16 = (d16.abs().sum(dim=1) > 1e-8).float().mean().item()
                if hasattr(self, '_last_dino_hit') and isinstance(self._last_dino_hit, dict):
                    self._last_dino_hit['s16'] = float(hit16)
            except Exception:
                pass

        if memory is not None:
            out_b1p2_temp, out_b2p4_temp, out_b3p8_temp, out_temp = out_b1p2, out_b2p4, out_b3p8, out
            out_b1p2, out_b2p4, out_b3p8, out = memory([out_b1p2, out_b2p4, out_b3p8, out])
            out_b1p2 = ME.SparseTensor(coordinate_map_key=out_b1p2_temp.coordinate_map_key, features=out_b1p2.features_at_coordinates(out_b1p2_temp.coordinates.float()), tensor_stride=out_b1p2_temp.tensor_stride, coordinate_manager=out_b1p2_temp.coordinate_manager)
            out_b2p4 = ME.SparseTensor(coordinate_map_key=out_b2p4_temp.coordinate_map_key, features=out_b2p4.features_at_coordinates(out_b2p4_temp.coordinates.float()), tensor_stride=out_b2p4_temp.tensor_stride, coordinate_manager=out_b2p4_temp.coordinate_manager)
            out_b3p8 = ME.SparseTensor(coordinate_map_key=out_b3p8_temp.coordinate_map_key, features=out_b3p8.features_at_coordinates(out_b3p8_temp.coordinates.float()), tensor_stride=out_b3p8_temp.tensor_stride, coordinate_manager=out_b3p8_temp.coordinate_manager)
            out = ME.SparseTensor(coordinate_map_key=out_temp.coordinate_map_key, features=out.features_at_coordinates(out_temp.coordinates.float()), tensor_stride=out_temp.tensor_stride, coordinate_manager=out_temp.coordinate_manager)
        if self._object_feedback_placement == 'post_memory':
            out_b1p2 = self._apply_object_feedback_s2(
                out_b1p2,
                dino_s2=dino_s2,
                sp_pts_mask=sp_pts_mask,
                point_coords=point_coords,
                point_xyz=point_xyz,
                point_instance_ids=point_instance_ids)

        # pixel_dist=8
        out = self.convtr4p16s2(out)
        out = self.bntr4(out)
        out = self.relu(out)
        if (self.use_dino and self._dino_point_fusion_enable and
                not self._dual_output):
            out = self._fuse_with_dino(out, out_b3p8, dino_s8, self.dino_fuse_8)
        else:
            out = me.cat(out, out_b3p8)
        out = self.block5(out)

        # pixel_dist=4
        out = self.convtr5p8s2(out)
        out = self.bntr5(out)
        out = self.relu(out)
        if (self.use_dino and self._dino_point_fusion_enable and
                not self._dual_output):
            out = self._fuse_with_dino(out, out_b2p4, dino_s4, self.dino_fuse_4)
        else:
            out = me.cat(out, out_b2p4)
        out = self.block6(out)

        # pixel_dist=2
        out = self.convtr6p4s2(out)
        out = self.bntr6(out)
        out = self.relu(out)
        if (self.use_dino and self._dino_point_fusion_enable and
                not self._dual_output):
            out = self._fuse_with_dino(out, out_b1p2, dino_s2, self.dino_fuse_2)
        else:
            out = me.cat(out, out_b1p2)
        out = self.block7(out)

        # pixel_dist=1
        out = self.convtr7p2s2(out)
        out = self.bntr7(out)
        out = self.relu(out)

        if (self._dual_output and self.use_dino and
                self._dino_point_fusion_enable):
            # ── Semantic side branch: detach 3D features + DINO fusion ──
            out_sg = ME.SparseTensor(
                features=out.features.detach(),
                coordinate_map_key=out.coordinate_map_key,
                tensor_stride=out.tensor_stride,
                coordinate_manager=out.coordinate_manager,
            )
            sem_feat = self._fuse_with_dino(out_sg, out_p1, dino_s1, self.dino_fuse_1)
            sem_1_feats = self.sem_out_proj(sem_feat.features)
            sem_1 = ME.SparseTensor(
                features=sem_1_feats,
                coordinate_map_key=sem_feat.coordinate_map_key,
                tensor_stride=sem_feat.tensor_stride,
                coordinate_manager=sem_feat.coordinate_manager,
            )
            # Main path: pure 3D (no DINO injection)
            out = me.cat(out, out_p1)
            out = self.block8(out)
            return out, sem_1
        else:
            if self.use_dino and self._dino_point_fusion_enable:
                out = self._fuse_with_dino(out, out_p1, dino_s1, self.dino_fuse_1)
            else:
                out = me.cat(out, out_p1)
            out = self.block8(out)
            return out

class Res16UNetBase_FF(Res16UNetBase):
    """Base class for Minkowski U-Net."""

    def forward(self, x, f=None, dino_feats=None, memory=None):
        dino_s1 = dino_s2 = dino_s4 = dino_s8 = None
        if self.use_dino and dino_feats is not None and len(dino_feats) >= 4:
            dino_s1 = dino_feats[0]
            dino_s2 = dino_feats[1]
            dino_s4 = dino_feats[2]
            dino_s8 = dino_feats[3]

        out = self.conv0p1s1(x)
        out = self.bn0(out)
        out_p1 = self.relu(out)
        if f is not None:
            out_p1 = f(out_p1)
        
        out = self.conv1p1s2(out_p1)
        out = self.bn1(out)
        out = self.relu(out)
        out_b1p2 = self.block1(out)

        out = self.conv2p2s2(out_b1p2)
        out = self.bn2(out)
        out = self.relu(out)
        out_b2p4 = self.block2(out)

        out = self.conv3p4s2(out_b2p4)
        out = self.bn3(out)
        out = self.relu(out)
        out_b3p8 = self.block3(out)

        # pixel_dist=16
        out = self.conv4p8s2(out_b3p8)
        out = self.bn4(out)
        out = self.relu(out)
        out = self.block4(out)

        if memory is not None:
            out_b1p2_temp, out_b2p4_temp, out_b3p8_temp, out_temp = out_b1p2, out_b2p4, out_b3p8, out
            out_b1p2, out_b2p4, out_b3p8, out = memory([out_b1p2, out_b2p4, out_b3p8, out])
            out_b1p2 = ME.SparseTensor(coordinate_map_key=out_b1p2_temp.coordinate_map_key, features=out_b1p2.features_at_coordinates(out_b1p2_temp.coordinates.float()), tensor_stride=out_b1p2_temp.tensor_stride, coordinate_manager=out_b1p2_temp.coordinate_manager)
            out_b2p4 = ME.SparseTensor(coordinate_map_key=out_b2p4_temp.coordinate_map_key, features=out_b2p4.features_at_coordinates(out_b2p4_temp.coordinates.float()), tensor_stride=out_b2p4_temp.tensor_stride, coordinate_manager=out_b2p4_temp.coordinate_manager)
            out_b3p8 = ME.SparseTensor(coordinate_map_key=out_b3p8_temp.coordinate_map_key, features=out_b3p8.features_at_coordinates(out_b3p8_temp.coordinates.float()), tensor_stride=out_b3p8_temp.tensor_stride, coordinate_manager=out_b3p8_temp.coordinate_manager)
            out = ME.SparseTensor(coordinate_map_key=out_temp.coordinate_map_key, features=out.features_at_coordinates(out_temp.coordinates.float()), tensor_stride=out_temp.tensor_stride, coordinate_manager=out_temp.coordinate_manager)

        # pixel_dist=8
        out = self.convtr4p16s2(out)
        out = self.bntr4(out)
        out = self.relu(out)
        if self.use_dino:
            out = self._inject_dino(out, dino_s8, self.dino_proj_8)
        out = me.cat(out, out_b3p8)
        out = self.block5(out)

        # pixel_dist=4
        out = self.convtr5p8s2(out)
        out = self.bntr5(out)
        out = self.relu(out)
        if self.use_dino:
            out = self._inject_dino(out, dino_s4, self.dino_proj_4)
        out = me.cat(out, out_b2p4)
        out = self.block6(out)

        # pixel_dist=2
        out = self.convtr6p4s2(out)
        out = self.bntr6(out)
        out = self.relu(out)
        if self.use_dino:
            out = self._inject_dino(out, dino_s2, self.dino_proj_2)
        out = me.cat(out, out_b1p2)
        out = self.block7(out)

        # pixel_dist=1
        out = self.convtr7p2s2(out)
        out = self.bntr7(out)
        out = self.relu(out)
        if self.use_dino:
            out = self._inject_dino(out, dino_s1, self.dino_proj_1)
        out = me.cat(out, out_p1)
        out = self.block8(out)

        return out


class Res16UNet34(Res16UNetBase):
    BLOCK = BasicBlock
    LAYERS = (2, 3, 4, 6, 2, 2, 2, 2)

class Res16UNet34_FF(Res16UNetBase_FF):
    BLOCK = BasicBlock
    LAYERS = (2, 3, 4, 6, 2, 2, 2, 2)

@MODELS.register_module()
class Res16UNet34C(Res16UNet34):
    PLANES = (32, 64, 128, 256, 256, 128, 96, 96)

@MODELS.register_module()
class Res16UNet34C_FF(Res16UNet34_FF):
    PLANES = (32, 64, 128, 256, 256, 128, 96, 96)    
    
