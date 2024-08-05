from typing import List, Optional, Union

import torch
from torch import nn
import torch.nn.functional as F

from .mlp import MLP
from .normalization_layers import AdaIN
from .skip_connections import skip_connection
from .spectral_convolution import SpectralConv
from ..utils import validate_scaling_factor


Number = Union[int, float]


class FNOBlocks(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        n_modes,
        output_scaling_factor: Optional[Union[Number, List[Number]]] = None,
        n_layers=1,
        incremental_n_modes=None,
        fno_block_precision="full",
        use_mlp=False,
        mlp_dropout=0,
        mlp_expansion=0.5,
        non_linearity=F.gelu,
        stabilizer=None,
        norm=None,
        ada_in_features=None,
        preactivation=False,
        fno_skip="linear",
        mlp_skip="soft-gating",
        separable=False,
        factorization=None,
        rank=1.0,
        SpectralConv=SpectralConv,
        joint_factorization=False,
        fixed_rank_modes=False,
        implementation="factorized",
        decomposition_kwargs=dict(),
        fft_norm="forward",
        **kwargs,
    ):
        super().__init__()
        if isinstance(n_modes, int):
            n_modes = [n_modes]
        self.n_modes = n_modes
        self.n_dim = len(n_modes)

        self.output_scaling_factor: Union[
            None, List[List[float]]
        ] = validate_scaling_factor(output_scaling_factor, self.n_dim, n_layers)

        self._incremental_n_modes = incremental_n_modes
        self.fno_block_precision = fno_block_precision
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_layers = n_layers
        self.joint_factorization = joint_factorization
        self.non_linearity = non_linearity
        self.stabilizer = stabilizer
        self.rank = rank
        self.factorization = factorization
        self.fixed_rank_modes = fixed_rank_modes
        self.decomposition_kwargs = decomposition_kwargs
        self.fno_skip = fno_skip
        self.mlp_skip = mlp_skip
        self.use_mlp = use_mlp
        self.mlp_expansion = mlp_expansion
        self.mlp_dropout = mlp_dropout
        self.fft_norm = fft_norm
        self.implementation = implementation
        self.separable = separable
        self.preactivation = preactivation
        self.ada_in_features = ada_in_features

        self.convs = SpectralConv(
            self.in_channels,
            self.out_channels,
            self.n_modes,
            output_scaling_factor=output_scaling_factor,
            incremental_n_modes=incremental_n_modes,
            rank=rank,
            fixed_rank_modes=fixed_rank_modes,
            implementation=implementation,
            separable=separable,
            factorization=factorization,
            decomposition_kwargs=decomposition_kwargs,
            joint_factorization=joint_factorization,
            n_layers=n_layers,
        )

        self.fno_skips = nn.ModuleList(
            [
                skip_connection(
                    self.in_channels,
                    self.out_channels,
                    skip_type=fno_skip,
                    n_dim=self.n_dim,
                )
                for _ in range(n_layers)
            ]
        )

        if use_mlp:
            self.mlp = nn.ModuleList(
                [
                    MLP(
                        in_channels=self.out_channels,
                        hidden_channels=round(self.out_channels * mlp_expansion),
                        dropout=mlp_dropout,
                        n_dim=self.n_dim,
                    )
                    for _ in range(n_layers)
                ]
            )
            self.mlp_skips = nn.ModuleList(
                [
                    skip_connection(
                        self.in_channels,
                        self.out_channels,
                        skip_type=mlp_skip,
                        n_dim=self.n_dim,
                    )
                    for _ in range(n_layers)
                ]
            )
        else:
            self.mlp = None

        # Each block will have 2 norms if we also use an MLP
        self.n_norms = 1 if self.mlp is None else 2
        if norm is None:
            self.norm = None
        elif norm == "instance_norm":
            self.norm = nn.ModuleList(
                [
                    getattr(nn, f"InstanceNorm{self.n_dim}d")(
                        num_features=self.out_channels
                    )
                    for _ in range(n_layers * self.n_norms)
                ]
            )
        elif norm == "group_norm":
            self.norm = nn.ModuleList(
                [
                    nn.GroupNorm(num_groups=1, num_channels=self.out_channels)
                    for _ in range(n_layers * self.n_norms)
                ]
            )
        # elif norm == 'layer_norm':
        #     self.norm = nn.ModuleList(
        #         [
        #             nn.LayerNorm(elementwise_affine=False)
        #             for _ in range(n_layers*self.n_norms)
        #         ]
        #     )
        elif norm == "ada_in":
            self.norm = nn.ModuleList(
                [
                    AdaIN(ada_in_features, out_channels)
                    for _ in range(n_layers * self.n_norms)
                ]
            )
        else:
            raise ValueError(
                f"Got norm={norm} but expected None or one of "
                "[instance_norm, group_norm, layer_norm]"
            )

    def set_ada_in_embeddings(self, *embeddings):
        """Sets the embeddings of each Ada-IN norm layers

        Parameters
        ----------
        embeddings : tensor or list of tensor
            if a single embedding is given, it will be used for each norm layer
            otherwise, each embedding will be used for the corresponding norm layer
        """
        if len(embeddings) == 1:
            for norm in self.norm:
                norm.set_embedding(embeddings[0])
        else:
            for norm, embedding in zip(self.norm, embeddings):
                norm.set_embedding(embedding)

    def forward(self, x, index=0, output_shape=None):
        if self.preactivation:
            return self.forward_with_preactivation(x, index, output_shape)
        else:
            return self.forward_with_postactivation(x, index, output_shape)

    def forward_with_postactivation(self, x, index=0, output_shape=None):
        x_skip_fno = self.fno_skips[index](x)
        x_skip_fno = self.convs[index].transform(x_skip_fno, output_shape=output_shape)

        if self.mlp is not None:
            x_skip_mlp = self.mlp_skips[index](x)
            x_skip_mlp = self.convs[index].transform(x_skip_mlp, output_shape=output_shape)

        if self.stabilizer == "tanh":
            x = torch.tanh(x)

        x_fno = self.convs(x, index, output_shape=output_shape)

        if self.norm is not None:
            x_fno = self.norm[self.n_norms * index](x_fno)

        x = x_fno + x_skip_fno

        if (self.mlp is not None) or (index < (self.n_layers - 1)):
            x = self.non_linearity(x)

        if self.mlp is not None:
            x = self.mlp[index](x) + x_skip_mlp

            if self.norm is not None:
                x = self.norm[self.n_norms * index + 1](x)

            if index < (self.n_layers - 1):
                x = self.non_linearity(x)

        return x

    def forward_with_preactivation(self, x, index=0, output_shape=None):
        # Apply non-linear activation (and norm)
        # before this block's convolution/forward pass:
        x = self.non_linearity(x)

        if self.norm is not None:
            x = self.norm[self.n_norms * index](x)

        x_skip_fno = self.fno_skips[index](x)
        x_skip_fno = self.convs[index].transform(x_skip_fno, output_shape=output_shape)

        if self.mlp is not None:
            x_skip_mlp = self.mlp_skips[index](x)
            x_skip_mlp = self.convs[index].transform(x_skip_mlp, output_shape=output_shape)

        if self.stabilizer == "tanh":
            x = torch.tanh(x)

        x_fno = self.convs(x, index, output_shape=output_shape)
        x = x_fno + x_skip_fno

        if self.mlp is not None:
            if index < (self.n_layers - 1):
                x = self.non_linearity(x)

            if self.norm is not None:
                x = self.norm[self.n_norms * index + 1](x)

            x = self.mlp[index](x) + x_skip_mlp

        return x

    @property
    def incremental_n_modes(self):
        return self._incremental_n_modes

    @incremental_n_modes.setter
    def incremental_n_modes(self, incremental_n_modes):
        self.convs.incremental_n_modes = incremental_n_modes

    def get_block(self, indices):
        """Returns a sub-FNO Block layer from the jointly parametrized main block

        The parametrization of an FNOBlock layer is shared with the main one.
        """
        if self.n_layers == 1:
            raise ValueError(
                "A single layer is parametrized, directly use the main class."
            )

        return SubModule(self, indices)

    def __getitem__(self, indices):
        return self.get_block(indices)

class SubModule(nn.Module):
    """Class representing one of the sub_module from the mother joint module

    Notes
    -----
    This relies on the fact that nn.Parameters are not duplicated:
    if the same nn.Parameter is assigned to multiple modules,
    they all point to the same data, which is shared.
    """

    def __init__(self, main_module, indices):
        super().__init__()
        self.main_module = main_module
        self.indices = indices

    def forward(self, x):
        return self.main_module.forward(x, self.indices)
    
from .quad_prod import *

"""
    the original version merely considers mlp for channel mixing, 
    but here we would provide a wider range of choice.
    stablizer added clamp(-1, 1)
    factorization added f-fno
"""
class FNOBlocks1(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        n_modes,
        output_scaling_factor: Optional[Union[Number, List[Number]]] = None,
        n_layers=1,
        incremental_n_modes=None,
        fno_block_precision="full",
        channel_mixing="",
        mlp_dropout=0,
        mlp_expansion=0.5,
        num_prod=2,
        non_linearity=F.gelu,
        stabilizer=None,
        norm=None,
        ada_in_features=None,
        preactivation=False,
        fno_skip="linear",
        mixer_skip="soft-gating",
        separable=False,
        factorization=None,
        rank=1.0,
        SpectralConv=SpectralConv,
        joint_factorization=False,
        fixed_rank_modes=False,
        implementation="factorized",
        decomposition_kwargs=dict(),
        fft_norm="forward",
        **kwargs,
    ):
        super().__init__()
        if isinstance(n_modes, int):
            n_modes = [n_modes]
        self.n_modes = n_modes
        self.n_dim = len(n_modes)

        self.output_scaling_factor: Union[
            None, List[List[float]]
        ] = validate_scaling_factor(output_scaling_factor, self.n_dim, n_layers)

        self._incremental_n_modes = incremental_n_modes
        self.fno_block_precision = fno_block_precision
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_layers = n_layers
        self.joint_factorization = joint_factorization
        self.non_linearity = non_linearity
        self.rank = rank
        self.factorization = factorization
        self.fixed_rank_modes = fixed_rank_modes
        self.decomposition_kwargs = decomposition_kwargs
        self.fno_skip = fno_skip
        self.mixer_skip = mixer_skip
        self.use_channel_mixing = channel_mixing != ""
        self.channel_mixing = channel_mixing
        self.mlp_expansion = mlp_expansion
        self.mlp_dropout = mlp_dropout
        self.num_prod = num_prod
        self.fft_norm = fft_norm
        self.implementation = implementation
        self.separable = separable
        self.preactivation = preactivation
        self.ada_in_features = ada_in_features
        self.stabilizer = None
        if stabilizer == 'tanh':
            self.stabilizer = torch.tanh
        elif stabilizer == 'clamp':
            self.stabilizer = lambda x: torch.clamp(x, -1., 1.)

        self.convs = SpectralConv(
            self.in_channels,
            self.out_channels,
            self.n_modes,
            output_scaling_factor=output_scaling_factor,
            incremental_n_modes=incremental_n_modes,
            rank=rank,
            fixed_rank_modes=fixed_rank_modes,
            implementation=implementation,
            separable=separable,
            factorization=factorization,
            decomposition_kwargs=decomposition_kwargs,
            joint_factorization=joint_factorization,
            n_layers=n_layers,
        )

        self.fno_skips = nn.ModuleList(
            [
                skip_connection(
                    self.in_channels,
                    self.out_channels,
                    skip_type=fno_skip,
                    n_dim=self.n_dim,
                )
                for _ in range(n_layers)
            ]
        )

        if channel_mixing == 'mlp':
            self.channel_mixer = nn.ModuleList(
                [
                    MLP(
                        in_channels=self.out_channels,
                        hidden_channels=round(self.out_channels * mlp_expansion),
                        dropout=mlp_dropout,
                        n_dim=self.n_dim,
                    )
                    for _ in range(n_layers)
                ]
            )
        elif channel_mixing == 'prod-gating':
            self.channel_mixer = nn.ModuleList(
                [
                    ProductGating(self.out_channels, num_prod=num_prod)
                    for _ in range(n_layers)
                ]
            )
        elif channel_mixing == 'prod-layer':
            self.channel_mixer = nn.ModuleList(
                [
                    nn.Sequential(
                        MLP(
                            in_channels=self.out_channels,
                            hidden_channels=round(self.out_channels * mlp_expansion),
                            dropout=mlp_dropout,
                            n_dim=self.n_dim,
                            n_layers=1
                        ),
                        nn.SiLU(),
                        ProductLayer(
                            in_dim=self.out_channels,
                            num_prods=num_prod,
                            out_dim=self.out_channels,
                            n_dim=self.n_dim
                        )
                    )
                    for _ in range(n_layers)
                ]
            )
        elif channel_mixing == 'quad-layer':
            self.channel_mixer = nn.ModuleList(
                [
                    QuadPath(in_dim=self.out_channels,out_dim=self.out_channels,num_quad=num_prod, num_prod=num_prod)
                    for _ in range(n_layers)
                ]
            )
        else:
            self.use_channel_mixing=False
            self.channel_mixer = None
        
        if self.use_channel_mixing:
            self.mixer_skips = nn.ModuleList(
                [
                    skip_connection(
                        self.in_channels,
                        self.out_channels,
                        skip_type=mixer_skip,
                        n_dim=self.n_dim,
                    )
                    for _ in range(n_layers)
                ]
            )
        else:
            self.mixer_skips=None

        # Each block will have 2 norms if we use an channel mixing
        self.n_norms = 2 if self.use_channel_mixing else 1
        if norm is None:
            self.norm = None
        elif norm == "instance_norm":
            self.norm = nn.ModuleList(
                [
                    getattr(nn, f"InstanceNorm{self.n_dim}d")(
                        num_features=self.out_channels
                    )
                    for _ in range(n_layers * self.n_norms)
                ]
            )
        elif norm == "group_norm":
            self.norm = nn.ModuleList(
                [
                    nn.GroupNorm(num_groups=1, num_channels=self.out_channels)
                    for _ in range(n_layers * self.n_norms)
                ]
            )
        # elif norm == 'layer_norm':
        #     self.norm = nn.ModuleList(
        #         [
        #             nn.LayerNorm(elementwise_affine=False)
        #             for _ in range(n_layers*self.n_norms)
        #         ]
        #     )
        elif norm == "ada_in":
            self.norm = nn.ModuleList(
                [
                    AdaIN(ada_in_features, out_channels)
                    for _ in range(n_layers * self.n_norms)
                ]
            )
        else:
            raise ValueError(
                f"Got norm={norm} but expected None or one of "
                "[instance_norm, group_norm, layer_norm]"
            )

    def set_ada_in_embeddings(self, *embeddings):
        """Sets the embeddings of each Ada-IN norm layers

        Parameters
        ----------
        embeddings : tensor or list of tensor
            if a single embedding is given, it will be used for each norm layer
            otherwise, each embedding will be used for the corresponding norm layer
        """
        if len(embeddings) == 1:
            for norm in self.norm:
                norm.set_embedding(embeddings[0])
        else:
            for norm, embedding in zip(self.norm, embeddings):
                norm.set_embedding(embedding)

    def forward(self, x, index=0, output_shape=None):
        if self.preactivation:
            return self.forward_with_preactivation(x, index, output_shape)
        else:
            return self.forward_with_postactivation(x, index, output_shape)

    def forward_with_postactivation(self, x, index=0, output_shape=None):
        x_skip_fno = self.fno_skips[index](x)
        x_skip_fno = self.convs[index].transform(x_skip_fno, output_shape=output_shape)

        if self.channel_mixer is not None:
            # print(self.mixer_skips)
            x_skip_mixer = self.mixer_skips[index](x)
            x_skip_mixer = self.convs[index].transform(x_skip_mixer, output_shape=output_shape)

        if self.stabilizer is not None:
            x = self.stabilizer(x)

        x_fno = self.convs(x, index, output_shape=output_shape)

        if self.norm is not None:
            x_fno = self.norm[self.n_norms * index](x_fno)

        x = x_fno + x_skip_fno

        if (self.channel_mixer is not None) or (index < (self.n_layers - 1)):
            x = self.non_linearity(x)

        if self.channel_mixer is not None:
            x = self.channel_mixer[index](x) + x_skip_mixer

            if self.norm is not None:
                x = self.norm[self.n_norms * index + 1](x)

            if index < (self.n_layers - 1):
                x = self.non_linearity(x)

        return x

    def forward_with_preactivation(self, x, index=0, output_shape=None):
        # Apply non-linear activation (and norm)
        # before this block's convolution/forward pass:
        x = self.non_linearity(x)

        if self.norm is not None:
            x = self.norm[self.n_norms * index](x)

        x_skip_fno = self.fno_skips[index](x)
        x_skip_fno = self.convs[index].transform(x_skip_fno, output_shape=output_shape)

        if self.channel_mixer is not None:
            x_skip_mixer = self.mixer_skips[index](x)
            x_skip_mixer = self.convs[index].transform(x_skip_mixer, output_shape=output_shape)

        if self.stabilizer == "tanh":
            x = torch.tanh(x)

        x_fno = self.convs(x, index, output_shape=output_shape)
        x = x_fno + x_skip_fno

        if self.channel_mixer is not None:
            if index < (self.n_layers - 1):
                x = self.non_linearity(x)

            if self.norm is not None:
                x = self.norm[self.n_norms * index + 1](x)

            x = self.channel_mixer[index](x) + x_skip_mixer

        return x

    @property
    def incremental_n_modes(self):
        return self._incremental_n_modes

    @incremental_n_modes.setter
    def incremental_n_modes(self, incremental_n_modes):
        self.convs.incremental_n_modes = incremental_n_modes

    def get_block(self, indices):
        """Returns a sub-FNO Block layer from the jointly parametrized main block

        The parametrization of an FNOBlock layer is shared with the main one.
        """
        if self.n_layers == 1:
            raise ValueError(
                "A single layer is parametrized, directly use the main class."
            )

        return SubModule(self, indices)

    def __getitem__(self, indices):
        return self.get_block(indices)

class ForgetGating(nn.Module):
    def __init__(self, in_features=0, out_features=0, n_dim=2, weight=False, bias=False):
        super().__init__()
        if out_features<=0 or in_features < out_features:
            raise ValueError(
                f"Got in_features={in_features} and out_features={out_features}"
                "but should have out_features>0 and in_features >= out_features"
            )
        self.out_features = out_features
        if weight:
            self.weight = nn.Parameter(torch.ones(1, self.out_features, *(1,) * n_dim))
        else:
            self.weight = None
        if bias:
            self.bias = nn.Parameter(torch.ones(1, self.out_features, *(1,) * n_dim))
        else:
            self.bias = None

    def forward(self, x):
        """Applies soft-gating to a batch of activations"""
        x = x[:, :self.out_features, ...]
        if self.weight is not None:
            x = self.weight * x
        if self.bias is not None:
            return x + self.bias
        else:
            return x
        

class MultiActProdLayer(nn.Module):
    def __init__(self, in_dim, out_dim, exp_ratio=0.3, inv_ratio=0.3, num_prods=0, n_dim=2):
        super(MultiActProdLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_prods = num_prods
        self.exp_dim = round(exp_ratio*in_dim)
        self.inv_dim = round(inv_ratio*in_dim)
        self.id_dim = self.in_dim - self.exp_dim - self.inv_dim
        self.linear_out_dim = out_dim - num_prods
        
        assert self.linear_out_dim >= 2*num_prods, "Error: in_dim < 2*num_prods!"
        # self.linear = nn.Linear(self.in_dim_linear, self.out_dim, bias=False)
        self.linear = MLP(self.in_dim, self.linear_out_dim, n_layers=1, n_dim=n_dim)
        self.range_prods = torch.tensor(np.array(range(self.num_prods)), dtype=int)

        self.safe_exp = torch.nn.ELU(10)
        # self.inv_thresh = torch.nn.Parameter(torch.tensor(1.), requires_grad=True)
    
    def safe_inv(self, x:torch.Tensor):
        return 1/(torch.abs(x)+2e-1)
        # condition = x > self.inv_thresh
        # return torch.where(condition, 1/x, -(x - 2*self.inv_thresh) / self.inv_thresh**2)

    def get_prods(self, x):
        return torch.mul(x[:, : 2*self.num_prods-1: 2, ...], x[:, 1: 2*self.num_prods: 2, ...])

    def forward(self, x):
        x_exp = self.safe_exp(-x[:, -(self.exp_dim+self.inv_dim):-self.inv_dim, ...])
        x_inv = self.safe_inv(x[:, -self.inv_dim:, ...])
        x = torch.cat([x[:, :self.id_dim], x_exp, x_inv], dim=1)
        x = self.linear(x)
        x = torch.cat([x, self.get_prods(x)], dim=1)
        return x


class FNOBlocks2(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        n_modes,
        output_scaling_factor: Optional[Union[Number, List[Number]]] = None,
        n_layers=1,
        incremental_n_modes=None,
        fno_block_precision="full",
        channel_mixing="complex-mlp",
        n_mixer_layers=1,
        mlp_dropout=0,
        mlp_expansion=0.5,
        exp_ratio=0.3,
        inv_ratio=0.3,
        num_prod=2,
        non_linearity=torch.nn.Identity(),
        stabilizer=None,
        fno_skip_softgating=False,
        fno_skip_bias=True,
        skip_retain_ratio=0.25,
        mixer_skip='soft-gating',
        separable=False,
        factorization=None,
        rank=1.0,
        SpectralConv=SpectralConv,
        joint_factorization=False,
        fixed_rank_modes=False,
        implementation="factorized",
        decomposition_kwargs=dict(),
        fft_norm="forward",
        **kwargs,
    ):
        super().__init__()
        if isinstance(n_modes, int):
            n_modes = [n_modes]
        self.n_modes = n_modes
        self.n_dim = len(n_modes)

        self.output_scaling_factor: Union[
            None, List[List[float]]
        ] = validate_scaling_factor(output_scaling_factor, self.n_dim, n_layers)

        self._incremental_n_modes = incremental_n_modes
        self.fno_block_precision = fno_block_precision
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_layers = n_layers
        self.joint_factorization = joint_factorization
        self.non_linearity = non_linearity
        self.rank = rank
        self.factorization = factorization
        self.fixed_rank_modes = fixed_rank_modes
        self.decomposition_kwargs = decomposition_kwargs
        self.num_prod = num_prod
        self.fft_norm = fft_norm
        self.implementation = implementation
        self.separable = separable
        self.fno_skip_softgating = fno_skip_softgating
        self.fno_skip_bias = fno_skip_bias
        self.use_channel_mixing = channel_mixing != ""
        self.channel_mixing = channel_mixing
        self.n_mixer_layers = n_mixer_layers
        self.mlp_expansion = mlp_expansion
        self.mlp_dropout = mlp_dropout

        self.retain_dim = round(skip_retain_ratio * self.in_channels) 
        self.real_out_dim = out_channels - self.retain_dim

        self.stabilizer = None
        if stabilizer == 'tanh':
            self.stabilizer = torch.tanh
        elif stabilizer == 'clamp':
            self.stabilizer = lambda x: torch.clamp(x, -1., 1.)

        self.convs = SpectralConv(
            self.in_channels,
            self.out_channels,
            self.n_modes,
            output_scaling_factor=output_scaling_factor,
            incremental_n_modes=incremental_n_modes,
            rank=rank,
            fixed_rank_modes=fixed_rank_modes,
            implementation=implementation,
            separable=separable,
            factorization=factorization,
            decomposition_kwargs=decomposition_kwargs,
            joint_factorization=joint_factorization,
            n_layers=n_layers,
        )

        if channel_mixing == 'mlp':
            self.channel_mixer = nn.ModuleList(
                [
                    MLP(
                        in_channels=self.out_channels,
                        hidden_channels=round(self.out_channels * mlp_expansion),
                        dropout=mlp_dropout,
                        n_dim=self.n_dim,
                    )
                    for _ in range(n_layers)
                ]
            )
        elif channel_mixing == 'complex-mlp':
            self.channel_mixer = nn.ModuleList(
                [
                    nn.Sequential(
                        *[
                            MultiActProdLayer(
                                in_dim=self.out_channels,
                                # out_dim=self.real_out_dim,
                                out_dim=self.out_channels,
                                exp_ratio=exp_ratio,
                                inv_ratio=inv_ratio,
                                num_prods=num_prod,
                                n_dim=self.n_dim
                                )
                            for _ in range(n_mixer_layers)
                        ]
                    )
                    for _ in range(n_layers)
                ]
            )

        if self.use_channel_mixing:
            self.mixer_skips = nn.ModuleList(
                [
                    skip_connection(
                        self.in_channels,
                        self.out_channels,
                        skip_type=mixer_skip,
                        n_dim=self.n_dim,
                    )
                    for _ in range(n_layers)
                ]
            )
        else:
            self.mixer_skips=None
        
        self.fno_skips = nn.ModuleList(
            [
                ForgetGating(self.in_channels, self.retain_dim, weight=fno_skip_softgating, bias=fno_skip_bias, n_dim=self.n_dim)
                for _ in range(n_layers)
            ]
        )

    def forward(self, x, index=0, output_shape=None):
        x_skip_fno = self.fno_skips[index](x)
        x_skip_fno = self.convs[index].transform(x_skip_fno, output_shape=output_shape).squeeze(0)

        if self.channel_mixer is not None:
            x_skip_mixer = self.mixer_skips[index](x)
            # x_skip_mixer = x_skip_fno

        if self.stabilizer is not None:
            x = self.stabilizer(x)

        x_fno = self.convs(x, index, output_shape=output_shape)
        
        x = torch.cat([x_fno, x_skip_fno], dim=1)

        if (self.channel_mixer is not None) or (index < (self.n_layers - 1)):
            x = self.non_linearity(x)

        if self.channel_mixer is not None:
            x = self.channel_mixer[index](x) + x_skip_mixer
            # x = torch.cat([self.channel_mixer[index](x), x_skip_mixer], dim=1)

            if index < (self.n_layers - 1):
                x = self.non_linearity(x)

        return x

    

    @property
    def incremental_n_modes(self):
        return self._incremental_n_modes

    @incremental_n_modes.setter
    def incremental_n_modes(self, incremental_n_modes):
        self.convs.incremental_n_modes = incremental_n_modes

    def get_block(self, indices):
        """Returns a sub-FNO Block layer from the jointly parametrized main block

        The parametrization of an FNOBlock layer is shared with the main one.
        """
        if self.n_layers == 1:
            raise ValueError(
                "A single layer is parametrized, directly use the main class."
            )

        return SubModule(self, indices)

    def __getitem__(self, indices):
        return self.get_block(indices)

from .new_spectral_conv import SpectralConvFFNO2d, SpectralConvProd

class F_FNOBlocks2D(FNOBlocks1):
    def __init__(
        self,
        in_channels,
        out_channels,
        n_modes,
        output_scaling_factor: Optional[Union[Number, List[Number]]] = None,
        n_layers=1,
        incremental_n_modes=None,
        fno_block_precision="full",
        channel_mixing="",
        mlp_dropout=0,
        mlp_expansion=0.5,
        num_prod=2,
        non_linearity=F.gelu,
        stabilizer=None,
        norm=None,
        ada_in_features=None,
        preactivation=False,
        fno_skip="linear",
        mixer_skip="soft-gating",
        separable=False,
        factorization=None,
        rank=1.0,
        joint_factorization=False,
        fixed_rank_modes=False,
        implementation="factorized",
        decomposition_kwargs=dict(),
        fft_norm="forward",
        ffno_channel_mixing='add',
        **kwargs,
    ):
        super().__init__(in_channels=in_channels,
                         out_channels=out_channels,
                         n_modes=n_modes,
                         output_scaling_factor=output_scaling_factor,
                         n_layers=n_layers,
                         incremental_n_modes=incremental_n_modes,
                         fno_block_precision=fno_block_precision,
                         channel_mixing=channel_mixing,
                         mlp_dropout=mlp_dropout,
                         mlp_expansion=mlp_expansion,
                         num_prod=num_prod,
                         non_linearity=non_linearity,
                         stabilizer=stabilizer,
                         norm=norm,
                         ada_in_features=ada_in_features,
                         preactivation=preactivation,
                         fno_skip=fno_skip,
                         mixer_skip=mixer_skip,
                         separable=separable,
                         factorization='tucker', # creates a small temp spectral conv just to be deleted
                         rank=0.,
                         SpectralConv=SpectralConv,
                         joint_factorization=joint_factorization,
                         fixed_rank_modes=fixed_rank_modes,
                         implementation=implementation,
                         decomposition_kwargs=decomposition_kwargs,
                         fft_norm=fft_norm,
                         kwargs=kwargs
                        )
        del self.convs
        self.convs = SpectralConvFFNO2d(
            in_channels=in_channels, out_channels=out_channels, 
            n_modes=n_modes, n_layers=n_layers, 
            # factorization=factorization, 
            channel_mixing=ffno_channel_mixing, 
            num_prod=num_prod)