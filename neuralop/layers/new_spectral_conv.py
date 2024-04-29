from typing import List
from .spectral_convolution import *
from .quad_prod import *
from .mlp import MLP
from einops import rearrange

"""
    2 jobs: introducing convolutions between channels and use F-FNO
"""

class SpectralConvFFNO2d(SpectralConv):
    """ 2D Spectral Conv, but destructed the sum of 2* Spectral Conv 1d (equivalent to F-FNO)
        Here we introduce more options, 
        say, you can directly add the outcomes up like the original F-FNO,
        or you can introduce some product terms
        Here we define a channel mixing function, which would mix the channel of the outputs.
    """
    def __init__(self, in_channels, out_channels, n_modes, n_layers=1, channel_mixing='add', num_prod=2, **kwargs):
        # for super(SpectralConv), n_layers=2 * n_layers since we need 2* Spectral Conv 1d
        super().__init__(in_channels=in_channels, out_channels=out_channels, factorization=None, n_modes=n_modes[0], n_layers=2 * n_layers, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_modes = n_modes
        assert len(n_modes) == 2, f"len(n_modes)=={len(n_modes)}, != 2"
        assert n_modes[0] == n_modes[1], f"Currently only supports the same resolution along axises"
        self.n_mode = n_modes[0]
        self.n_layers = n_layers
        self.num_prod = num_prod
        self.merge_mixer = None
        self.linear = None
        if channel_mixing=='prod':
            self.linear = MLP(in_channels=2*out_channels+num_prod, out_channels=out_channels, n_layers=1, n_dim=2)
            def merge_and_mix(a,b):
                x = torch.cat((a, b), dim=1)
                prods = torch.stack([x[:, i, ...] * x[:, out_channels + i, ...] for i in torch.arange(0, self.num_prod, 1, dtype=torch.int)], dim=1)
                x = torch.cat((x, prods), dim=1)
                x = self.linear(x)
                return x
            self.merge_mixer = merge_and_mix
        elif channel_mixing=='linear':
            self.linear = MLP(in_channels=2*out_channels, out_channels=out_channels, n_layers=1, n_dim=2)
            def merge_and_mix(a,b):
                x = torch.cat((a, b), dim=1)
                x = self.linear(x)
                return x
            self.merge_mixer = merge_and_mix
        elif channel_mixing=='add':
            self.merge_mixer = torch.add
        else:
            assert False, 'Unsupported Channel Mixing!'


    def forward(self, x: torch.Tensor, indices=0, **kwargs):
        B, I, M, N = x.shape

        # # # Dimesion Y # # #
        x_fty = torch.fft.rfft(x, dim=-1, norm='ortho')
        # x_ft.shape == [batch_size, in_dim, grid_size, grid_size // 2 + 1]

        out_ft = x_fty.new_zeros(B, I, M, N // 2 + 1)
        # out_ft.shape == [batch_size, in_dim, grid_size, grid_size // 2 + 1, 2]

        slices = (
            slice(None),  # Equivalent to: [:,
            slice(None),  # ............... :,
            slice(None),  # ............... :
            slice(self.half_n_modes[0]),  # :half_n_modes[0]]
        )
        if x_fty.dtype == torch.complex32:
            # if x is half precision, run a specialized einsum
            out_ft[slices] = einsum_complexhalf(
                    "bixy,ioy->boxy",
                    x_fty[slices], self._get_weight(2 * indices).to_tensor()
                )
        else:
            out_ft[slices] = tl.einsum(
                    "bixy,ioy->boxy",
                    x_fty[slices], self._get_weight(2 * indices).to_tensor()
                )
                
        xy = torch.fft.irfft(out_ft, n=N, dim=-1, norm='ortho')
        # x.shape == [batch_size, in_dim, grid_size, grid_size]

        # # # Dimesion X # # #
        x_ftx = torch.fft.rfft(x, dim=-2, norm='ortho')
        # x_ft.shape == [batch_size, in_dim, grid_size // 2 + 1, grid_size]

        out_ft = x_ftx.new_zeros(B, I, M // 2 + 1, N)
        # out_ft.shape == [batch_size, in_dim, grid_size // 2 + 1, grid_size, 2]

        slices = (
            slice(None),  # Equivalent to: [:,
            slice(None),  # ............... :,
            slice(self.half_n_modes[0]),  # :half_n_modes[0]]
            slice(None),  # ............... :
        )
        if x_ftx.dtype == torch.complex32:
            # if x is half precision, run a specialized einsum
            out_ft[slices] = einsum_complexhalf(
                    "bixy,iox->boxy",
                    x_ftx[slices], self._get_weight(2 * indices + 1).to_tensor()
                )
        else:
            out_ft[slices] = tl.einsum(
                    "bixy,iox->boxy",
                    x_ftx[slices], self._get_weight(2 * indices + 1).to_tensor()
                )

        xx = torch.fft.irfft(out_ft, n=M, dim=-2, norm='ortho')

        # # # merge and Channel mixing # # #
        x = self.merge_mixer(xx, xy)

        return x


class SpectralConvProd(SpectralConv):
    """ 2D Spectral Conv, but destructed the sum of 2* Spectral Conv 1d (equivalent to F-FNO)
        Here we introduce more options, 
        say, you can directly add the outcomes up like the original F-FNO,
        or you can introduce some product terms
        Here we define a channel mixing function, which would mix the channel of the outputs.
    """
    def __init__(self, in_channels, out_channels, n_modes, n_prods=2, incremental_n_modes=None, bias=True, n_layers=1, separable=False, output_scaling_factor: int | float | List[int | float] | None = None, fno_block_precision="full", rank=0.5, factorization=None, implementation="reconstructed", fixed_rank_modes=False, joint_factorization=False, decomposition_kwargs: dict | None = None, init_std="auto", fft_norm="backward", device=None, dtype=None):
        super().__init__(in_channels, out_channels, n_modes, incremental_n_modes, bias, n_layers, separable, output_scaling_factor, fno_block_precision, rank, factorization, implementation, fixed_rank_modes, joint_factorization, decomposition_kwargs, init_std, fft_norm, device, dtype)
        self.out_channels = out_channels
        self.n_prods = n_prods
        self.prod_path = ProductPath(in_channels, self.n_prods)
        half_total_n_modes = self.half_n_modes
        tensor_kwargs = decomposition_kwargs if decomposition_kwargs is not None else {}
        del self.weight

        if init_std == "auto":
            init_std = (2 / (in_channels + out_channels))**0.5
        else:
            init_std = init_std

        if isinstance(fixed_rank_modes, bool):
            if fixed_rank_modes:
                # If bool, keep the number of layers fixed
                fixed_rank_modes = [0]
            else:
                fixed_rank_modes = None

        # Make sure we are using a Complex Factorized Tensor to parametrize the conv
        if factorization is None:
            factorization = "Dense"  # No factorization
        if not factorization.lower().startswith("complex"):
            factorization = f"Complex{factorization}"

        if separable:
            if in_channels != out_channels:
                raise ValueError(
                    "To use separable Fourier Conv, in_channels must be equal "
                    f"to out_channels, but got in_channels={in_channels} and "
                    f"out_channels={out_channels}",
                )
            weight_shape = (in_channels, *half_total_n_modes)
        else:
            weight_shape = (in_channels, out_channels-self.n_prods, *half_total_n_modes)
        self.separable = separable

        self.n_weights_per_layer = 2 ** (self.order - 1)
        tensor_kwargs = decomposition_kwargs if decomposition_kwargs is not None else {}
        if joint_factorization:
            self.weight = FactorizedTensor.new(
                (self.n_weights_per_layer * n_layers, *weight_shape),
                rank=self.rank,
                factorization=factorization,
                fixed_rank_modes=fixed_rank_modes,
                **tensor_kwargs,
            )
            self.weight.normal_(0, init_std)
        else:
            self.weight = nn.ModuleList(
                [
                    FactorizedTensor.new(
                        weight_shape,
                        rank=self.rank,
                        factorization=factorization,
                        fixed_rank_modes=fixed_rank_modes,
                        **tensor_kwargs,
                    )
                    for _ in range(self.n_weights_per_layer * n_layers)
                ]
            )
            for w in self.weight:
                w.normal_(0, init_std)

    def forward(
        self, x: torch.Tensor, indices=0, output_shape: Optional[Tuple[int]] = None
    ):
        """Generic forward pass for the Factorized Spectral Conv

        Parameters
        ----------
        x : torch.Tensor
            input activation of size (batch_size, channels, d1, ..., dN)
        indices : int, default is 0
            if joint_factorization, index of the layers for n_layers > 1

        Returns
        -------
        tensorized_spectral_conv(x)
        """
        batchsize, channels, *mode_sizes = x.shape

        fft_size = list(mode_sizes)
        fft_size[-1] = fft_size[-1] // 2 + 1  # Redundant last coefficient

        # Compute Fourier coeffcients
        fft_dims = list(range(-self.order, 0))

        if self.fno_block_precision == "half":
            x = x.half()

        x = torch.fft.rfftn(x, norm=self.fft_norm, dim=fft_dims)

        if self.fno_block_precision == "mixed":
            # if 'mixed', the above fft runs in full precision, but the
            # following operations run at half precision
            x = x.chalf()

        if self.fno_block_precision in ["half", "mixed"]:
            out_fft = torch.zeros(
                [batchsize, self.out_channels-self.n_prods, *fft_size],
                device=x.device,
                dtype=torch.chalf,
            )
        else:
            out_fft = torch.zeros(
                [batchsize, self.out_channels-self.n_prods, *fft_size],
                device=x.device,
                dtype=torch.cfloat,
            )

        # We contract all corners of the Fourier coefs
        # Except for the last mode: there, we take all coefs as redundant modes
        # were already removed
        mode_indexing = [((None, m), (-m, None)) for m in self.half_n_modes[:-1]] + [
            ((None, self.half_n_modes[-1]),)
        ]

        for i, boundaries in enumerate(itertools.product(*mode_indexing)):
            # Keep all modes for first 2 modes (batch-size and channels)
            idx_tuple = [slice(None), slice(None)] + [slice(*b) for b in boundaries]

            # For 2D: [:, :, :height, :width] and [:, :, -height:, width]
            out_fft[idx_tuple] = self._contract(
                x[idx_tuple],
                self._get_weight(self.n_weights_per_layer * indices + i),
                separable=self.separable,
            )

        if self.output_scaling_factor is not None and output_shape is None:
            mode_sizes = tuple(
                [
                    round(s * r)
                    for (s, r) in zip(mode_sizes, self.output_scaling_factor[indices])
                ]
            )

        if output_shape is not None:
            mode_sizes = output_shape

        out_fft_prod = self.prod_path(out_fft)
        out_fft = torch.cat((out_fft, out_fft_prod), dim=1)

        x = torch.fft.irfftn(out_fft, s=mode_sizes, norm=self.fft_norm)

        if self.bias is not None:
            x = x + self.bias[indices, ...]

        return x


class SpectralConvAttn2d(BaseSpectralConv):
    """
        2D spectral convolution with an attention block as the kernel.
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        n_modes,
        bias=True,
        n_layers=1,
        fno_block_precision="full",
        rank=0.5,
        factorization=None,                                                              
        implementation="reconstructed",
        fixed_rank_modes=False,
        joint_factorization=False,
        decomposition_kwargs: Optional[dict] = None,
        init_std="auto",
        fft_norm="backward",
        device=None,
        dtype=None,
    ):
        super().__init__(dtype=dtype, device=device)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.joint_factorization = joint_factorization

        # We index quadrands only
        # n_modes is the total number of modes kept along each dimension
        # half_n_modes is half of that except in the last mode, correponding to
        # the number of modes to keep in *each* quadrant for each dim
        if isinstance(n_modes, int):
            n_modes = [n_modes]
        self.n_modes = n_modes
        self.order = len(n_modes)

        half_total_n_modes = [m // 2 for m in n_modes]
        self.half_total_n_modes = half_total_n_modes

        # We use half_total_n_modes to build the full weights
        # During training we can adjust incremental_n_modes which will also
        # update half_n_modes
        # So that we can train on a smaller part of the Fourier modes and total
        # weights

        self.fno_block_precision = fno_block_precision
        self.rank = rank
        self.factorization = factorization
        self.n_layers = n_layers
        self.implementation = implementation


        if init_std == "auto":
            init_std = (2 / (in_channels + out_channels))**0.5
        else:
            init_std = init_std

        if isinstance(fixed_rank_modes, bool):
            if fixed_rank_modes:
                # If bool, keep the number of layers fixed
                fixed_rank_modes = [0]
            else:
                fixed_rank_modes = None
        self.fft_norm = fft_norm

        # Make sure we are using a Complex Factorized Tensor to parametrize the
        # conv
        if factorization is None:
            factorization = "Dense"  # No factorization
        if not factorization.lower().startswith("complex"):
            factorization = f"Complex{factorization}"


        weight_shape = (in_channels, out_channels, *half_total_n_modes)

        self.n_weights_per_layer = 2 ** (self.order - 1)
        tensor_kwargs = decomposition_kwargs if decomposition_kwargs is not None else {}
        if joint_factorization:
            self.weight = FactorizedTensor.new(
                (self.n_weights_per_layer * n_layers, *weight_shape),
                rank=self.rank,
                factorization=factorization,
                fixed_rank_modes=fixed_rank_modes,
                **tensor_kwargs,
            )
            self.weight.normal_(0, init_std)
        else:
            self.weight = nn.ModuleList(
                [
                    FactorizedTensor.new(
                        weight_shape,
                        rank=self.rank,
                        factorization=factorization,
                        fixed_rank_modes=fixed_rank_modes,
                        **tensor_kwargs,
                    )
                    for _ in range(self.n_weights_per_layer * n_layers)
                ]
            )
            for w in self.weight:
                w.normal_(0, init_std)
        self._contract = get_contract_fun(
            self.weight[0], implementation=implementation
        )

        if bias:
            self.bias = nn.Parameter(
                init_std
                * torch.randn(*((n_layers, self.out_channels) + (1,) * self.order))
            )
        else:
            self.bias = None

    def __init__(self, in_channels, out_channels, n_modes, n_layers=1, **kwargs):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_modes = n_modes
        self.softmax = nn.Softmax(dim=-1)
        self.Wq = nn.Parameter()

    def self_attn(self, q, k, v):
        # q,k,v: B H L C/H
        attn = self.softmax(torch.einsum("bhlc,bhsc->bhls", q, k))
        return torch.einsum("bhls,bhsc->bhlc", attn, v)


    def forward(self, x: torch.Tensor, indices=0, **kwargs):
        B, I, M, N = x.shape

        # # # Dimesion Y # # #
        x_fty = torch.fft.rfft(x, dim=-1, norm='ortho')
        # x_ft.shape == [batch_size, in_dim, grid_size, grid_size // 2 + 1]
        


        out_ft = x_fty.new_zeros(B, I, M, N // 2 + 1)
        # out_ft.shape == [batch_size, in_dim, grid_size, grid_size // 2 + 1, 2]

        slices = (
            slice(None),  # Equivalent to: [:,
            slice(None),  # ............... :,
            slice(None),  # ............... :
            slice(self.half_n_modes[0]),  # :half_n_modes[0]]
        )
        if x_fty.dtype == torch.complex32:
            # if x is half precision, run a specialized einsum
            out_ft[slices] = einsum_complexhalf(
                    "bixy,ioy->boxy",
                    x_fty[slices], self._get_weight(2 * indices).to_tensor()
                )
        else:
            out_ft[slices] = tl.einsum(
                    "bixy,ioy->boxy",
                    x_fty[slices], self._get_weight(2 * indices).to_tensor()
                )
                
        xy = torch.fft.irfft(out_ft, n=N, dim=-1, norm='ortho')
        # x.shape == [batch_size, in_dim, grid_size, grid_size]

        # # # Dimesion X # # #
        x_ftx = torch.fft.rfft(x, dim=-2, norm='ortho')
        # x_ft.shape == [batch_size, in_dim, grid_size // 2 + 1, grid_size]

        out_ft = x_ftx.new_zeros(B, I, M // 2 + 1, N)
        # out_ft.shape == [batch_size, in_dim, grid_size // 2 + 1, grid_size, 2]

        slices = (
            slice(None),  # Equivalent to: [:,
            slice(None),  # ............... :,
            slice(self.half_n_modes[0]),  # :half_n_modes[0]]
            slice(None),  # ............... :
        )
        if x_ftx.dtype == torch.complex32:
            # if x is half precision, run a specialized einsum
            out_ft[slices] = einsum_complexhalf(
                    "bixy,iox->boxy",
                    x_ftx[slices], self._get_weight(2 * indices + 1).to_tensor()
                )
        else:
            out_ft[slices] = tl.einsum(
                    "bixy,iox->boxy",
                    x_ftx[slices], self._get_weight(2 * indices + 1).to_tensor()
                )

        xx = torch.fft.irfft(out_ft, n=M, dim=-2, norm='ortho')

        # # # merge and Channel mixing # # #
        x = self.merge_mixer(xx, xy)

        return x


# class SpectralConvAttn2d(SpectralConv):
#     """ 2D Spectral Conv, but destructed the sum of 2* Spectral Conv 1d (equivalent to F-FNO)
#         Here we introduce more options, 
#         say, you can directly add the outcomes up like the original F-FNO,
#         or you can introduce some product terms
#         Here we define a channel mixing function, which would mix the channel of the outputs.
#     """
#     def __init__(self, in_channels, out_channels, n_modes, n_layers=1, channel_mixing='add', num_prod=2, **kwargs):
#         # for super(SpectralConv), n_layers=2 * n_layers since we need 2* Spectral Conv 1d
#         super().__init__(in_channels=in_channels, out_channels=out_channels, factorization=None, n_modes=n_modes[0], n_layers=2 * n_layers, **kwargs)
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.n_modes = n_modes
#         assert len(n_modes) == 2, f"len(n_modes)=={len(n_modes)}, != 2"
#         assert n_modes[0] == n_modes[1], f"Currently only supports the same resolution along axises"
#         self.n_mode = n_modes[0]
#         self.n_layers = n_layers
#         self.num_prod = num_prod
#         self.merge_mixer = None
#         self.linear = None
#         if channel_mixing=='prod':
#             self.linear = MLP(in_channels=2*out_channels+num_prod, out_channels=out_channels, n_layers=1, n_dim=2)
#             def merge_and_mix(a,b):
#                 x = torch.cat((a, b), dim=1)
#                 prods = torch.stack([x[:, i, ...] * x[:, out_channels + i, ...] for i in torch.arange(0, self.num_prod, 1, dtype=torch.int)], dim=1)
#                 x = torch.cat((x, prods), dim=1)
#                 x = self.linear(x)
#                 return x
#             self.merge_mixer = merge_and_mix
#         elif channel_mixing=='linear':
#             self.linear = MLP(in_channels=2*out_channels, out_channels=out_channels, n_layers=1, n_dim=2)
#             def merge_and_mix(a,b):
#                 x = torch.cat((a, b), dim=1)
#                 x = self.linear(x)
#                 return x
#             self.merge_mixer = merge_and_mix
#         elif channel_mixing=='add':
#             self.merge_mixer = torch.add
#         else:
#             assert False, 'Unsupported Channel Mixing!'


#     def forward(self, x: torch.Tensor, indices=0, **kwargs):
#         B, I, M, N = x.shape

#         # # # Dimesion Y # # #
#         x_fty = torch.fft.rfft(x, dim=-1, norm='ortho')
#         # x_ft.shape == [batch_size, in_dim, grid_size, grid_size // 2 + 1]

#         out_ft = x_fty.new_zeros(B, I, M, N // 2 + 1)
#         # out_ft.shape == [batch_size, in_dim, grid_size, grid_size // 2 + 1, 2]

#         slices = (
#             slice(None),  # Equivalent to: [:,
#             slice(None),  # ............... :,
#             slice(None),  # ............... :
#             slice(self.half_n_modes[0]),  # :half_n_modes[0]]
#         )
#         if x_fty.dtype == torch.complex32:
#             # if x is half precision, run a specialized einsum
#             out_ft[slices] = einsum_complexhalf(
#                     "bixy,ioy->boxy",
#                     x_fty[slices], self._get_weight(2 * indices).to_tensor()
#                 )
#         else:
#             out_ft[slices] = tl.einsum(
#                     "bixy,ioy->boxy",
#                     x_fty[slices], self._get_weight(2 * indices).to_tensor()
#                 )
                
#         xy = torch.fft.irfft(out_ft, n=N, dim=-1, norm='ortho')
#         # x.shape == [batch_size, in_dim, grid_size, grid_size]

#         # # # Dimesion X # # #
#         x_ftx = torch.fft.rfft(x, dim=-2, norm='ortho')
#         # x_ft.shape == [batch_size, in_dim, grid_size // 2 + 1, grid_size]

#         out_ft = x_ftx.new_zeros(B, I, M // 2 + 1, N)
#         # out_ft.shape == [batch_size, in_dim, grid_size // 2 + 1, grid_size, 2]

#         slices = (
#             slice(None),  # Equivalent to: [:,
#             slice(None),  # ............... :,
#             slice(self.half_n_modes[0]),  # :half_n_modes[0]]
#             slice(None),  # ............... :
#         )
#         if x_ftx.dtype == torch.complex32:
#             # if x is half precision, run a specialized einsum
#             out_ft[slices] = einsum_complexhalf(
#                     "bixy,iox->boxy",
#                     x_ftx[slices], self._get_weight(2 * indices + 1).to_tensor()
#                 )
#         else:
#             out_ft[slices] = tl.einsum(
#                     "bixy,iox->boxy",
#                     x_ftx[slices], self._get_weight(2 * indices + 1).to_tensor()
#                 )

#         xx = torch.fft.irfft(out_ft, n=M, dim=-2, norm='ortho')

#         # # # merge and Channel mixing # # #
#         x = self.merge_mixer(xx, xy)

#         return x