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
    

