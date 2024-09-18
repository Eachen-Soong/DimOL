from get_parser import BaseModelParser
from neuralop1.models import FNO, LSM_2D


class FNOParser(BaseModelParser):
    def __init__(self) -> None:
        super().__init__()
        self.name = 'FNO'
        
    def add_parser_args(self, parser):
        # # # Model Configs # # #
        parser.add_argument('--n_modes', type=int, default=21) #
        parser.add_argument('--num_prod', type=int, default=2) #
        parser.add_argument('--n_layers', type=int, default=4) ##
        parser.add_argument('--raw_in_channels', type=int, default=1, help='')
        parser.add_argument('--pos_encoding', type=int, default=1) ##
        parser.add_argument('--hidden_channels', type=int, default=32) #
        parser.add_argument('--lifting_channels', type=int, default=256) #
        parser.add_argument('--projection_channels', type=int, default=64) #
        parser.add_argument('--factorization', type=str, default='') #####
        parser.add_argument('--channel_mixing', type=str, default='', help='') #####
        parser.add_argument('--mixing_layers', type=int, default=2, help='') #####
        parser.add_argument('--rank', type=float, default=0.42, help='the compression rate of tensor') #

        return parser

    def get_model(self, args):
        n_modes=args.n_modes
        num_prod=args.num_prod
        in_channels = args.raw_in_channels
        if args.pos_encoding:
            in_channels += 2
        model = FNO(in_channels=in_channels, n_modes=(n_modes,), hidden_channels=args.hidden_channels, lifting_channels=args.lifting_channels,
                    projection_channels=args.projection_channels, n_layers=args.n_layers, factorization=args.factorization, channel_mixing=args.channel_mixing, mixing_layers=args.mixing_layers, rank=args.rank, num_prod=num_prod)
        return model


class LSMParser(BaseModelParser):
    def __init__(self) -> None:
        super().__init__()
        self.name = 'LSM'

    def add_parser_args(self, parser):
        parser.add_argument('--in_dim', default=1, type=int, help='input data dimension')
        parser.add_argument('--out_dim', default=1, type=int, help='output data dimension')
        parser.add_argument('--h', default=421, type=int, help='input data height')
        parser.add_argument('--w', default=421, type=int, help='input data width')
        parser.add_argument('--T-in', default=10, type=int,
                            help='input data time points (only for temporal related experiments)')
        parser.add_argument('--T-out', default=10, type=int,
                            help='predict data time points (only for temporal related experiments)')
        parser.add_argument('--pos_encoding', type=int, default=1) ##
        parser.add_argument('--d-model', default=64, type=int, help='channels of hidden variates')
        parser.add_argument('--num-basis', default=12, type=int, help='number of basis operators')
        parser.add_argument('--num-token', default=4, type=int, help='number of latent tokens')
        parser.add_argument('--patch-size', default='6,6', type=str, help='patch size of different dimensions')
        parser.add_argument('--padding', default='11,11', type=str, help='padding size of different dimensions')
        parser.add_argument('--channel_mixing', type=str, default='', help='') #####
        parser.add_argument('--num_prod', type=int, default=2) #

        return parser

    def get_model(self, args):
        in_channels = args.in_dim
        if args.pos_encoding:
            in_channels += 2
        out_channels = args.out_dim
        width = args.d_model
        num_token = args.num_token
        num_basis = args.num_basis
        patch_size = [int(x) for x in args.patch_size.split(',')]
        padding = [int(x) for x in args.padding.split(',')]

        model = LSM_2D(in_dim=in_channels, out_dim=out_channels, d_model=width,
                            num_token=num_token, num_basis=num_basis, patch_size=patch_size, padding=padding, channel_mixing=args.channel_mixing, num_prod=args.num_prod)
        return model