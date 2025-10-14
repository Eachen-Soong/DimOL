from .get_parser import BaseModelParser
from models import FNO, LSM_2D, CNO1d, CNO2d, FNO_2D_Original, ProdFNO_2D_Original, FNO_1D_Original, ProdFNO_1D_Original, DimFNO, CRNO2d, FFNO2d


class CROP2DParser(BaseModelParser):
    def __init__(self) -> None:
        super().__init__()
        self.name = 'CROP'
        
    def add_parser_args(self, parser):
        # # # Model Configs # # #
        parser.add_argument('--modes', type=int, default=24) #
        parser.add_argument('--ini_channels', type=int, default=32) #
        parser.add_argument('--N_layers', type=int, default=3) #
        parser.add_argument('--N_res', type=int, default=4) #
        parser.add_argument('--N_res_neck', type=int, default=6) #
        parser.add_argument('--in_out_size', type=int, default=64) #
        parser.add_argument('--latent_size', type=int, default=64) #
        # parser.add_argument('--size_ratio', type=float, default=4/3)
        parser.add_argument('--kernel_size', type=int, default=3) #

        parser.add_argument('--raw_in_channels', type=int, default=1, help='')
        parser.add_argument('--raw_in_consts', type=int, default=0, help='')
        parser.add_argument('--n_dim', type=int, default=2, help='')
        parser.add_argument('--out_channels', type=int, default=1, help='')
        parser.add_argument('--norm', type=str, default='', help='which norm to use') ##
        parser.add_argument('--append_const', type=int, default=1) ##
        # parser.add_argument('--pos_encoding', type=int, default=1) ##
        parser.add_argument('--use_dim', type=int, default=1) ##
        parser.add_argument('--pre_norm', type=int, default=1, help='whether to use pre_norm') ##
        parser.add_argument('--align_final', type=int, default=1, help='whether to use pre_norm') ##
        parser.add_argument('--prediction_dims', type=int, nargs='+', default=[], help='which entries are prediction')
        parser.add_argument('--num_consts', type=int, default=2, help='number of constants used in DimNorm')
        parser.add_argument('--append_dimless', type=int, default=0, help='whether to append_dimless') ##
        return parser
    
    def get_model(self, args):
        in_channels = args.raw_in_channels
        if hasattr(args, 'initial_steps'):
            if args.initial_steps:
                in_channels *= args.initial_steps

        model = CRNO2d(     in_dim      = in_channels,               # Number of input channels.
                            out_dim     = args.out_channels,
                            in_out_size = args.in_out_size,
                            latent_size = args.latent_size,              # Latent Spacial size
                            modes       = args.modes,
                            N_layers    = args.N_layers,                    # Number of (D) and (U) Blocks in the network
                            N_res       = args.N_res,                          # Number of (R) Blocks per level
                            N_res_neck  = args.N_res_neck,
                            ini_channel = args.ini_channels,
                            norm        = args.norm,
                            append_const= args.append_const,
                            use_dim     = args.use_dim,
                            pre_norm    = args.pre_norm,
                            align_final = args.align_final,
                            num_dimless = args.num_consts,
                            num_consts  = args.raw_in_consts,
                            prediction_dims = args.prediction_dims,
                            )
        return model
    
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
        parser.add_argument('--raw_in_consts', type=int, default=0, help='')
        parser.add_argument('--out_channels', type=int, default=1, help='')
        parser.add_argument('--n_dim', type=int, default=1, help='')
        parser.add_argument('--append_const', type=int, default=1) ##
        parser.add_argument('--pos_encoding', type=int, default=1) ##
        parser.add_argument('--hidden_channels', type=int, default=32) #
        parser.add_argument('--lifting_channels', type=int, default=256) #
        parser.add_argument('--projection_channels', type=int, default=64) #
        parser.add_argument('--factorization', type=str, default='') #####
        parser.add_argument('--ffno', type=int, default=0, help='whether to use F-FNO spectral conv') #####
        parser.add_argument('--channel_mixing', type=str, default='', help='') #####
        parser.add_argument('--mixing_layers', type=int, default=2, help='') #####
        parser.add_argument('--rank', type=float, default=0.42, help='the compression rate of tensor') #
        parser.add_argument('--norm', type=str, default='', help='which norm to use') ##
        parser.add_argument('--pre_norm', type=int, default=1, help='whether to use pre_norm') ##
        parser.add_argument('--preactivation', type=int, default=0, help='whether to use preactivation') ##
        parser.add_argument('--prediction_dims', type=int, nargs='+', default=[], help='which entries are prediction')
        parser.add_argument('--num_consts', type=int, default=2, help='number of constants used in DimNorm')
        parser.add_argument('--append_dimless', type=int, default=0, help='whether to append_dimless') ##
        parser.add_argument('--pos_aug_consts', type=int, default=0, help='whether to use pos_aug consts')
        parser.add_argument('--align_final', type=int, default=1, help='whether to align final dimension')
        parser.add_argument('--linear_proj_last', type=int, default=0, help='whether to apply another linear projection to the last dim')
        return parser
    
    def get_model(self, args):
        n_modes=args.n_modes
        num_prod=args.num_prod
        in_channels = args.raw_in_channels
        if hasattr(args, 'initial_steps'):
            if args.initial_steps:
                in_channels *= args.initial_steps
        norm = args.norm
        dim_norm = args.norm == 'dim_norm'
        if not len(args.norm): norm = None
        new_n_modes = [n_modes,] * args.n_dim
        # if not dim_norm:
        #     model = FNO(in_channels=in_channels, in_consts=args.raw_in_consts, out_channels=args.out_channels, n_modes=new_n_modes, hidden_channels=args.hidden_channels, lifting_channels=args.lifting_channels,
        #                     projection_channels=args.projection_channels, n_layers=args.n_layers, factorization=args.factorization, channel_mixing=args.channel_mixing, mixing_layers=args.mixing_layers, 
        #                     rank=args.rank, num_prod=num_prod, norm=norm, preactivation=args.preactivation, positional_encoding=args.pos_encoding)
        # else:
        # append_const = not dim_norm and not args.pos_aug_consts
        append_const = args.append_const
        if args.ffno:
            model = FFNO2d(in_channels=in_channels, in_consts=args.raw_in_consts, append_const=append_const, out_channels=args.out_channels, n_modes=new_n_modes, hidden_channels=args.hidden_channels, lifting_channels=args.lifting_channels,
                            projection_channels=args.projection_channels, n_layers=args.n_layers, factorization=args.factorization, channel_mixing=args.channel_mixing, mixing_layers=args.mixing_layers, 
                            rank=args.rank, num_prod=num_prod, norm=norm, pre_norm=args.pre_norm, num_consts=args.num_consts, align_final=args.align_final, linear_proj_last=args.linear_proj_last,
                            align_prediction_dims=args.prediction_dims, preactivation=args.preactivation, positional_encoding=args.pos_encoding)
        else:
            model = DimFNO(in_channels=in_channels, in_consts=args.raw_in_consts, append_const=append_const, out_channels=args.out_channels, n_modes=new_n_modes, hidden_channels=args.hidden_channels, lifting_channels=args.lifting_channels,
                            projection_channels=args.projection_channels, n_layers=args.n_layers, factorization=args.factorization, channel_mixing=args.channel_mixing, mixing_layers=args.mixing_layers, 
                            rank=args.rank, num_prod=num_prod, norm=norm, pre_norm=args.pre_norm, num_consts=args.num_consts, align_final=args.align_final, linear_proj_last=args.linear_proj_last,
                            align_prediction_dims=args.prediction_dims, preactivation=args.preactivation, positional_encoding=args.pos_encoding)
        return model


class FNO_OriginalParser(BaseModelParser):
    def __init__(self) -> None:
        super().__init__()
        self.name = 'FNO_Original'
        
    def add_parser_args(self, parser):
        # # # Model Configs # # #
        parser.add_argument('--n_modes', type=int, default=21) #
        parser.add_argument('--num_prod', type=int, default=2) #
        # parser.add_argument('--n_layers', type=int, default=4) ##
        parser.add_argument('--raw_in_channels', type=int, default=1, help='')
        parser.add_argument('--out_channels', type=int, default=1, help='')
        parser.add_argument('--n_dim', type=int, default=2, help='')
        parser.add_argument('--pos_encoding', type=int, default=0) ##
        parser.add_argument('--model_pos_encoding', type=int, default=1) ##
        parser.add_argument('--hidden_channels', type=int, default=32) #

        return parser

    def get_model(self, args):
        n_modes=args.n_modes
        num_prod=args.num_prod
        in_channels = args.raw_in_channels

        width = args.hidden_channels

        if args.n_dim == 2:
            if num_prod:
                model = ProdFNO_2D_Original(in_dim=in_channels, out_dim=args.out_channels,
                    modes1=n_modes, modes2=n_modes, width=width, num_prod=num_prod, use_position=args.model_pos_encoding)
                
            else:
                model = FNO_2D_Original(in_dim=in_channels, out_dim=args.out_channels,
                    modes1=n_modes, modes2=n_modes, width=width, use_position=args.model_pos_encoding)
        elif args.n_dim == 1:
            if num_prod:
                model = ProdFNO_1D_Original(in_dim=in_channels, out_dim=args.out_channels,
                    modes=n_modes, width=width, num_prod=num_prod, use_position=args.model_pos_encoding, )
                
            else:
                model = FNO_1D_Original(in_dim=in_channels, out_dim=args.out_channels,
                    modes=n_modes, width=width, use_position=args.model_pos_encoding)

        else:
            assert False, f"Unsupported Input Shape: {args.n_dim}"

        return model


class LSMParser(BaseModelParser):
    def __init__(self) -> None:
        super().__init__()
        self.name = 'LSM'

    def add_parser_args(self, parser):
        parser.add_argument('--raw_in_channels', default=1, type=int, help='input data dimension')
        parser.add_argument('--raw_in_consts', type=int, default=0, help='')
        parser.add_argument('--out_channels', default=1, type=int, help='output data dimension')
        parser.add_argument('--pos_encoding', type=int, default=1) ##
        parser.add_argument('--d-model', default=64, type=int, help='channels of hidden variates')
        parser.add_argument('--num-basis', default=12, type=int, help='number of basis operators')
        parser.add_argument('--num-token', default=4, type=int, help='number of latent tokens')
        parser.add_argument('--patch-size', default='6,6', type=str, help='patch size of different dimensions')
        parser.add_argument('--padding', default='11,11', type=str, help='padding size of different dimensions')
        parser.add_argument('--norm', default='layer_norm', type=str, help='which norm to use')
        parser.add_argument('--append_const', type=int, default=1) ##
        parser.add_argument('--use_dim', type=int, default=1) ##
        parser.add_argument('--pre_norm', type=int, default=1, help='whether to use pre_norm') ##
        parser.add_argument('--align_final', type=int, default=1, help='whether to use pre_norm') ##
        parser.add_argument('--prediction_dims', type=int, nargs='+', default=[], help='which entries are prediction')
        parser.add_argument('--num_consts', type=int, default=2, help='number of constants used in DimNorm')
        parser.add_argument('--append_dimless', type=int, default=0, help='whether to append_dimless') ##
        return parser

    def get_model(self, args):
        in_channels = args.raw_in_channels
        if hasattr(args, 'initial_steps'):
            if args.initial_steps:
                in_channels *= args.initial_steps
        out_channels = args.out_channels
        args.norm = None
        args.n_dim = 2
        width = args.d_model
        num_token = args.num_token
        num_basis = args.num_basis
        patch_size = [int(x) for x in args.patch_size.split(',')]
        padding = [int(x) for x in args.padding.split(',')]

        model = LSM_2D( in_dim=in_channels, out_dim=out_channels, d_model=width,
                        num_token=num_token, num_basis=num_basis, patch_size=patch_size, padding=padding, norm=args.norm,
                        append_const=args.append_const, use_dim=args.use_dim, pre_norm=args.pre_norm,
                        align_final=args.align_final, num_dimless=args.num_consts, num_consts=args.raw_in_consts,
                        prediction_dims=args.prediction_dims,
                        )
        return model


class CNOParser(BaseModelParser):
    def __init__(self) -> None:
        super().__init__()
        self.name = 'CNO'
        
    def add_parser_args(self, parser):
        # # # Model Configs # # #
        parser.add_argument('--raw_in_channels', type=int, default=1, help='')
        parser.add_argument('--out_channels', type=int, default=0, help='')
        parser.add_argument('--n_dim', type=int, default=2, help='')
        parser.add_argument('--spatial_size', type=int, default=128, help='Input and Output spatial size (required )') ##
        parser.add_argument('--n_layers', type=int, default=4, help='Number of (D) or (U) blocks in the network') ##
        parser.add_argument('--n_res', type=int, default=4, help='Number of (R) blocks per level (except the neck)') ##
        parser.add_argument('--n_res_neck', type=int, default=16, help='Number of (R) blocks in the neck') ##
        parser.add_argument('--channel_multiplier', type=int, default=16, help='How the number of channels evolve?') ##
        parser.add_argument('--use_bn', type=int, default=0, help='whether to use batch_normalization') ##

        parser.add_argument('--pos_encoding', type=int, default=0) ##
        parser.add_argument('--channel_mixing', type=str, default='', help='') #####
        parser.add_argument('--num_prod', type=int, default=2) #
        parser.add_argument('--mixing_layers', type=int, default=2, help='') #####

        return parser

    def get_model(self, args):
        in_channels = args.raw_in_channels
        if args.pos_encoding:
            in_channels += args.n_dim
        out_channels = args.out_channels
        if args.out_channels == 0:
            out_channels = args.raw_in_channels

        if args.n_dim == 1:
            model = CNO1d(  in_dim = in_channels,
                            out_dim = out_channels,
                            size = args.spatial_size,
                            N_layers = args.n_layers,
                            N_res = args.n_res,
                            N_res_neck = args.n_res_neck,
                            channel_multiplier = args.channel_multiplier,
                            use_bn = bool(args.use_bn))
        elif args.n_dim == 2:
            model = CNO2d(  in_dim = in_channels,
                            out_dim = out_channels,
                            size = args.spatial_size,
                            N_layers = args.n_layers,
                            N_res = args.n_res,
                            N_res_neck = args.n_res_neck,
                            channel_multiplier = args.channel_multiplier,
                            use_bn = bool(args.use_bn))
        return model

