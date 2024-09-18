from get_parser import BaseDataParser
from data.neuralop_datasets import load_burgers_mat, load_darcy_mat, load_autoregressive_traintestsplit_v3

class BurgersParser(BaseDataParser):
    def __init__(self) -> None:
        super().__init__()
        self.name = 'Burgers'

    def add_parser_args(self, parser):
        super().add_parser_args(parser)
        return
    
    def get_data(self, args):
        train_loader, val_loader, _ = load_burgers_mat(
        data_path=args.data_path, n_train=args.n_train, n_test=args.n_test, batch_size=args.batch_size, test_batch_size = args.batch_size, 
        train_ssr=args.train_subsample_rate, test_ssrs=args.test_subsample_rate, 
        positional_encoding=args.pos_encoding
        )
        return train_loader, val_loader


class DarcyParser(BaseDataParser):
    def __init__(self) -> None:
        super().__init__()
        self.name = 'Darcy'

    def add_parser_args(self, parser):
        super().add_parser_args(parser)
        return
    
    def get_data(self, args):
        train_loader, val_loader, _ = load_darcy_mat(
            data_path=args.data_path, n_train=args.n_train, n_test=args.n_test, batch_size=args.batch_size, test_batch_size = args.batch_size, 
            train_ssr=args.train_subsample_rate, test_ssrs=args.test_subsample_rate, 
            positional_encoding=args.pos_encoding,
            predict_feature=args.predict_feature,
        )
        return train_loader, val_loader


class TorusLiParser(BaseDataParser):
    def __init__(self) -> None:
        super().__init__()
        self.name = 'TorusLi'

    def add_parser_args(self, parser):
        super().add_parser_args(parser)
        parser.add_argument('--time_step', type=int, default=1, help='subsample rate of time')
        parser.add_argument('--predict_feature', type=str, default='u')
        return
    
    def get_data(self, args):
        train_loader, val_loader = load_autoregressive_traintestsplit_v3(
            data_path=args.data_path, n_train=args.n_train, n_test=args.n_test, batch_size=args.batch_size, test_batch_size = args.batch_size, 
            train_ssr=args.train_subsample_rate, test_ssrs=args.test_subsample_rate, time_step=args.time_step,
            positional_encoding=args.pos_encoding,
            predict_feature=args.predict_feature,
        )
        return train_loader, val_loader
    

class TorusVisForceParser(BaseDataParser):
    def __init__(self) -> None:
        super().__init__()
        self.name = 'TorusVisForce'

    def add_parser_args(self, parser):
        super().add_parser_args(parser)
        parser.add_argument('--time_step', type=int, default=1, help='subsample rate of time')
        parser.add_argument('--predict_feature', type=str, default='u')
        return
    
    def get_data(self, args):
        train_loader, val_loader = load_autoregressive_traintestsplit_v3(
            data_path=args.data_path, n_train=args.n_train, n_test=args.n_test, batch_size=args.batch_size, test_batch_size = args.batch_size, 
            train_ssr=args.train_subsample_rate, test_ssrs=args.test_subsample_rate, time_step=args.time_step,
            positional_encoding=args.pos_encoding,
            predict_feature=args.predict_feature,
        )
        return train_loader, val_loader
