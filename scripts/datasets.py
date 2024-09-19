from get_parser import BaseDataParser
from data.neuralop_datasets import load_burgers_mat, load_darcy_mat, load_autoregressive_traintestsplit_v3
import types
import copy

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
    
def gen_similar_dataloaders_dt_divided_p(origin_loader, scaling_ps, batch_size=0):

    def new_get_item(self, index):
        # returns: {'x', 'y', other features}
        p = self.scaling_p
        p_int = int(p)
        origin_item =  self.get_item(index, p_int)
        new_item = {}
        new_item['x'] = origin_item['x'] * p
        new_item['y'] = origin_item['y'] * p
        new_item['mu'] = origin_item['mu'] * p
        new_item['f'] = origin_item['f'] * (p * p)
        return new_item
    
    # def new_get_len(self):
    #     return self.n_samples * self.n_ticks

    sim_loaders = {}
    n_scale_coeff = to_torch_tensor(scaling_ps).shape[0]
    # shallow copy to share the same raw data
    sim_dataset = copy.copy(origin_loader.dataset)
    sim_dataset.__getitem__ = types.MethodType(new_get_item, sim_dataset)
    # sim_dataset.__len__ = types.MethodType(new_get_len, sim_dataset)
    sim_dataset.scaling_p = 1.
    sim_dataset.scaling_k = 1.
    sim_dataset.is_k_integer = False
    if batch_size==0:
        batch_size=origin_loader.batch_size
    for i in range(n_scale_coeff):
        new_dataset = copy.copy(sim_dataset)
        new_dataset.scaling_p = scaling_ps[i]
        sim_dataset.scaling_k = 1 / new_dataset.scaling_p
        total_time_number = new_dataset.time_step + new_dataset.n_ticks
        new_dataset.time_step = new_dataset.time_step * new_dataset.scaling_p
        new_dataset.n_ticks = total_time_number - new_dataset.time_step
        new_loader = ns_contextual_loader(new_dataset, batch_size=batch_size, shuffle=False, num_workers=0,
                                          append_positional_encoding=True
                                          )
        sim_loaders[f"dt/=p_{scaling_ps[i]}"] = new_loader

    return sim_loaders

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
        if args.simaug is not None:
            
        val_loader['']
        return train_loader, val_loader
