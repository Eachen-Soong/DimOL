from typing import Any
import lightning as L
import torch


class InitialStepsModule(L.LightningModule):
    """
    Basic Pytorch-Lightning module for AI for Science Tasks.
    """
    def __init__(self, model, optimizer, scheduler, train_loss, metric_dict:dict, average_over_batch=True, initial_steps=10, t_train=101) -> None:
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loss = train_loss
        self.metric_dict = metric_dict
        self.average_over_batch = average_over_batch
        self.initial_steps = initial_steps
        self.t_train = t_train
        self.average_over_time = True

    def configure_optimizers(self):
        return {'optimizer':self.optimizer, 'lr_scheduler': self.scheduler}

    def training_step(self, batch, batch_idx, *args, **kwargs):
        # pred = self.model(**batch)
        # has_nan = torch.isnan(pred).any()
        # if has_nan:
        #     import pdb; pdb.set_trace()
        yy = batch['y']
        xx = batch['x']
        try:
            consts = batch['consts']
        except:
            consts = None
        pred = yy[..., :self.initial_steps]


        inp_shape = list(xx.shape)
        inp_shape[1] *= inp_shape[-1]
        inp_shape.pop(-1)

        loss = 0.

        for t in range(self.initial_steps, self.t_train):
            inp = xx.reshape(inp_shape)
            # Extract target at current time step
            y = yy[..., t]
            inputs = {'x': inp, 'y': y}
            if consts is not None:
                inputs.update({'consts': consts})
            im = self.model(**inputs)
            _batch = im.size(0)
            loss += self.train_loss(im.reshape(_batch, -1), y.reshape(_batch, -1))
            pred = torch.cat((pred, im.unsqueeze(-1)), -1)
            xx = torch.cat((xx[..., 1:], im.unsqueeze(-1)), dim=-1)

        loss = self.train_loss(pred, batch['y'])
        if self.average_over_batch:
            loss /=  batch['y'].shape[0]

        if self.average_over_time:
            loss /= self.t_train - self.initial_steps
        
        loss = torch.mean(loss)
        self.log('train_err', value=loss)
        return loss
    
    def validation_step(self, batch, batch_idx, *args, **kwargs):
        yy = batch['y'] # [b, c, x, y, t]
        xx = batch['x'] # [b, c, x, y, t]
        try:
            consts = batch['consts']
        except:
            consts = None
        pred = yy[..., :self.initial_steps]
        
        inp_shape = list(xx.shape)
        inp_shape[1] *= inp_shape[-1]
        inp_shape.pop(-1)

        loss_dict = {key: 0. for key in self.metric_dict.keys()}

        for t in range(self.initial_steps, self.t_train):
            inp = xx.reshape(inp_shape)
            # Extract target at current time step
            y = yy[..., t]
            inputs = {'x': inp, 'y': y}
            if consts is not None:
                inputs.update({'consts': consts})
            im = self.model(**inputs)

            _batch = im.size(0)
            for key in self.metric_dict.keys():
                loss_dict[key] += self.metric_dict[key](
                    im.reshape(_batch, -1), y.reshape(_batch, -1)
                )
            pred = torch.cat((pred, im.unsqueeze(-1)), -1)
            xx = torch.cat((xx[..., 1:], im.unsqueeze(-1)), dim=-1)
        # _batch = yy.size(0)
        # _pred = pred[..., self.initial_steps:self.t_train, :]
        # _yy = yy[..., self.initial_steps:self.t_train, :]
        for key in self.metric_dict.keys():
            # loss = self.metric_dict[key](_pred, _yy)
            if self.average_over_batch:
                loss_dict[key] /=  batch['y'].shape[0]
            if self.average_over_time:
                loss_dict[key] /= self.t_train - self.initial_steps
            loss_dict[key] = torch.mean(loss_dict[key])
        self.log_dict(loss_dict)
    
    def test_step(self, batch, batch_idx, *args, **kwargs):
        pred = self.model(**batch)
        loss_dict = dict()
        for key in self.metric_dict.keys():
            loss = self.metric_dict[key](pred, batch['y'])
            if self.average_over_batch:
                loss /=  batch['y'].shape[0]
            loss_dict[key] = loss
        self.log_dict(loss_dict)

    def predict_step(self, batch, *args: Any, **kwargs: Any) -> Any:
        pred = self.model(**batch)
        return pred, batch['y']


class MultiMetricModule(L.LightningModule):
    """
    Basic Pytorch-Lightning module for AI for Science Tasks.
    Defines the train/val/test/predict steps, the major features are:
        1. The prediction target for each batch is batch['y']
        2. The 
    TODO: add single metric
    """
    def __init__(self, model, optimizer, scheduler, train_loss, metric_dict:dict, average_over_batch=True, apply_rollout=False) -> None:
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loss = train_loss
        self.metric_dict = metric_dict
        self.average_over_batch = average_over_batch
        self.apply_rollout = apply_rollout

    def configure_optimizers(self):
        return {'optimizer':self.optimizer, 'lr_scheduler': self.scheduler}

    def training_step(self, batch, batch_idx, *args, **kwargs):
        # pred = self.model(**batch)
        # has_nan = torch.isnan(pred).any()
        # if has_nan:
        #     import pdb; pdb.set_trace()
        pred = self.model(**batch)
        loss = self.train_loss(pred, batch['y'])
        if self.average_over_batch:
            loss /=  batch['y'].shape[0]
        loss = torch.mean(loss)
        self.log('train_err', value=loss)
        return loss
    
    def validation_step(self, batch, batch_idx, *args, **kwargs):
        pred = self.model(**batch)
        loss_dict = dict()
        for key in self.metric_dict.keys():
            loss = self.metric_dict[key](pred, batch['y'])
            if self.average_over_batch:
                loss /=  batch['y'].shape[0]
            loss_dict[key] = torch.mean(loss)
        self.log_dict(loss_dict)
    
    def test_step(self, batch, batch_idx, *args, **kwargs):
        if not self.apply_rollout:
            return self.original_test_step(batch, batch_idx, *args, **kwargs)
        # else:
        rollouts = batch['t'][0]
        # Specifically designed for TorusVisForce!
        if batch_idx == 0:
            print(rollouts)
        for t in range(rollouts):
            pred = self.model(**batch)
            batch['x'][:, 0:1] = pred
        loss_dict = dict()
        for key in self.metric_dict.keys():
            loss = self.metric_dict[key](pred, batch['y'])
            if self.average_over_batch:
                loss /=  batch['y'].shape[0]
            loss_dict[key] = loss
        self.log_dict(loss_dict)

    def original_test_step(self, batch, batch_idx, *args, **kwargs):
        pred = self.model(**batch)
        loss_dict = dict()
        for key in self.metric_dict.keys():
            loss = self.metric_dict[key](pred, batch['y'])
            if self.average_over_batch:
                loss /=  batch['y'].shape[0]
            loss_dict[key] = loss
        self.log_dict(loss_dict)

    def predict_step(self, batch, *args: Any, **kwargs: Any) -> Any:
        pred = self.model(**batch)
        return pred, batch['y']
    

class MultiTaskModule(L.LightningModule):
    """
    Multi-Task Learning Pytorch-Lightning module for AI for Science Tasks.
    Defines the train/val/test/predict steps, the major features are:
        1. The prediction target for each batch is batch['y']

    Must be cocupled with the MultiTask DataModule!
    TODO: add single metric
    """
    def __init__(self, model, optimizer, scheduler, train_loss, metric_dict:dict, n_tasks, n_val_tasks=-1, train_data_names:list=None, val_data_names:list=None, log_on_epoch=True, average_over_batch=True, prediction_output_x=False) -> None:
        super().__init__()
        self.automatic_optimization = False
        self.model = model
        self.optimizer = optimizer
        if type(train_loss) != type([]):
            train_loss = [train_loss for _ in range(n_tasks)]
        assert len(train_loss) == n_tasks, f'len(train_loss) != n_tasks: {train_loss, n_tasks}'
        self.train_loss = train_loss
        self.metric_dict = metric_dict
        self.scheduler = scheduler
        self.log_on_epoch = log_on_epoch
        self.average_over_batch = average_over_batch

        if n_val_tasks == -1:
            n_val_tasks = n_tasks
        
        self.n_tasks = n_tasks
        self.n_val_tasks = n_val_tasks

        self.train_data_names = train_data_names
        if train_data_names == None:
            self.train_data_names = [f'train_err_{i}' for i in range(n_tasks)]

        self.val_data_names = val_data_names
        if val_data_names == None:
            self.val_data_names = [f'val_err_{i}' for i in range(n_val_tasks)]
        
        self.train_loss = train_loss
        self.metric_dict = metric_dict
        self.prediction_output_x = prediction_output_x

    def configure_optimizers(self):
        return {'optimizer':self.optimizer, 'lr_scheduler': self.scheduler}
    
    def training_step(self, batch, *args, **kwargs):
        # train loader is a CombinedLoader(dataloaders, "max_size_cycle")
        # batch: [sampled(datasets[i]) for i in range(num_datasets)]
        assert len(batch) == self.n_tasks, f'len(batch) != self.n_tasks, {len(batch), self.n_tasks}'
        train_err_batch = 0
        for i in range(self.n_tasks):
            self.optimizer.zero_grad()
            pred = self.model(**batch[i])
            loss = self.train_loss[i](pred, batch[i]['y']).mean()
            self.log(self.train_data_names[i] + '_batch', loss, on_epoch=False, on_step=True)
            self.log(self.train_data_names[i], loss, on_epoch=True, on_step=False)
            loss.backward()
            self.optimizer.step()
            train_loss = loss.detach()
            if self.average_over_batch:
                train_loss /=  batch[i]['y'].shape[0]
            train_err_batch += train_loss
        self.log('train_err_batch', train_err_batch, on_epoch=False, on_step=True)
        self.log('train_err', train_err_batch, on_epoch=True, on_step=False)
        return loss
    
    def validation_step(self, batch, *args, **kwargs):
        # val loader is a CombinedLoader(dataloaders, "sequential")
        # assert len(batch) == self.n_val_tasks, f'len(batch) != self.n_val_tasks, {len(batch), self.n_val_tasks}'
        pred = self.model(**batch)
        loss_dict = dict()
        # data_name = self.val_data_names[dataloader_idx]
        for key in self.metric_dict.keys():
            loss = self.metric_dict[key](pred, batch['y']).mean()
            if self.average_over_batch:
                loss /=  batch['y'].shape[0]
            loss_dict[key] = loss.mean()
            # loss_dict[key+"/"+data_name] = self.metric_dict[key](pred, batch['y'])
        self.log_dict(loss_dict)
    
    def test_step(self, batch, *args, **kwargs):
        pred = self.model(**batch)
        loss_dict = dict()
        for key in self.metric_dict.keys():
            loss = self.metric_dict[key](pred, batch['y'])
            if self.average_over_batch:
                loss /=  batch['y'].shape[0]
            loss_dict[key] = loss
        self.log_dict(loss_dict)

    def predict_step(self, batch, *args: Any, **kwargs: Any) -> Any:
        pred = self.model(**batch)
        if self.prediction_output_x:
            return pred, batch['y'], batch['x']
        else:
            return pred, batch['y']
    
