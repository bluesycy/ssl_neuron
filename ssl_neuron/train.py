import os
import torch
import torch.optim as optim
from ssl_neuron.utils import AverageMeter, compute_eig_lapl_torch_batch

class Trainer(object):
    def __init__(self, config, model, dataloaders, resume_path=None):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.config = config
        self.ckpt_dir = config['trainer']['ckpt_dir']
        self.save_every = config['trainer']['save_ckpt_every']

        ### datasets
        self.train_loader = dataloaders[0]
        self.val_loader= dataloaders[1]

        ### trainings params
        self.max_iter = config['optimizer']['max_iter']
        self.init_lr = config['optimizer']['lr']
        self.exp_decay = config['optimizer']['exp_decay']
        self.lr_warmup = torch.linspace(0., self.init_lr,  steps=(self.max_iter // 50)+1)[1:]
        self.lr_decay = self.max_iter // 5
        
        self.optimizer = optim.Adam(list(self.model.parameters()), lr=0)
        self.curr_iter = 0
        self.start_epoch = 0

        if resume_path is not None:
            self._load_checkpoint(resume_path)
        
        
    def set_lr(self): 
        if self.curr_iter < len(self.lr_warmup):
            lr = self.lr_warmup[self.curr_iter]
        else:
            lr = self.init_lr * self.exp_decay ** ((self.curr_iter - len(self.lr_warmup)) / self.lr_decay)
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            
        return lr
        

    def train(self):     
        epoch = self.start_epoch
        while self.curr_iter < self.max_iter:
            # Run one epoch.
            loss = self._train_epoch(epoch)

            if epoch % self.save_every == 0:
                # Save checkpoint.
                self._save_checkpoint(epoch, loss)
            
            epoch += 1


    def _train_epoch(self, epoch):
        self.model.train()
        losses = AverageMeter()
        for i, data in enumerate(self.train_loader, 0):
            f1, f2, a1, a2 = [x.float().to(self.device, non_blocking=True) for x in data]
            n = a1.shape[0]

            # compute positional encoding
            l1 = compute_eig_lapl_torch_batch(a1)
            l2 = compute_eig_lapl_torch_batch(a2)
            
            self.lr = self.set_lr()
            self.optimizer.zero_grad(set_to_none=True)
            
            loss = self.model(f1, f2, a1, a2, l1, l2)

            # optimize 
            loss.sum().backward()
            self.optimizer.step()
            
            # update teacher weights
            self.model.update_moving_average()
            
            losses.update(loss.detach(), n)
            self.curr_iter += 1

        print('Epoch {} | Loss {:.4f}'.format(epoch, losses.avg))
        return losses.avg


    def _save_checkpoint(self, epoch, loss):
        filename = 'ckpt_{}.pt'.format(epoch)
        path = os.path.join(self.ckpt_dir, filename)
        payload = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epoch': epoch,
            'curr_iter': self.curr_iter,
            'loss': loss
        }
        torch.save(payload, path)
        print('Save model after epoch {} as {}.'.format(epoch, filename))


    def _load_checkpoint(self, resume_path):
        if not os.path.isfile(resume_path):
            raise FileNotFoundError('Checkpoint not found: {}'.format(resume_path))

        print('Resume from checkpoint {}'.format(resume_path))
        checkpoint = torch.load(resume_path, map_location=self.device)

        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            self.model.load_state_dict(checkpoint['model'])

            optimizer_state = checkpoint.get('optimizer')
            if optimizer_state:
                self.optimizer.load_state_dict(optimizer_state)
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(self.device, non_blocking=True)

            self.curr_iter = checkpoint.get('curr_iter', 0)
            last_epoch = checkpoint.get('epoch', -1)
            self.start_epoch = max(last_epoch + 1, 0)
        else:
            self.model.load_state_dict(checkpoint)
            self.curr_iter = 0
            self.start_epoch = 0
            print('Loaded weights only; optimizer and iteration state reset.')