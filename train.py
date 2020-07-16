# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import torch
import torch.utils.data as td
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import os
from tqdm import tqdm, trange
import argparse

# %%
import model.brats_dataset
import model.unet
import model.utils as utils
import model.loss



# %%

class Context:
    def __init__(self, params, checkpoint_filename=None, criterion=None):
        self.params = params

        self.net = model.unet.Net()
        if checkpoint_filename:
            checkpoint = torch.load(checkpoint_filename)
            self.net.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])
            self.run_iter = checkpoint['run_iter'] + 1
            self.global_iter = checkpoint['global_iter']
        else:
            self.net.apply(model.unet.init_weights)
            self.optimizer = torch.optim.SGD(
                self.net.parameters(),
                lr=self.params['lr'],
                weight_decay=1e-7,
                momentum=0.8
            )
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                'min',
                patience=2
            )
            self.run_iter = 0
            self.global_iter = 0
            self.save_checkpoint()

        self.device = model.utils.try_gpu()
        print('Using device {}'.format(self.device))

        self.run_iter = 0

        self.model_shape = (1, 4, 240, 240)

    def check_topology(self):
        with torch.no_grad():
            dummy_input = torch.empty(size=self.model_shape, dtype=torch.float32)
            y = self.net(dummy_input)

    def export_onnx(self, filename):
        dummy_input = torch.empty(size=self.model_shape)
        torch.onnx.export(self.net, dummy_input, filename, verbose=True)

    def save_checkpoint(self):
        fn = 'checkpoint_{}_{}.pt'.format(self.params['meta_name'], self.run_iter)
        print('Writing checkpoint {}'.format(fn))
        state_dict = {
            'model': self.net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'run_iter': self.run_iter,
            'global_iter': self.global_iter,
            }
        torch.save(state_dict, os.path.join('checkpoints', fn))
        print('done')

class TrainContext:
    def __init__(self, context, data_context, criterion):
        self.ctx = context
        self.data = data_context
        self.criterion = criterion

    def run(self, num_epochs):
        self.ctx.run_iter += 1
        run_name = 'run_{}_{}_{}'.format(self.ctx.params['meta_name'], self.ctx.params['lr'], self.ctx.run_iter)

        print('Commencing {}'.format(run_name))
        self.writer = SummaryWriter(os.path.join('runs', run_name))

        print('Writing graph')
        dummy_input = torch.empty(size=self.ctx.model_shape, dtype=torch.float32)
        self.writer.add_graph(self.ctx.net, dummy_input)
        del dummy_input
        print('done')

        self.ctx.net.to(self.ctx.device)

        self.brats_train_perepoch = self.data.split_data(num_epochs)

        for epoch in trange(num_epochs-1, desc='epoch', position=0):
            train_loss_epoch, test_loss_epoch = self.run_epoch(epoch)
            print('epoch {}/{}, train loss {}, test loss {}'.format(epoch+1,
                num_epochs, train_loss_epoch, test_loss_epoch))

        self.writer.close()


    def run_epoch(self, epoch):
        train_iter = td.DataLoader(
            self.brats_train_perepoch[epoch],
            self.ctx.params['batch_size'],
            shuffle=True,
            num_workers=self.ctx.params['num_workers'],
            pin_memory=True
        )
        test_iter = td.DataLoader(
            self.brats_train_perepoch[epoch+1],
            self.ctx.params['batch_size'],
            shuffle=True,
            num_workers=self.ctx.params['num_workers'],
            pin_memory=True
        )
        
        batch_count = len(train_iter)

        # Training
        t = tqdm(desc='batch', total=len(train_iter), position=1, leave=False)
        train_loss_epoch = 0.0
        for X, y in train_iter:
            batch_loss = self.run_batch(X, y)
            train_loss_epoch += batch_loss
            t.update(1)
            t.set_postfix({'batchloss': batch_loss})
        train_loss_epoch /= batch_count
        t.close()

        self.writer.add_scalar('loss/train', train_loss_epoch, self.ctx.global_iter)

        # Evaluating
        self.ctx.net.eval()
        test_loss_epoch = 0.0
        test_acc_epoch = 0.0
        with torch.no_grad():
            for X_test, y_test in test_iter:
                X_test = X_test.float().to(self.ctx.device)
                y_test = y_test.float().to(self.ctx.device)
                y_test_hat = self.ctx.net(X_test).squeeze(1)
                b_l = self.criterion(y_test_hat, y_test)
                test_acc_epoch += utils.jaccard(y_test_hat, y_test)
                test_loss_epoch += b_l.item()
            test_acc_epoch /= len(test_iter)
            test_loss_epoch /= len(test_iter)
            self.ctx.scheduler.step(test_loss_epoch)

        self.writer.add_scalar('accuracy/test_acc', test_acc_epoch, self.ctx.global_iter)
        self.writer.add_scalar('loss/test', test_loss_epoch, self.ctx.global_iter)
        for idx, group in enumerate(self.optimizer.param_groups):
            self.writer.add_scalar('meta/lr/group_{}'.format(idx + 1), group['lr'], self.ctx.global_iter)
        self.writer.flush()

        self.ctx.save_checkpoint()

        return train_loss_epoch, test_loss_epoch


    def run_batch(self, X, y):
        self.ctx.global_iter += 1
        self.ctx.net.train()

        X = X.float().to(self.ctx.device)
        y = y.float().to(self.ctx.device)
        y_hat = self.ctx.net(X).squeeze(1)

        # Accuracy metric
        acc = utils.jaccard(y_hat, y)
        #print('Accuracy: {}'.format(acc))

        # Loss metrics
        l = self.criterion(y_hat, y)

        self.ctx.optimizer.zero_grad()
        l.backward()
        nn.utils.clip_grad_value_(self.ctx.net.parameters(), 0.1)
        self.ctx.optimizer.step()

        for tag, value in self.ctx.net.named_parameters():
            tag = tag.replace('.', '/')
            self.writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), self.ctx.global_iter)
            self.writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), self.ctx.global_iter)

        self.writer.add_scalar('accuracy/train_acc', float(acc), self.ctx.global_iter)
        self.writer.add_scalar('loss/total_loss', float(l), self.ctx.global_iter)
        self.writer.add_images('masks/0_base', X[:, 0:3, :, :], self.ctx.global_iter)
        y_us = y.unsqueeze(1)
        y_hat_us = y_hat.unsqueeze(1)
        y_hat_us_sig = torch.sigmoid(y_hat_us) > 0.5
        self.writer.add_images('masks/1_true', y_us, self.ctx.global_iter)
        self.writer.add_images('masks/2_predicted', y_hat_us_sig, self.ctx.global_iter)
        self.writer.add_images('extra/raw', y_hat_us, self.ctx.global_iter)
        overlaid = torch.cat([y_hat_us_sig.float(), y_us, torch.zeros_like(y_us)], dim=1)
        self.writer.add_images('extra/overlaid', overlaid, self.ctx.global_iter)

        self.writer.flush()

        return float(l)


# %%

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="file name of HDF5 dataset")
    parser.add_argument("--checkpoint", help="checkpoint file")
    args = parser.parse_args()
    print(args)

    dctx = model.brats_dataset.DataSplitter(args.dataset)

    params = {
        # Identifier for this group of runs
        'meta_name' : 'brats4',
        'batch_size' : 2,
        'num_workers' : 2,
        'num_epochs' : 50,
        'lr' : 1e-3,
    }

    should_check = True

    if args.checkpoint:
        ctx = Context(params, args.checkpoint)
    else:
        ctx = Context(params)

    if should_check:
        # Check whether layer inputs/outputs dimensions are correct
        # by conducting a test run
        ctx.check_topology()

    # %%

    pos_weight = torch.Tensor([5.0]).to(ctx.device)
    criterion = model.loss.Loss(pos_weight)

    tctx = TrainContext(ctx, dctx, criterion=criterion)

    tctx.run(params['num_epochs'])

    # %%

    ctx.export_onnx("net5.onnx")

