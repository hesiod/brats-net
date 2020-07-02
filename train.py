# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import torch
import torch.utils.data as td
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import os

# %%
import model.brats_dataset
import model.unet
import model.utils
import model.loss

# %% [markdown]
# # Possible improvements
# 
# * Go 3D (e.g. V-Net)
# * Change model parameters (more down/up layers)
# * Fix data loader
#  - Detect empty slices
#  - Detect slices without matching labels
#  - Data augmentation (cropping, deform)
# * Try other optimizers
# * Tune hyperparameters
#  - k-fold-Validation
# * One Channel, Transfer Learning, Add another layer later
# * Try other datasets (hippocampus, liver)
# * Experiment with Loss Functions
#  - https://lars76.github.io/neural-networks/object-detection/losses-for-segmentation/
#  - https://github.com/JunMa11/SegLoss
# * Accuracy metric (measure overlay predicted/true)
# * Participate in Challenge
# 
# # Orga
# * Comment code
# * Try on Colab
# * Upload to GitHub
# * Executive Summary


# %%

# Identifier for this group of runs
meta_name = 'brats4'
batch_size = 32
num_workers = 6
num_epochs = 50
lr = 0.0001
should_check = False


# %%

class Context:
    def __init__(self, filename=None, criterion=None):
        self.net = model.unet.Net()
        if filename:
            self.net.load_state_dict(torch.load(filename))
        else:
            self.net.apply(model.unet.init_weights)

        self.device = model.utils.try_gpu()
        self.run_iter = 0
        if criterion:
            self.criterion = criterion
        else:
            self.criterion = model.loss.Loss()
        print('Using device {}'.format(self.device))

    def check_topology(self):
        dummy_input = torch.randn(size=(2, 4, 240, 240), dtype=torch.float32)
        y = self.net(dummy_input)


    def export_onnx(self, filename):
        dummy_input = torch.randn(1, 4, 240, 240)
        torch.onnx.export(self.net, dummy_input, filename, verbose=True)

class TrainContext:
    def __init__(self, context, data_context):
        self.ctx = context
        self.data = data_context
        self.global_iter = 1

        self.optimizer = torch.optim.SGD(self.ctx.net.parameters(), lr=lr)
        #optimizer = torch.optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
        #self.scheduler = ReduceLROnPlateau(self.optimizer, 'min')

        self.brats_train_perepoch = self.data.split_data(num_epochs)

    def run(self):
        self.ctx.run_iter += 1
        run_name = 'runs/run_{}_{}_{}'.format(meta_name, lr, self.ctx.run_iter)

        print('Commencing {}'.format(run_name))
        self.writer = SummaryWriter(os.path.join('runs', run_name))
        print('Writing graph')
        X = torch.randn(size=(2, 4, 240, 240), dtype=torch.float32)
        self.writer.add_graph(self.ctx.net, X)
        print('done')

        self.ctx.net.to(self.ctx.device)

        for epoch in range(num_epochs-1):
            self.run_epoch(epoch)

        self.writer.close()

    def checkpoint(self):
        torch.save(self.ctx.net.state_dict(), 'checkpoints/checkpoint_{}_{}.pt'.format(meta_name, self.ctx.run_iter))

    def run_epoch(self, epoch):
        train_iter = td.DataLoader(self.brats_train_perepoch[epoch], batch_size, shuffle=True, num_workers=num_workers)
        test_iter = td.DataLoader(self.brats_train_perepoch[epoch+1], batch_size, shuffle=True, num_workers=num_workers)
        #test_iter = td.DataLoader(brats_test, batch_size, shuffle=False, num_workers=num_workers)
        
        batch_count = len(train_iter)
        
        print('Total batches: {}'.format(batch_count))

        self.train_loss_epoch = 0.0
        for i, (X, y) in enumerate(train_iter):
            batch_loss = self.run_batch(i, X, y)
            self.train_loss_epoch += batch_loss
            print('batch {:4}/{} batchloss {}'.format(i+1, batch_count, batch_loss))
        self.train_loss_epoch /= batch_count

        self.writer.add_scalar('loss/train', self.train_loss_epoch, self.global_iter)

        self.ctx.net.eval()
        test_loss_epoch = 0.0
        with torch.no_grad():
            for X_test, y_test in test_iter:
                X_test = X_test.float().to(self.ctx.device)
                y_test = y_test.float().to(self.ctx.device)
                y_test_hat = self.ctx.net(X_test).squeeze(1)
                b_l = self.ctx.criterion(y_test_hat, y_test)
                test_loss_epoch += float(b_l)
            test_loss_epoch /= len(test_iter)

        self.writer.add_scalar('loss/test', test_loss_epoch, self.global_iter)
        self.writer.flush()

        self.checkpoint()

        print('epoch {}/{}, train loss {}, test loss {}'.format(epoch+1,
            num_epochs, self.train_loss_epoch, test_loss_epoch))

    def run_batch(self, i, X, y):
        self.global_iter += 1
        self.ctx.net.train()

        X = X.float().to(self.ctx.device)
        y = y.float().to(self.ctx.device)
        y_hat = self.ctx.net(X).squeeze(1)

        l = self.ctx.criterion(y_hat, y)

        self.optimizer.zero_grad()
        l.backward()
        nn.utils.clip_grad_value_(self.ctx.net.parameters(), 0.1)
        self.optimizer.step()

        for tag, value in self.ctx.net.named_parameters():
            tag = tag.replace('.', '/')
            self.writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), self.global_iter)
            self.writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), self.global_iter)

        self.writer.add_scalar('loss/total_loss', float(l), self.global_iter)
        self.writer.add_images('masks/0_base', X[:, 0:3, :, :], self.global_iter)
        y_us = y.unsqueeze(1)
        y_hat_us = y_hat.unsqueeze(1)
        y_hat_us_sig = torch.sigmoid(y_hat_us) > 0.5
        self.writer.add_images('masks/1_true', y_us, self.global_iter)
        self.writer.add_images('masks/2_predicted', y_hat_us_sig, self.global_iter)
        self.writer.add_images('extra/raw', y_hat_us, self.global_iter)
        overlaid = torch.cat([y_hat_us_sig.float(), y_us, torch.zeros_like(y_us)], dim=1)
        self.writer.add_images('extra/overlaid', overlaid, self.global_iter)

        self.writer.flush()

        return float(l)


# %%

dctx = model.brats_dataset.DataSplitter()

# %%

ctx = Context('checkpoint_sd.pt')

if should_check:
    # Check whether layer inputs/outputs dimensions are correct
    # by conducting a test run
    ctx.check_topology()

# %%

tctx = TrainContext(ctx, dctx)

# %%

tctx.run()

# %%

ctx.export_onnx("net5.onnx")

