# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import torch
import torch.utils.data as td
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import os
from tqdm import tqdm, trange
from pathlib import Path
import sys
import json

# %%
import model.brats_dataset
import model.unet
import model.utils as utils
import model.loss
import model.kfold

# %%

class Context:
    def __init__(self, filename=None, criterion=None):
        self.net = model.unet.Net()
        if filename:
            self.net.load_state_dict(torch.load(filename))
        else:
            self.net.apply(model.unet.init_weights)

        self.device = model.utils.try_gpu()
        print('Using device {}'.format(self.device))

        self.run_iter = 0

    def check_topology(self):
        dummy_input = torch.randn(size=(2, 4, 240, 240), dtype=torch.float32)
        y = self.net(dummy_input)


    def export_onnx(self, filename):
        dummy_input = torch.randn(1, 4, 240, 240)
        torch.onnx.export(self.net, dummy_input, filename, verbose=True)

class TrainContext:
    def __init__(self, context, data_context, criterion, lr, batch_size, experiment_name='unnamed'):
        self.ctx = context
        self.data = data_context
        self.global_iter = 1
        self.experiment_name = experiment_name

        self.criterion = criterion
        self.batch_size = batch_size
        self.ctx.net.to(context.device)
        self.optimizer = torch.optim.SGD(self.ctx.net.parameters(), lr=lr)
        #optimizer = torch.optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
        #self.scheduler = ReduceLROnPlateau(self.optimizer, 'min')

    def run(self, num_epochs):
        self.ctx.run_iter += 1
        run_name = 'runs/run_{}_{}_{}'.format(self.experiment_name, lr, self.ctx.run_iter)

        print('Commencing {}'.format(run_name))
        self.writer = SummaryWriter(os.path.join('runs', run_name))

        print('Writing graph')
        dummy_input = torch.randn(size=(2, 4, 240, 240), dtype=torch.float32)
        self.writer.add_graph(self.ctx.net, dummy_input)
        del dummy_input
        print('done')

        self.ctx.net.to(self.ctx.device)

        self.brats_train_perepoch = self.data.split_data(num_epochs)

        for epoch in trange(num_epochs-1, desc='epoch', position=0):
           self.run_epoch(epoch)

        self.writer.close()

    def checkpoint(self):
        Path("checkpoints").mkdir(parents=True, exist_ok=True)
        torch.save(self.ctx.net.state_dict(), 'checkpoints/checkpoint_{}_{}.pt'.format(self.experiment_name, self.ctx.run_iter))

    def run_epoch(self, epoch):
        train_iter = td.DataLoader(self.brats_train_perepoch[epoch], batch_size, shuffle=True, num_workers=num_workers)
        test_iter = td.DataLoader(self.brats_train_perepoch[epoch+1], batch_size, shuffle=True, num_workers=num_workers)
        #test_iter = td.DataLoader(brats_test, batch_size, shuffle=False, num_workers=num_workers)
        
        batch_count = len(train_iter)

        # Training
        t = tqdm(desc='batch', total=len(train_iter), position=1)
        train_loss_epoch = 0.0
        for i, (X, y) in enumerate(train_iter):
            batch_loss = self.run_batch(i, X, y)
            train_loss_epoch += batch_loss
            t.update(1)
            t.set_postfix({'batchloss': batch_loss})
        train_loss_epoch /= batch_count
        t.close()

        self.writer.add_scalar('loss/train', train_loss_epoch, self.global_iter)

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
                test_acc_epoch = utils.jaccard(y_test_hat, y_test)
                test_loss_epoch += b_l.item()
            test_acc_epoch /= len(test_iter)
            test_loss_epoch /= len(test_iter)

        self.writer.add_scalar('accuracy/test_acc', test_acc_epoch, self.global_iter)
        self.writer.add_scalar('loss/test', test_loss_epoch, self.global_iter)
        self.writer.flush()

        self.checkpoint()

        print('epoch {}/{}, train loss {}, test loss {}'.format(epoch+1,
            num_epochs, train_loss_epoch, test_loss_epoch))

    def run_batch(self, i, X, y):
        self.global_iter += 1
        self.ctx.net.train()

        X = X.float().to(self.ctx.device)
        y = y.float().to(self.ctx.device)
        y_hat = self.ctx.net(X).squeeze(1)

        # Accuracy metric
        acc = utils.jaccard(y_hat, y)
        #print('Accuracy: {}'.format(acc))

        # Loss metrics
        l = self.criterion(y_hat, y)

        self.optimizer.zero_grad()
        l.backward()
        nn.utils.clip_grad_value_(self.ctx.net.parameters(), 0.1)
        self.optimizer.step()

        for tag, value in self.ctx.net.named_parameters():
            tag = tag.replace('.', '/')
            self.writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), self.global_iter)
            self.writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), self.global_iter)

        self.writer.add_scalar('accuracy/train_acc', float(acc), self.global_iter)
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

    def run_train(self, X, y):
        self.ctx.net.train()

        X = X.float().to(self.ctx.device)
        y = y.float().to(self.ctx.device)
        y_hat = self.ctx.net(X).squeeze(1)

        # Accuracy metric
        acc = utils.jaccard(y_hat, y)
        #print('Accuracy: {}'.format(acc))

        # Loss metrics
        l = self.criterion(y_hat, y)

        self.optimizer.zero_grad()
        l.backward()
        nn.utils.clip_grad_value_(self.ctx.net.parameters(), 0.1)
        self.optimizer.step()

        return acc

    def run_test(self, X, y):
        self.ctx.net.eval()
        with torch.no_grad():
            X = X.float().to(self.ctx.device)
            y = y.float().to(self.ctx.device)
            y_hat = self.ctx.net(X)

            # May change loss to accuracy
            return self.criterion(y_hat, y)

# %%

if __name__ == '__main__':
    dctx = model.brats_dataset.DataSplitter('dataset_BRATS.hdf5')
    # %%

    # Identifier for this group of runs
    meta_name = 'brats4'
    batch_size = 4
    num_workers = 4
    num_epochs = 50
    lr = 0.0001
    should_check = False

    # %%

    ctx = Context() #'checkpoint_sd.pt'

    if should_check:
        # Check whether layer inputs/outputs dimensions are correct
        # by conducting a test run
        ctx.check_topology()

    # %%

    pos_weight = torch.Tensor([5.0]).to(ctx.device)
    criterion = model.loss.Loss(pos_weight)

    tctx = TrainContext(ctx, dctx, criterion=criterion, lr=lr, batch_size=batch_size, experiment_name=meta_name)
    
    # %%
    # Currently expecting X (Image), y (labels) tensors for the kfold 
    # Not finished yet
    do_kfold = True
    if do_kfold:
        KFold = model.kfold.KFold()
        best_result = 0
        best_split = []
        dataset = model.brats_dataset.BRATS("./dataset_BRATS.hdf5")
        size = len(dataset)
        amount = 1

        for train_idx, test_idx in KFold.k_folds(size*0.10, size):
            ctx = Context()
            tctx = TrainContext(ctx, dctx, criterion=criterion, lr=lr, batch_size=batch_size, experiment_name=meta_name)
            train_dataset = torch.utils.data.Subset(dataset, train_idx)
            test_dataset = torch.utils.data.Subset(dataset, test_idx)
    
            train_data = td.DataLoader(train_dataset, batch_size, shuffle=True, num_workers=num_workers)
            test_data = td.DataLoader(test_dataset, batch_size, shuffle=True, num_workers=num_workers)

            result_train = 0        
            for X, y in train_data:
                result_train += tctx.run_train(X, y)
            result_train = result_train / len(train_data)

            result_test = 0
            for X, y in test_data:
                result_test += tctx.run_test(X, y)
            result_test = result_test / len(test_data)

            print('Fold {}/{}, train acc {}, test acc {}'.format(amount, int(size/0.10), result_train, result_test))
            if result_test > best_result:
                best_result = result_test
                best_split = [train_idx, test_idx]
            
            amount += 1

        export_data = {
            "train": best_split[0],
            "test": best_split[1]
        }

        with open('indices.json', 'w') as out:
            json.dump(export_data, out)


    
    # %%

    tctx.run(num_epochs)

    # %%

    ctx.export_onnx("net5.onnx")

