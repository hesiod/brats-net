# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import torch
import torch.utils.data as td
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import math


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
brats_train = model.brats_dataset.BRATS('Task01_BrainTumour', train=True)
brats_test = model.brats_dataset.BRATS('Task01_BrainTumour', train=False)


# %%
batch_size = 8
num_workers = 6

train_iter = td.DataLoader(brats_train, batch_size, shuffle=True, num_workers=num_workers)
test_iter = td.DataLoader(brats_test, batch_size, shuffle=False, num_workers=num_workers)


# %%
num_epochs = 50
lr = 0.0001

should_check = False


# %%
net = model.unet.Net()
# print(net)
# net.apply(init_weights)


# %%
def check_topology():
    X = torch.randn(size=(2, 4, 240, 240), dtype=torch.float32)
    y = net(X)


if should_check:
    # Check whether layer inputs/outputs dimensions are correct
    # by conducting a test run
    check_topology()

# %%
#net = torch.load('checkpoint_1599.pt', map_location=torch.device('cpu'))
net = model.unet.Net()
net.load_state_dict(torch.load('checkpoint_sd.pt'))


# %%
run_iter = 0


# %%
run_iter += 1
writer = SummaryWriter('runs/brats3_{}_{}'.format(lr, run_iter))
print('Writing graph')
X = torch.randn(size=(2, 4, 240, 240), dtype=torch.float32)
writer.add_graph(net, X)
print('done')

device = model.utils.try_gpu()
print('Using device {}'.format(device))

net.to(device)

global_iter = 1

optimizer = torch.optim.SGD(net.parameters(), lr=lr)
#optimizer = torch.optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
#scheduler = ReduceLROnPlateau(optimizer, 'min')

criterion = Loss()

datasize = len(brats_train)
data_per_epoch = int(math.floor(datasize/num_epochs))
print('total count = {}, num_epochs = {}, per epoch = {}'.format(datasize, num_epochs, data_per_epoch))

brats_train_perepoch = td.random_split(brats_train, torch.full(size=[num_epochs], fill_value=data_per_epoch, dtype=torch.int))

for epoch in range(num_epochs-1): 
    train_loss_epoch = 0.0
    
    train_iter = td.DataLoader(brats_train_perepoch[epoch], batch_size, shuffle=True, num_workers=num_workers)
    test_iter = td.DataLoader(brats_train_perepoch[epoch+1], batch_size, shuffle=True, num_workers=num_workers)
    #test_iter = td.DataLoader(brats_test, batch_size, shuffle=False, num_workers=num_workers)
    
    batch_count = len(train_iter)
    
    print('Total batches: {}'.format(batch_count))

    for i, (X, y) in enumerate(train_iter):
        global_iter += 1
        net.train()

        X = X.float().to(device)
        y = y.float().to(device)
        y_hat = net(X).squeeze(1)

        l = criterion(y_hat, y)
        train_loss_epoch += float(l)

        optimizer.zero_grad()
        l.backward()
        nn.utils.clip_grad_value_(net.parameters(), 0.1)
        optimizer.step()

        if i > 0 and (i % 50 == 0):
            print('saving checkpoint...')
            torch.save(net, 'checkpoint_{}.pt'.format(i))
            print('done')
            for tag, value in net.named_parameters():
                tag = tag.replace('.', '/')
                writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_iter)
                writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_iter)
        
        writer.add_scalar('batch_loss', float(l), global_iter)
        writer.add_scalar('loss/total_loss', float(l), global_iter)
        writer.add_scalar('loss/bce_loss', float(bl), global_iter)
        writer.add_scalar('loss/dice_loss', float(dl), global_iter)
        writer.add_images('masks/0_base', X[:, 0:3, :, :], global_iter)
        y_us = y.unsqueeze(1)
        y_hat_us = y_hat.unsqueeze(1)
        y_hat_us_sig = torch.sigmoid(y_hat_us) > 0.5
        writer.add_images('masks/1_true', y_us, global_iter)
        writer.add_images('masks/2_predicted', y_hat_us_sig, global_iter)
        writer.add_images('extra/raw', y_hat_us, global_iter)
        overlaid = torch.cat([y_hat_us_sig.float(), y_us, torch.zeros_like(y_us)], dim=1)
        writer.add_images('extra/overlaid', overlaid, global_iter)
        
        writer.flush()
    
        print('batch {:4}/{} batchloss {}'.format(i, batch_count, float(l)))

    train_loss_epoch /= batch_count
    
    writer.add_scalar('loss/train', train_loss_epoch, global_iter)
  
    net.eval()
    test_loss_epoch = 0.0
    with torch.no_grad():
        for X_test, y_test in test_iter:
            X_test = X_test.float().to(device)
            y_test = y_test.float().to(device)
            y_test_hat = net(X_test).squeeze(1)
            b_l = criterion(y_test_hat, y_test) 
            test_loss_epoch += float(b_l)
        test_loss_epoch /= len(test_iter)  
        
    writer.add_scalar('loss/test', test_loss_epoch, global_iter)
    
    print('epoch {}/{}, train loss {}, test loss {}'.format(epoch+1, num_epochs, train_loss_epoch, test_loss_epoch)) 


# %%
#torch.save(net, 'checkpoint_rg2.pt')
torch.save(net.state_dict(), 'checkpoint_sd.pt')


# %%
def export_onnx(filename):
    dummy_input = torch.randn(1, 4, 240, 240)
    torch.onnx.export(net, dummy_input, filename, verbose=True)


export_onnx("net5.onnx")


# %%


