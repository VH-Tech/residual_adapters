import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from datasets import get_dataset
from utils_pytorch import train , test, AverageMeter, accuracy, log, init_logfile
from torch.utils.data import DataLoader
import time
import os

# Create Datasets
train_dataset = get_dataset("cifar10", 'train', "/storage/vatsal/datasets/cifar10")
test_dataset = get_dataset("cifar10", 'test', "/storage/vatsal/datasets/cifar10")

train_loader = DataLoader(train_dataset, shuffle=True, batch_size=128,num_workers=4)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=128, num_workers=4)


# Load the pre-trained ResNet model
model = torchvision.models.resnet18(pretrained=True)
VIT = False

# Freeze the pre-trained layers
for param in model.parameters():
    param.requires_grad = False

# Modify the final layer
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 10)

# Define the optimizer
optimizer = optim.Adam(model.fc.parameters(), lr=0.1, weight_decay=5e-4, momentum=0.9)

# Define the loss function and scheduler
criterion = nn.CrossEntropyLoss()
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

# Define the device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Move the model to the device
model.to(device)
outdir="/storage/vatsal/models/cifar10/Resnet18"
logfilename = os.path.join(outdir, 'log.txt')

init_logfile(logfilename, "epoch\ttime\tlr\ttrainloss\ttestloss\ttrainAcc\ttestAcc")
best = 0.0 
starting_epoch = 0
epochs = 200
for epoch in range(starting_epoch, epochs):
    before = time.time()
    train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch, 0)
    test_loss, test_acc = test(test_loader, model, criterion, 0)
    after = time.time()
    log(logfilename, "{}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}\t{:.3}".format(
        epoch, after - before,
        scheduler.get_lr()[0], train_loss, test_loss, train_acc, test_acc))
    scheduler.step(epoch)
   
    if test_acc > best:
        print(f'New Best Found: {test_acc}%')
        best = test_acc
        torch.save({
            'epoch': epoch + 1,
            'arch': "Resnet18",
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, os.path.join(outdir, 'checkpoint.pth.tar'))



def train(loader: DataLoader, model: torch.nn.Module, criterion, optimizer: Optimizer, epoch: int, noise_sd: float):
    """
    Function to do one training epoch
        :param loader:DataLoader: dataloader (train) 
        :param model:torch.nn.Module: the classifer being trained
        :param criterion: the loss function
        :param optimizer:Optimizer: the optimizer used during trainined
        :param epoch:int: the current epoch number (for logging)
        :param noise_sd:float: the std-dev of the Guassian noise perturbation of the input
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    # top5 = AverageMeter()
    end = time.time()
  
    # switch to train mode
    model.train()

    for i, (inputs, targets) in enumerate(loader):
        # measure data loading time
        data_time.update(time.time() - end)

        inputs = inputs.cuda()
        targets = targets.cuda()

        # augment inputs with noise
        inputs = inputs + torch.randn_like(inputs, device='cuda') * noise_sd

        # compute output
        outputs = model(inputs)
        if VIT == True :
            outputs = outputs.logits
        
        # print(outputs.shape, targets.shape)
        
        loss = criterion(outputs, targets)

        # measure accuracy and record loss
        acc1 = accuracy(outputs, targets, topk=(1,))[0]
        losses.update(loss.item(), inputs.size(0))
        top1.update(acc1.item(), inputs.size(0))
        # top5.update(acc5.item(), inputs.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                epoch, i, len(loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1))

    return (losses.avg, top1.avg)


def test(loader: DataLoader, model: torch.nn.Module, criterion, noise_sd: float):
    """
    Function to evaluate the trained model
        :param loader:DataLoader: dataloader (train)
        :param model:torch.nn.Module: the classifer being evaluated
        :param criterion: the loss function
        :param noise_sd:float: the std-dev of the Guassian noise perturbation of the input
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    # top5 = AverageMeter()
    end = time.time()

    # switch to eval mode
    model.eval()

    with torch.no_grad():
        for i, (inputs, targets) in enumerate(loader):
            # measure data loading time
            data_time.update(time.time() - end)

            inputs = inputs.cuda()
            targets = targets.cuda()

            # augment inputs with noise
            inputs = inputs + torch.randn_like(inputs, device='cuda') * noise_sd

            # compute output
            outputs = model(inputs)
            if VIT == True :
                outputs = outputs.logits
                
            loss = criterion(outputs, targets)

            # measure accuracy and record loss
            acc1 = accuracy(outputs, targets, topk=(1,))[0]
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1.item(), inputs.size(0))
            # top5.update(acc5.item(), inputs.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 10 == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    i, len(loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, top1=top1))

        return (losses.avg, top1.avg)
