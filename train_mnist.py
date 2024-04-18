from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import logging
import os
from datetime import datetime
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from timm.models.vision_transformer import Block
from Data2Seq.Data2Seq import Data2Seq as dataseq

# class ExpandChannelTransform:
#     def __call__(self, x):
#         return x.expand(-1, 3, -1, -1)

ckpt = torch.load("/root/genni/Meta-Transformer_base_patch16_encoder.pth")

class Net(nn.Module):
    def __init__(self, ckpt, num_classes=10, device = 'cuda'):
        super(Net, self).__init__()
        # self.image_tokenizer = None
        # self.image_tokenizer = dataseq(modality='image',dim=768)
        # self.image_tokenizer = self.image_tokenizer.to(device)
        self.num_classes = num_classes
        self.encoder = nn.Sequential(*[
            Block(
                dim=768,
                num_heads=12,
                mlp_ratio=4.,
                qkv_bias=True,
                norm_layer=nn.LayerNorm,
                act_layer=nn.GELU
            )
            for i in range(12)])
        self.encoder.load_state_dict(ckpt,strict=True)
        self.encoder.to(device)
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        # self.conv1 = torch.nn.Conv2d(768, 100, kernel_size=(14, 14)) # Assuming (196 = 14*14)
        self.fc1 = nn.Linear(768, 1024)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, self.num_classes)

    def forward(self, x):
        # x = x.expand(-1, 3,-1,-1)
        x = self.image_tokenizer(x) # batch * 196 * 768
        x = self.encoder(x)
        x = x.mean(dim=1) # Average Pooling
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        # x = self.dropout2(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        # output = F.log_softmax(x, dim=1)
        return x


def train(model, device, train_loader, optimizer, epoch, loss):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        train_loss = loss(output, target)
        train_loss.backward()
        optimizer.step()
        if batch_idx % 30 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), train_loss.item()))
            logging.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), train_loss.item()))


def test(model, device, test_loader, loss):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss(output, target).item()  # sum up batch loss
            # test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    logging.info('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=8, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--dataset', type=str, default="mnist",)
    parser.add_argument('--img-size', type=int, default=224,)
    parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.9, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()
    now = datetime.now()

    current_time = now.strftime("%m-%d-%H-%M")
    save_PATH = f"/root/genni/MetaTransformer/train_log/{current_time}-{args.dataset}"
    os.makedirs(save_PATH, exist_ok=True)
    logging.basicConfig(level=logging.INFO, filename=f"{save_PATH}/logfile", filemode="a+",
                        format="%(asctime)-15s %(levelname)-8s %(message)s")
    logging.info("Input args: %r", args)

    torch.manual_seed(args.seed)

    ### CONFIG ###
    device = 'cpu'
    if torch.cuda.is_available():
        device = torch.device("cuda")
    epochs = 10
    gamma = 0.7
    lr = 0.001
    ##############

    

    # Load Datset
    if args.dataset == 'mnist':
        
        transform=transforms.Compose([
            transforms.ToTensor(),
            # transforms.Resize(size=(224,224)),
            transforms.Resize(size=(args.img_size,args.img_size)),
            transforms.Normalize((0.1307,), (0.3081,)),
            lambda x: x.expand(3, -1, -1)
            ])
        
        train_dataset = datasets.MNIST('/root/genni/MNIST', train=True, download=True,transform=transform)
        eval_dataset = datasets.MNIST('/root/genni/MNIST', train=False, download=True,transform=transform)

        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                batch_size=args.batch_size,
                                                shuffle=True,)

        eval_loader = torch.utils.data.DataLoader(eval_dataset,
                                            batch_size=args.test_batch_size,
                                            shuffle=True)
        num_classes = 10

    elif args.dataset == 'texture':
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(size=(args.img_size,args.img_size)),
            transforms.Normalize((0.5,), (0.5,))
            ])

        train_dataset = datasets.ImageFolder('/root/genni/textures/images',transform=transform)
        eval_dataset = datasets.ImageFolder('/root/genni/textures/images', transform=transform)

        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                    batch_size=args.batch_size,
                                                    shuffle=True,)

        eval_loader = torch.utils.data.DataLoader(eval_dataset,
                                                batch_size=args.test_batch_size,
                                          shuffle=True)
        # loss = nn.CrossEntropyLoss()
        num_classes = 47

    loss = nn.CrossEntropyLoss()
    model = Net(ckpt,num_classes).to(device)
    image_tokenizer = dataseq(modality='image', dim=768, image_size = args.img_size).to(device)
    Net.image_tokenizer = image_tokenizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    for epoch in range(1, args.epochs + 1):
        train(model, device, train_loader, optimizer, epoch, loss)
        test(model, device, eval_loader,loss)
        scheduler.step()

    if args.save_model:
        torch.save(model.state_dict(), f"{save_PATH}/{args.dataset}.pt")


if __name__ == '__main__':
    
    main()