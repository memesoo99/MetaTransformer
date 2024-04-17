from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from timm.models.vision_transformer import Block
from Data2Seq.Data2Seq import Data2Seq as dataseq



ckpt = torch.load("/root/genni/Meta-Transformer_base_patch16_encoder.pth")

class Net(nn.Module):
    def __init__(self, ckpt, device = 'cuda'):
        super(Net, self).__init__()
        # self.image_tokenizer = dataseq(modality='image',dim=768)
        # self.image_tokenizer = self.image_tokenizer.to(device)
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
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.conv1 = torch.nn.Conv2d(768, 100, kernel_size=(14, 14)) # Assuming (196 = 14*14)
        self.fc1 = nn.Linear(768, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.expand(-1, 3,-1,-1)
        x = self.image_tokenizer(x) # batch * 196 * 768
        x = self.encoder(x)
        x = x.mean(dim=1) #Average Pooling
        # x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


# def main():
#     # Training settings
#     parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
#     parser.add_argument('--batch-size', type=int, default=64, metavar='N',
#                         help='input batch size for training (default: 64)')
#     parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
#                         help='input batch size for testing (default: 1000)')
#     parser.add_argument('--epochs', type=int, default=10, metavar='N',
#                         help='number of epochs to train (default: 14)')
#     parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
#                         help='learning rate (default: 1.0)')
#     parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
#                         help='Learning rate step gamma (default: 0.7)')
#     parser.add_argument('--no-cuda', action='store_true', default=False,
#                         help='disables CUDA training')
#     parser.add_argument('--no-mps', action='store_true', default=False,
#                         help='disables macOS GPU training')
#     parser.add_argument('--dry-run', action='store_true', default=False,
#                         help='quickly check a single pass')
#     parser.add_argument('--seed', type=int, default=1, metavar='S',
#                         help='random seed (default: 1)')
#     parser.add_argument('--log-interval', type=int, default=10, metavar='N',
#                         help='how many batches to wait before logging training status')
#     parser.add_argument('--save-model', action='store_true', default=True,
#                         help='For Saving the current Model')
#     args = parser.parse_args()
#     use_cuda = not args.no_cuda and torch.cuda.is_available()
#     use_mps = not args.no_mps and torch.backends.mps.is_available()

#     torch.manual_seed(args.seed)

#     if use_cuda:
#         device = torch.device("cuda")
#     elif use_mps:
#         device = torch.device("mps")
#     else:
#         device = torch.device("cpu")

#     transform=transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Resize(size=(224,224)),
#         transforms.Normalize((0.1307,), (0.3081,))
#         ])
#     # dataset1 = datasets.MNIST('../data', train=True, download=True,
#     #                    transform=transform)
#     # dataset2 = datasets.MNIST('../data', train=False,
#     #                    transform=transform)
#     # train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
#     # test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)


#     train_dataset = datasets.MNIST('/root/genni/MNIST', train=True, download=True)
#     eval_dataset = datasets.MNIST('/root/genni/MNIST', train=False, download=True)

#     train_loader = torch.utils.data.DataLoader(train_dataset,
#                                             batch_size=64,
#                                             shuffle=True,)

#     eval_loader = torch.utils.data.DataLoader(eval_dataset,
#                                           batch_size=64,
#                                           shuffle=True)
#     model = Net(ckpt).to(device)
#     optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

#     scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
#     for epoch in range(1, args.epochs + 1):
#         train(args, model, device, train_loader, optimizer, epoch)
#         test(model, device, eval_loader)
#         scheduler.step()

#     if args.save_model:
#         torch.save(model.state_dict(), "mnist_cnn.pt")


# if __name__ == '__main__':
#     main()

### CONFIG ###
device = torch.device("cuda")
epochs = 10
save_model = True
gamma = 0.7
lr = 0.001
##############

transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(size=(224,224)),
    transforms.Normalize((0.1307,), (0.3081,))
    ])


train_dataset = datasets.MNIST('/root/genni/MNIST', train=True, download=True,transform=transform)
eval_dataset = datasets.MNIST('/root/genni/MNIST', train=False, download=True,transform=transform)

train_loader = torch.utils.data.DataLoader(train_dataset,
                                            batch_size=64,
                                            shuffle=True,)

eval_loader = torch.utils.data.DataLoader(eval_dataset,
                                          batch_size=64,
                                          shuffle=True)
model = Net(ckpt).to(device)
image_tokenizer = dataseq(modality='image',dim=768).to(device)
Net.image_tokenizer = image_tokenizer

optimizer = optim.Adadelta(model.parameters(), lr=lr)

scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
for epoch in range(1, epochs + 1):
    train(model, device, train_loader, optimizer, epoch)
    test(model, device, eval_loader)
    scheduler.step()

    if save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")