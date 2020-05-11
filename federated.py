import pandas as pd
import numpy as np
import syft as sy
from syft import federated
import warnings
import time
import torch.utils.data as data_utils
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split


warnings.filterwarnings('ignore')

device = torch.device("cpu")
# device = torch.device("cuda")

# hooking pytorch
hook = sy.TorchHook(torch)
bob = sy.VirtualWorker(hook, id="bob")
alice = sy.VirtualWorker(hook, id="alice")
# james = sy.VirtualWorker(hook, id="james")

compute_nodes = [bob, alice]
# loading the dataset
df = pd.read_csv('/Users/ast/Desktop/labelled_1.csv')
df.drop(df.head(3).index, inplace=True)
df.drop(df.columns[[14, 15]], axis=1, inplace = True)
labels = df[df.columns[-1]]
labels = labels.map({ 'BENIGN' : 0, 'SSH-Patator' : 1, 'FTP-Patator' : 1})
features = df[df.columns[:-1]]
X_train, X_test, Y_train, Y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
N_FEATURES = X_train.shape[1]
# labels_tensor = torch.tensor(labels.values.astype(np.float32))
# features_tensor = torch.tensor(features.values.astype(np.float32))
X_train_tensor = torch.tensor(X_train.values.astype(np.float32))
Y_train_tensor = torch.tensor(Y_train.values.astype(np.float32))
X_test_tensor = torch.tensor(X_test.values.astype(np.float32))
Y_test_tensor = torch.tensor(Y_test.values.astype(np.float32))
train = data_utils.TensorDataset(X_train_tensor, Y_train_tensor)
test = data_utils.TensorDataset(X_test_tensor, Y_test_tensor)
# train_loader = data_utils.DataLoader(train, batch_size=128, shuffle=True)
# test_loader = data_utils.DataLoader(test, batch_size=128, shuffle=True)
train_federated_dataloader = sy.FederatedDataLoader(train.federate(compute_nodes), shuffle=True, batch_size=128)
test_federated_dataloader = sy.FederatedDataLoader(test.federate(compute_nodes), shuffle=True, batch_size=128)



#neural network structure
DROPOUT_PROB = 0.90
epochs = 60
LR = 0.005
MOMENTUM = 0.9
dropout = torch.nn.Dropout(p=1 - (DROPOUT_PROB))


hiddenLayer1Size = 512
hiddenLayer2Size = int(hiddenLayer1Size/4)
hiddenLayer3Size = int(hiddenLayer1Size/8)
hiddenLayer4Size = int(hiddenLayer1Size/16)
hiddenLayer5Size = int(hiddenLayer1Size/32)

linear1 = torch.nn.Linear(N_FEATURES, hiddenLayer1Size, bias=True)
torch.nn.init.xavier_uniform(linear1.weight)

linear2 = torch.nn.Linear(hiddenLayer1Size, hiddenLayer2Size)
torch.nn.init.xavier_uniform(linear2.weight)

linear6 = torch.nn.Linear(hiddenLayer2Size, 1)
torch.nn.init.xavier_uniform(linear6.weight)

sigmoid = torch.nn.Sigmoid()
tanh = torch.nn.Tanh()
relu = torch.nn.LeakyReLU()

net = torch.nn.Sequential(linear1, nn.BatchNorm1d(hiddenLayer1Size), relu,
                          linear2, dropout, relu,
                          linear6, sigmoid)

optimizer = torch.optim.Adam(net.parameters(), lr=LR, weight_decay=5e-3)
loss_func = torch.nn.BCELoss()
start_time = time.time()

def train(model, device, federated_train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(federated_train_loader):
        model.send(data.location)
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = loss_func(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.get()
        if batch_idx % 10 == 0:
            loss = loss.get()
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * 128, len(federated_train_loader) * 128,
                       100. * batch_idx / len(federated_train_loader), loss.item()))


for epoch in range(epochs):
    train(net, device, train_federated_dataloader, optimizer, epoch)
    
def test(model):
    model.eval()
    test_loss = 0
    for data, target in federated_test_loader:
        output = model(data)
        test_loss += F.mse_loss(output.view(-1), target, reduction='sum').item()
        pred = output.data.max(1, keepdim=True)[1]
        
    test_loss /= len(federated_test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}\n'.format(test_loss))
    
 test(net)
