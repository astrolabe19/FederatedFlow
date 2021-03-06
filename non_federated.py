import torch
from torch.autograd import Variable
import pandas as pd
import numpy as np
from sklearn import metrics, preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
# from sklearn.cross_validation import StratifiedKFold, ShuffleSplit, cross_val_score, train_test_split
import logging
import math
import os
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import time
from sklearn.preprocessing import PolynomialFeatures
import scipy
from pylab import rcParams
import warnings

warnings.filterwarnings('ignore')

lgr = logging.getLogger(__name__)
use_cuda = torch.cuda.is_available()
# use_cuda = False

FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
Tensor = FloatTensor

lgr.info("USE CUDA=" + str (use_cuda))
seed=17*19
np.random.seed(seed)
torch.manual_seed(seed)
if use_cuda:
    torch.cuda.manual_seed(seed)

df = pd.read_csv('/Users/ast/Desktop/labelled_1.csv')
df.drop(df.columns[[14, 15]], axis=1, inplace = True)
labels = df[df.columns[-1]]

labels = labels.map({ 'BENIGN' : 0, 'SSH-Patator' : 1, 'FTP-Patator' : 1})
# print(labels.describe())
# print(labels.value_counts())
features = df[df.columns[:-1]]
X_train, X_test, Y_train, Y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

min_max_scaler = preprocessing.MinMaxScaler()
N_FEATURES = X_train.shape[1]
print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)
numerical_feature = X_train.dtypes[X_train.dtypes != 'object'].index
categorical_feature = X_train.dtypes[X_train.dtypes == 'object'].index

print("There are {} numeric and {} categorical columns in train data".format(numerical_feature.shape[0],categorical_feature.shape[0]))
# correlation matrix
# corr = X_train[numerical_feature].corr()
# plt.imshow(corr, cmap='hot', interpolation='nearest')
# plt.show()

def XnumpyToTensor(x_data_np):
    x_data_np = np.array(x_data_np, dtype=np.float32)
    print(x_data_np.shape)
    print(type(x_data_np))

    if use_cuda:
        lgr.info("Using the GPU")
        X_tensor = Variable(torch.from_numpy(x_data_np).cuda())  # Note the conversion for pytorch
    else:
        lgr.info("Using the CPU")
        X_tensor = Variable(torch.from_numpy(x_data_np))  # Note the conversion for pytorch

    print(type(X_tensor.data))  # should be 'torch.cuda.FloatTensor'
    print(x_data_np.shape)
    print(type(x_data_np))
    return X_tensor


# Convert the np arrays into the correct dimention and type
# Note that BCEloss requires Float in X as well as in y
def YnumpyToTensor(y_data_np):
    y_data_np = y_data_np.values.reshape((y_data_np.values.shape[0], 1))  # Must be reshaped for PyTorch!
    print(y_data_np.shape)
    print(type(y_data_np))

    if use_cuda:
        lgr.info("Using the GPU")
        #     Y = Variable(torch.from_numpy(y_data_np).type(torch.LongTensor).cuda())
        Y_tensor = Variable(torch.from_numpy(y_data_np)).type(
            torch.FloatTensor).cuda()  # BCEloss requires Float
    else:
        lgr.info("Using the CPU")
        #     Y = Variable(torch.squeeze (torch.from_numpy(y_data_np).type(torch.LongTensor)))  #
        Y_tensor = Variable(torch.from_numpy(y_data_np)).type(torch.FloatTensor)  # BCEloss requires Float

    print(type(Y_tensor.data))  # should be 'torch.cuda.FloatTensor'
    print(y_data_np.shape)
    print(type(y_data_np))
    return Y_tensor

# NN params
DROPOUT_PROB = 0.90

LR = 0.005
MOMENTUM= 0.9
dropout = torch.nn.Dropout(p=1 - (DROPOUT_PROB))

lgr.info(dropout)

hiddenLayer1Size=512
hiddenLayer2Size=int(hiddenLayer1Size/4)
hiddenLayer3Size=int(hiddenLayer1Size/8)
hiddenLayer4Size=int(hiddenLayer1Size/16)
hiddenLayer5Size=int(hiddenLayer1Size/32)

linear1=torch.nn.Linear(N_FEATURES, hiddenLayer1Size, bias=True)
torch.nn.init.xavier_uniform(linear1.weight)

linear2=torch.nn.Linear(hiddenLayer1Size, hiddenLayer2Size)
torch.nn.init.xavier_uniform(linear2.weight)

linear6=torch.nn.Linear(hiddenLayer2Size, 1)
torch.nn.init.xavier_uniform(linear6.weight)

sigmoid = torch.nn.Sigmoid()
tanh=torch.nn.Tanh()
relu=torch.nn.LeakyReLU()

net = torch.nn.Sequential(linear1,nn.BatchNorm1d(hiddenLayer1Size),relu,
                          linear2,dropout,relu,
                          linear6,sigmoid
                          )
lgr.info(net)

optimizer = torch.optim.Adam(net.parameters(), lr=LR,weight_decay=5e-3)
loss_func=torch.nn.BCELoss()

start_time = time.time()
epochs= 60 # change to 1500 for better results
all_losses = []

X_tensor_train= XnumpyToTensor(X_train)
Y_tensor_train= YnumpyToTensor(Y_train)
print(type(X_tensor_train.data), type(Y_tensor_train.data))

for step in range(epochs):
    out = net(X_tensor_train)
    cost = loss_func(out, Y_tensor_train)
    optimizer.zero_grad()  # clear gradients for next train
    cost.backward()  # backpropagation, compute gradients
    optimizer.step()
    if step % 5 == 0:
        loss = cost.data.item
        all_losses.append(loss)
        print(step, cost.data.cpu().numpy())
        prediction = (net(X_tensor_train).data).float()
        pred_y = prediction.cpu().numpy().squeeze()
        target_y = Y_tensor_train.cpu().data.numpy()
        tu = (metrics.log_loss(target_y, pred_y), metrics.roc_auc_score(target_y, pred_y))
        print('LOG_LOSS={}, ROC_AUC={} '.format(*tu))

end_time = time.time()
print ('{} {:6.3f} seconds'.format('GPU:', end_time-start_time))

import matplotlib.pyplot as plt
plt.plot(all_losses)
plt.show()

false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(target_y,pred_y)
roc_auc = metrics.auc(false_positive_rate, true_positive_rate)

plt.title('LOG_LOSS=' + str(metrics.log_loss(target_y, pred_y)))
plt.plot(false_positive_rate, true_positive_rate, 'b', label='AUC = %0.6f' % roc_auc)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([-0.1, 1.2])
plt.ylim([-0.1, 1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
