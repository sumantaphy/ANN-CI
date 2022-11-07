from numpy import vstack
import math
from pandas import read_csv
#from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score,mean_squared_error
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torch import Tensor
import torch
from torch.nn import Sigmoid,Softmax,ReLU,Linear,Tanh
from torch.nn import Module
from torch.optim import SGD,Adam
from torch.nn import BCELoss,NLLLoss,CrossEntropyLoss,MSELoss
from torch.nn.init import kaiming_uniform_,xavier_uniform_
from sklearn.model_selection import train_test_split
from ann_setup import readInput

fr_test,n_inputs,H, H_s, batch, l_rate, input_file, inp_list, det, det_sign, epoch_a, epoch_s = readInput()


f1 = open("output_Amplitude_train.out","w")
f2 = open("output_Amplitude_test.out","w")
f_sign1 = open("output_sign_train.out","w")
f_sign2 = open("output_sign_test.out","w")


# dataset loading
class CSVDataset(Dataset):
    def __init__(self, path):
        df  = read_csv(path,usecols=inp_list, header=None)
        df_sign = read_csv(path,usecols=[det_sign], header=None)
        df_det = read_csv(path,usecols=[det], header=None)


        self.X = df.values[:, :-1]
        self.y = df.values[:, -1]
        self.sign = df_sign.values[:,-1]
        self.det = df_det.values[:,-1]

        self.X = self.X.astype('float32')
        self.y = self.y.astype('float32')
        self.sign = self.sign.astype('float32')
        self.det = self.det.astype('float32')

        self.y = self.y.reshape((len(self.y), 1))
        self.sign = self.sign.reshape((len(self.sign), 1))
        self.det = self.det.reshape((len(self.det), 1))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx],self.sign[idx], self.det[idx]]

    def get_splits(self, n_test=fr_test):                              # spliting of dataset 
        test_size = int(round(n_test * len(self.X)))
        train_size = int(len(self.X) - test_size)
        return random_split(self, [train_size, test_size])
############################################################################################################
#self defined non-linear activation function
class Cos(Module):
    def forward(self, input):
        return (torch.cos(input))
class Sigmoid_custom(Module):
    def forward(self, input):
        return ((1500**0.5)*torch.sigmoid(input))

############################################################# NETWORK MODEL #################################
# model definition
class Network(Module):
    def __init__(self,n_inputs):
        super(Network,self).__init__()
        
        #input descriptor
        self.hidden1 = Linear(n_inputs, H)                           # input to 1st hidden layer
        kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        self.relu = ReLU()
        # Output layer
        self.output = Linear(H, 1)                                  # 2nd hidden layer to output
        xavier_uniform_(self.output.weight)
        # Define sigmoid activation and softmax output 
        self.relu = ReLU()
    def forward(self, X):
        X = self.hidden1(X)            
        X = self.relu(X)
        X = self.output(X)
        X = self.relu(X)


        return X

class sign_Network(Module):                                         # This is for the sign part
    def __init__(self,n_inputs):
        super(sign_Network,self).__init__()

           #input descriptor
        # Inputs to hidden layer linear transformation
        self.hidden1 = Linear(n_inputs, H_s)
        self.cos = Cos()
        self.hidden2 = Linear(H_s,H_s)
        self.tanh = Tanh()
        # Output layer
        self.output = Linear(H_s, 1)

        # Define sigmoid activation and softmax output 
        self.sigmoid = Sigmoid()

    def forward(self, X):
        # Pass the input tensor through each of our operations
        X = self.hidden1(X)
        X = self.cos(X)
        X = self.hidden2(X)
        X = self.tanh(X)
        X = self.output(X)
        X = self.sigmoid(X)

        return X


#########################################################################################


# prepare the dataset
def prepare_data(path):
    dataset = CSVDataset(path)
    train, test = dataset.get_splits()
    train_dl = DataLoader(train, batch_size=batch, shuffle=False)
    test_dl = DataLoader(test, batch_size=batch, shuffle=False)
    return train_dl, test_dl
######################################### validation ############################
def validation(test_dl, model):                                          #send the test dataset through the network
    predictions, actuals = list(), list()
    for i, (inputs, targets,sign_targets,dets) in enumerate(test_dl):
        yhat = model(inputs)
        yhat = yhat.detach().numpy()
        actual = targets.numpy()
        actual = actual.reshape((len(actual), 1))
        predictions.append(yhat)
        actuals.append(actual)
    predictions, actuals = vstack(predictions), vstack(actuals)
    # calculate accuracy
    acc = mean_squared_error(actuals, predictions)
    return acc                                                         

################################### TRAINING ########################################################
def train_model(train_dl, test_dl, model):
    criterion = MSELoss()                                               #loss function
    optimizer = Adam(model.parameters(), lr=l_rate, betas=(0.9, 0.999))  #optimizer should be used
    for epoch in range(epoch_a):
        running_loss = 0.0
        for i, (inputs, targets,sign_targets,dets) in enumerate(train_dl):
            optimizer.zero_grad()
            yhat = model(inputs)
            loss = criterion(yhat, targets)
            loss.backward()
            optimizer.step()
            running_loss +=loss.item()
        epoch_loss = running_loss/len(train_dl)
        validation_error = validation(test_dl, model)
        print(str(epoch)+"  "+str(epoch_loss)+"   "+str(validation_error))
    print('Finished Training')

def train_sign_model(train_dl, sign_model):
    #criterion = MSELoss()
    criterion = BCELoss()
    optimizer = Adam(sign_model.parameters(), lr=l_rate, betas=(0.9, 0.999))
    for epoch in range(epoch_s):
        for i, (inputs, targets,sign_targets,dets) in enumerate(train_dl):
            optimizer.zero_grad()
            yhat = sign_model(inputs)
            loss = criterion(yhat, sign_targets)
            loss.backward()
            optimizer.step()
            #print ("loss.item",loss.item())
######################################### MODEL EVALUATION / Testing ##############################################################

def evaluate_model(test_dl, model):                                          #send the test dataset through the network
    predictions, actuals = list(), list()
    for i, (inputs, targets,sign_targets,dets) in enumerate(test_dl):
        yhat = model(inputs)
        yhat = yhat.detach().numpy()
        actual = targets.numpy()
        det = dets.numpy()
        for j in range(len(actual)):
            f2.write(str(int(det[j][0]))+"     "+str(10**(actual[j][0]*-1))+"   "+str(10**(yhat[j][0]*-1))+"\n")
    return True




def evaluate_trainmodel(train_dl, model):                                     #send the train dataset through the network
    predictions, actuals = list(), list()
    for i, (inputs, targets,sign_targets,dets) in enumerate(train_dl):
        yhat = model(inputs)
        yhat = yhat.detach().numpy()
        actual = targets.numpy()
        det = dets.numpy()
        for j in range(len(actual)):
                f1.write(str(int(det[j][0]))+"     "+str(10**(actual[j][0]*-1))+"   "+str(10**(yhat[j][0]*-1))+"\n")
    return True
################################################### SIGN MODEL EVALUATION ###################################################
def evaluate_sign_model(test_dl, sign_model):
    predictions, actuals = list(), list()
    for i, (inputs, targets,sign_targets,dets) in enumerate(test_dl):
        yhat = sign_model(inputs)
        yhat = yhat.detach().numpy()
        actual = sign_targets.numpy()
        yhat = yhat.round()
        det = dets.numpy()
        for j in range(len(actual)):
            f_sign2.write(str(int(det[j][0]))+"     "+str(actual[j][0])+"   "+str(yhat[j][0])+"\n")
        actual = actual.reshape((len(actual), 1))
        predictions.append(yhat)
        actuals.append(actual)
    predictions, actuals = vstack(predictions), vstack(actuals)
    acc = accuracy_score(actuals, predictions)
    # calculate accuracy
    return acc
def evaluate_sign_trainmodel(train_dl, sign_model):
    predictions, actuals = list(), list()
    for i, (inputs, targets,sign_targets,dets) in enumerate(train_dl):
        yhat = sign_model(inputs)
        yhat = yhat.detach().numpy()
        actual = sign_targets.numpy()
        yhat = yhat.round()
        det = dets.numpy()
        for j in range(len(actual)):
            f_sign1.write(str(int(det[j][0]))+"     "+str(actual[j][0])+"   "+str(yhat[j][0])+"\n")
        actual = actual.reshape((len(actual), 1))
        predictions.append(yhat)
        actuals.append(actual)
    predictions, actuals = vstack(predictions), vstack(actuals)
    acc = accuracy_score(actuals, predictions)
    return acc

###############################################################################################################
# prepare the data
path = input_file
train_dl, test_dl = prepare_data(path)

# define the two networks
model = Network(n_inputs)
sign_model = sign_Network(n_inputs)

# train the models
train_model(train_dl, test_dl, model)
torch.save(model.state_dict(), "model.pth")

train_sign_model (train_dl , sign_model)
torch.save(sign_model.state_dict(), "sign_model.pth")

# evaluate the model
evaluate_model(test_dl, model)
evaluate_trainmodel(train_dl , model)

acc3 = evaluate_sign_model(test_dl,sign_model)
acc4 = evaluate_sign_trainmodel(train_dl,sign_model)

f_sign1.write("#Train accuarcy= "+"       "+str(acc4)+"\n")
f_sign2.write("#Test accuarcy= "+"       "+str(acc3)+"\n")

f1.close()
f2.close()
f_sign1.close()
f_sign2.close()
