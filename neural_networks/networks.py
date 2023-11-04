# %%
import numpy as np 
import pandas as pd
import sys
import pdb
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt 
from sklearn.preprocessing import OneHotEncoder

def get_data(x,y):

    y = y.astype('float')
    x = x.astype('float')

    
    x = 2*(0.5 - x/255)
    return x, y


class NeuralNetwork:
    def __init__(self, input_size, hidden_layers, output_size):
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.weights, self.biases = self.initialize_weights_and_biases()
        
    def initialize_weights_and_biases(self):
        layers = [self.input_size] + self.hidden_layers + [self.output_size]
        weights = [np.random.randn(layers[i], layers[i+1]) for i in range(len(layers)-1)]
        biases = [np.zeros((1, layers[i+1])) for i in range(len(layers)-1)]
        return weights, biases
    
    def forward(self, x):
        z_values, activations = [], [x]
        for i in range(len(self.hidden_layers)):
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            a = self.sigmoid(z)
            z_values.append(z)
            activations.append(a)
        z_output = np.dot(activations[-1], self.weights[-1]) + self.biases[-1]
        a_output = self.softmax(z_output)
        z_values.append(z_output)
        activations.append(a_output)
        return z_values, activations
    
    def backward(self, x, y, learning_rate):
        z_values, activations = self.forward(x)
        dL_dy = activations[-1] - y
        for i in range(len(self.hidden_layers), 0, -1):
            # dL_dz = grad_z  
            # activations[i] is "a" of i th layer  
            dL_dz = dL_dy.dot(self.weights[i].T) * activations[i] * (1 - activations[i]) 
            
            self.weights[i] -= learning_rate * activations[i].T.dot(dL_dy) 
            self.biases[i] -= learning_rate * np.sum(dL_dy, axis=0, keepdims=True)
            dL_dy = dL_dz
        x_new= x.reshape(-1,1)
        self.weights[0] -= learning_rate * x_new.dot(dL_dy)
        self.biases[0] -= learning_rate * np.sum(dL_dy, axis=0, keepdims=True)
    
    def train(self, x_train, y_train, num_epochs, batch_size, learning_rate):
        for epoch in range(num_epochs):
            for i in range(0, len(x_train), batch_size):
                x_batch = x_train[i:i+batch_size]
                y_batch = y_train[i:i+batch_size]
                
                for j in range(len(x_batch)):
                    self.backward(x_batch[j], y_batch[j], learning_rate)
            # pred_y= self.predict2(x_train)
            # print(pred_y)
            # loss_obt= self.com_loss(pred_y, y_train)
            # print(loss_obt)
            print(f"Epoch {epoch+1}/{num_epochs} completed.")
    
    def softmax(self,x):
        tmp= np.exp(x- np.max(x,axis=1, keepdims= True))
        return tmp/np.sum(tmp)

    def sigmoid(self,x):
        return 1/(1+np.exp(-x))
    
    def predict(self, x):
        m,n= x.shape
        pred_y=[]
        for i in range(m):
            _, op= self.forward(x[i])
            predictions =op[-1] 
            y_pred= np.argmax(predictions, axis=1)
            pred_y.append(y_pred[0]+1)
        np.array(pred_y)
        return pred_y
    def predict2(self, x):
        m,n= x.shape
        pred_y=[]
        for i in range(m):
            _, op= self.forward(x[i])
            predictions =op[-1] 
            pred_y.append(predictions)
        pred_y= np.array(pred_y)
        return pred_y
    def com_loss(self,y_pred, y_true):
        
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

        loss = -np.sum(y_true * np.log(y_pred)) / len(y_true)

        return loss



# %%
input_size = 1024
hidden_layers = [100]  
output_size = 5 
x = np.load(r'C:\Users\HARSH GARG\Desktop\Assignment3\ass3_part_b (1)\part b\x_train.npy')
y = np.load(r'C:\Users\HARSH GARG\Desktop\Assignment3\ass3_part_b (1)\part b\y_train.npy')
x_train, y_train = get_data(x, y)

# print(y_train)
#you might need one hot encoded y in part a,b,c,d,e
label_encoder = OneHotEncoder(sparse_output = False)
label_encoder.fit(np.expand_dims(y_train, axis = -1))
y_train_onehot = label_encoder.transform(np.expand_dims(y_train, axis = -1))
y_test_onehot = label_encoder.transform(np.expand_dims(y_test, axis = -1))
print(x_train.shape)

nn = NeuralNetwork(input_size, hidden_layers, output_size)
num_epochs = 100
batch_size = 32
learning_rate = 0.01
nn.train(x_train, y_train_onehot, num_epochs, batch_size, learning_rate)



# %%
from sklearn.metrics import precision_score, recall_score, f1_score


# %%
y_pred= nn.predict(x_train)
classification = (y_pred == y_train)*1
accuracy = np.sum(classification)/len(classification)
print(accuracy)

# %%
x_test_path = np.load(r'C:\Users\HARSH GARG\Desktop\Assignment3\ass3_part_b (1)\part b\data_test\x_test.npy')
y_test_path = np.load(r'C:\Users\HARSH GARG\Desktop\Assignment3\ass3_part_b (1)\part b\data_test\y_test.npy')
X_test, y_test = get_data(x_test_path, y_test_path)

# %%
y_pred_test= nn.predict(X_test)
classification = (y_pred_test == y_test)*1
accuracy = np.sum(classification)/len(classification)
print(accuracy)

# %%

f1 = f1_score(y_train, y_pred, average=None )
precision = precision_score(y_train, y_pred, average=None)
recall = recall_score(y_train, y_pred, average=None)
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

# %% [markdown]
# Adaptive learning

# %%
eta0= 0.01
f1_scores_train = []
f1_scores_test = []
nn2= NeuralNetwork(input_size, hidden_layers, output_size)

for epoch in range(num_epochs):
    eta= (eta0)/(np.sqrt(epoch+1))
    nn2.train(x_train, y_train_onehot,1,  batch_size, eta)
    y_pred2= nn2.predict(x_train)


    

# %%
classification = (y_pred2 == y_train)*1
accuracy = np.sum(classification)/len(classification)
print(accuracy)

# %%
class NeuralNetwork2:
    def __init__(self, input_size, hidden_layers, output_size):
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.weights, self.biases = self.initialize_weights_and_biases()
        
    def initialize_weights_and_biases(self):
        layers = [self.input_size] + self.hidden_layers + [self.output_size]
        weights = [np.random.randn(layers[i], layers[i+1]) for i in range(len(layers)-1)]
        biases = [np.zeros((1, layers[i+1])) for i in range(len(layers)-1)]
        return weights, biases
    
    def forward(self, x):
        z_values, activations = [], [x]
        for i in range(len(self.hidden_layers)):
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            a = self.relu(z)
            z_values.append(z)
            activations.append(a)
        z_output = np.dot(activations[-1], self.weights[-1]) + self.biases[-1]
        a_output = self.softmax(z_output)
        z_values.append(z_output)
        activations.append(a_output)
        return z_values, activations
    
    def backward(self, x, y, learning_rate):
        z_values, activations = self.forward(x)
        dL_dy = activations[-1] - y
        for i in range(len(self.hidden_layers), 0, -1):
            
            dL_dz = dL_dy.dot(self.weights[i].T) *(activations[i]>0)
            
            self.weights[i] -= learning_rate * activations[i].T.dot(dL_dy)
            self.biases[i] -= learning_rate * np.sum(dL_dy, axis=0, keepdims=True)
            dL_dy = dL_dz
        x_new= x.reshape(-1,1)
        self.weights[0] -= learning_rate * x_new.dot(dL_dy)
        self.biases[0] -= learning_rate * np.sum(dL_dy, axis=0, keepdims=True)
    
    def train(self, x_train, y_train, num_epochs, batch_size, learning_rate):
        for epoch in range(num_epochs):
            for i in range(0, len(x_train), batch_size):
                x_batch = x_train[i:i+batch_size]
                y_batch = y_train[i:i+batch_size]
                
                for j in range(len(x_batch)):
                    self.backward(x_batch[j], y_batch[j], learning_rate)
            print(f"Epoch {epoch+1}/{num_epochs} completed.")
    
    def softmax(self,x):
        tmp= np.exp(x- np.max(x,axis=1, keepdims= True))
        return tmp/np.sum(tmp)

    def relu(self, x):
        return np.maximum(0, x)
    
    def predict(self, x):
        m,n= x.shape
        pred_y=[]
        for i in range(m):
            _, op= self.forward(x[i])
            predictions =op[-1] 
            y_pred= np.argmax(predictions, axis=1)
            pred_y.append(y_pred[0]+1)
        np.array(pred_y)
        return pred_y

# %%
nn3= NeuralNetwork2(input_size, hidden_layers, output_size)
num_epochs = 100
batch_size = 32
learning_rate = 0.01
nn3.train(x_train, y_train_onehot, num_epochs, batch_size, learning_rate)

# %%
y_pred3= nn3.predict(x_train)
classification = (y_pred3 == y_train)*1
accuracy = np.sum(classification)/len(classification)
print(accuracy)

# %%
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
hidden_layers= [512,256,128,64]
mlp = MLPClassifier(hidden_layer_sizes=hidden_layers, activation='relu', solver='sgd',
                    alpha=0, batch_size=32, learning_rate='invscaling', max_iter=50)
mlp.fit(x_train, y_train)
predic_y= mlp.predict(x_train)


# %%
from sklearn.metrics import precision_score, recall_score, f1_score
predic_y= mlp.predict(X_test)
precision = precision_score(y_test, predic_y, average=None)

classification = (predic_y == y_test)*1
accuracy = np.sum(classification)/len(classification)
print(accuracy)
print("Precision:", precision)


# %%
f1 = f1_score(y_test, predic_y, average=None )
precision = precision_score(y_test,predic_y, average=None)
recall = recall_score(y_test,predic_y, average=None)
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")


