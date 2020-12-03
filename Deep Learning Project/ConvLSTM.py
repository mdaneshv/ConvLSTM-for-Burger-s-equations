"""
Created on Mon Oct 7 2019

@author: Mohammad Daneshvar
"""

import numpy as np
from sklearn import preprocessing
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Model, Sequential
from keras.layers import GRU, LSTM, Dense, Input, Activation
from keras.layers import Dropout, BatchNormalization, MaxPooling3D
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras import optimizers
from keras import backend as K
import seaborn as sns
np.random.seed(123)


# create datasets for ConvLSTM2d
def ConvLSTM_dataset(original_dataset, train_size,
                     time_steps, rows, columns):
  
    (nfeatures, m) = original_dataset.shape 
    # preprocess data
    for i in range(nfeatures):
        original_dataset[i, :] = preprocessing.scale(original_dataset[i, :])

    num_img = int(m/columns)    # number of images to be created from original dataset
    
    # start converting dataset into 2D-images
    Z = np.zeros((num_img, rows, columns))  
    for i in range(num_img):
        Z[i, :, :] = original_dataset[:, i * columns:i * columns + columns]  
    image_dataset = np.transpose(Z)   # Now we have a new dataset with 2D images

    # create time series data from 2D-images
    Znew = {}  
    for i in range(time_steps):
        Znew[i] = image_dataset[:, :, i:num_img - (time_steps - i - 1)]

    X = Znew[0]
    for i in range(time_steps - 1):
        X = np.vstack([X, Znew[i + 1]])

    X = np.transpose(X)    # new time series dataset of images
    Y = Z[time_steps:, :, :]    # target values for X

    # create train and test sets and corresponding target values    
    test_size = num_img - train_size - time_steps
    X_train = X[:train_size, :]
    Y_train = Y[:train_size, :]
    X_test = X[train_size:train_size + test_size, :]
    Y_test = Y[train_size:train_size + test_size, :]

    # reshape input and output to be fed into ConvLastm2D layers: input must be 5 dimensional 
    Xtrain_set = X_train.reshape((X_train.shape[0], time_steps, rows, columns, 1))
    Xtest_set = X_test.reshape((X_test.shape[0], time_steps, rows, columns, 1))
    Ytrain_set = Y_train.reshape((Y_train.shape[0], rows, columns, 1))
    Ytest_set = Y_test.reshape((Y_test.shape[0], rows, columns, 1))
  
    return Xtrain_set, Ytrain_set, Xtest_set, Ytest_set


# define ConvLSTM2D model
def create_ConvLSTM_layers(X, Y, filters, kernel_size, batch_size,
                           epochs, learning_rate):
        
    (time_steps, rows, columns) = X.shape[1:]
    model = Sequential()
    model.add(ConvLSTM2D(filters=filters, kernel_size=kernel_size,
                         input_shape=(time_steps, rows, columns, 1),
                         padding='same', return_sequences=True))
    model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size=(5,1,1), padding='same'))
    model.add(ConvLSTM2D(filters=1, kernel_size=kernel_size,    
                         input_shape=(time_steps, rows, columns, 1),
                         padding='same', data_format='channels_last'))
    
    adam = optimizers.Adam(lr=learning_rate, beta_1=0.99, beta_2=0.999)
    model.compile(loss='mse', optimizer=adam, metrics=['accuracy'])

    # fit model to data
    history = model.fit(X, Y, epochs=epochs, batch_size=batch_size, verbose=2, shuffle=True)

    return model, history


# sequence to sequence predictions
# prediction starts from element (samples-test_size)*columns from original_data
def prediction(model, X, time_steps, rows, columns):  

    (time_steps, rows, columns) = X.shape[1:]
    Predictions = np.zeros((X.shape[0], rows, columns, 1))
    first_sequence = X[0, :, :, :, :].reshape((1, time_steps, rows, columns, 1))

    # predict the first sequence and store it in Predictions
    Predictions[0, :, :, :] = model.predict(first_sequence)
    
    # now predict the rest    
    for i in range(1, X.shape[0]):
        if i < time_steps:
            ith_sequence = X[i, :, :, :, :].reshape((1, time_steps, rows, columns, 1))
            # replace the last one in sequence with prediction 
            ith_sequence[0, (time_steps - i):time_steps, :, :, :] = Predictions[:i, :, :, :]
            # now predict the new sequence 
            Predictions[i, :, :, :] = model.predict(ith_sequence)
        else:
            ith_sequence = Predictions[i - time_steps:i, :, :, :].reshape((1, time_steps, rows, columns, 1))
            Predictions[i, :, :, :] = model.predict(ith_sequence)

    return Predictions


# plots
def make_plots(Predictions, Ytest_set, pred_steps):
    Ypred = np.zeros((rows, pred_steps*columns))
    Ytest = np.zeros((rows, pred_steps*columns))
    for i in range(rows):
        for j in range(pred_steps):
            Ypred[i,j*columns:j*columns+columns] =  Predictions[j, i, :, 0]
            Ytest[i,j*columns:j*columns+columns] =  Ytest_set[j, i, :, 0]    
    for i in range(rows):
        plot1, = plt.plot(Ytest[i,:])
        plot2, = plt.plot(Ypred[i,:])
        plt.xlabel('prediction horizon', fontsize=12)
        plt.ylabel('$x_{}$'.format(i), fontsize=14)
        plt.title('Prediction for feature $x_{}$'.format(i), fontsize=14)
        plt.legend([plot1, plot2], ["true_values", "prediction"])
        plt.savefig('predction for %i' % i)
        plt.tight_layout()
        plt.show()

    return Ypred, Ytest, plot1, plot2


# load original dataset (scaler values)
original_data = np.genfromtxt('Burgers.dat', delimiter=' ')  

# parameters
(nfeatures, m) = original_data.shape
time_steps = 2     # recurrent steps
rows = nfeatures      
columns = 50    
train_size = int(m/columns) - 10
pred_steps = 10    # < test size. Prediction horizon will be pred_steps * columns 
filters = 30      
kernel_size = (100,1)
batch_size = 512
epochs = 100
learning_rate = 0.003

Xtrain_set, Ytrain_set, Xtest_set, Ytest_set = ConvLSTM_dataset(original_data, train_size, time_steps,
                                                                rows, columns)
model, history = create_ConvLSTM_layers(Xtrain_set, Ytrain_set, filters, kernel_size, 
                                        batch_size, epochs, learning_rate)
Predictions = prediction(model, Xtest_set, time_steps, rows, columns)
Ypred, Ytest, plot1, plot2 = make_plots(Predictions, Ytest_set, pred_steps)

