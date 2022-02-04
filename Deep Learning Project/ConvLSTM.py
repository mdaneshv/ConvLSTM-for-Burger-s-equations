"""
Created on Mon Oct 7 2019

@author: Mohammad Daneshvar
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import GRU, LSTM, ConvLSTM2D
from tensorflow.keras.layers import Dense, Input, Activation
from tensorflow.keras.layers import Dropout, BatchNormalization, MaxPooling3D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from keras import backend as K
np.random.seed(123)


# Standardize data
# In the original dataset the rows represent the features
# and the columns represent different samples
def preprocess_data(original_dataset):
  scaler = StandardScaler()
  original_dataset = scaler.fit_transform(original_dataset.T)
  original_dataset = original_dataset.T
  return original_dataset


# Fuction to create images from the 
# original dataset and then create a 
# time-series dataset from the images
def create_dataset(original_dataset,
                   lookback, n_cols):
  n_features, n_samples = original_dataset.shape

  # Number of 2D-images(matrices)
  # to be created from the original dataset
  n_images = int(n_samples/n_cols)  
  print('n_images = ', n_images)  

  # Converting data into 2D-images(matrices).
  # Number of rows in the matrices equals 
  # the number of features in the original dataset
  images = np.zeros((n_images, n_features, n_cols))  
  for i in range(n_images):
      images[i, :] = original_dataset[:, i*n_cols:(i+1)*n_cols]  

 # Target values for the time-series images 
  Y = images[lookback:, :]   

  # Creating time-series data from images 
  images = np.transpose(images)  
  Z = {}  
  for i in range(lookback):
      Z[i] = images[:, :, i:n_images - (lookback - i - 1)]
  
  # Time-series data from images   
  X = Z[0]
  for i in range(lookback - 1):
      X = np.vstack([X, Z[i + 1]]) 
  X = np.transpose(X)   

  # Train and test sets and
  # the corresponding target values   
  train_size = int(0.8 * n_images)  
  test_size = n_images - train_size - lookback
  Xtrain = X[:train_size, :]
  Ytrain = Y[:train_size, :]
  Xtest = X[train_size:train_size + test_size, :]
  Ytest = Y[train_size:train_size + test_size, :]

  # Reshape the input and output to be fed
  # into ConvLSTM layers
  # Input must be 5 dimensional 
  Xtrain = Xtrain.reshape((Xtrain.shape[0], lookback, n_features, n_cols, 1))
  Xtest = Xtest.reshape((Xtest.shape[0], lookback, n_features, n_cols, 1))
  Ytrain = Ytrain.reshape((Ytrain.shape[0], n_features, n_cols, 1))
  Ytest = Ytest.reshape((Ytest.shape[0], n_features, n_cols, 1))

  print('n_rows =', n_features)
  print('n_columns =', n_cols)
  print('lookback =', lookback, '\n')
  print('Xtrain =', Xtrain.shape)
  print('Ytrain =', Ytrain.shape)
  print('Xtest =', Xtest.shape)
  print('Ytest =', Ytest.shape)

  return Xtrain, Ytrain, Xtest, Ytest


# Build the ConvLSTM model
# Using a functional model
def build_ConvLSTM_model(Xtrain):
  input_layer = Input(shape=Xtrain.shape[1:])
  x = ConvLSTM2D(filters=64, kernel_size=4, padding='same', return_sequences=True)(input_layer)
  x = BatchNormalization()(x)
  x = MaxPooling3D(pool_size=(5,1,1), padding='same')(x)
  output = ConvLSTM2D(filters=1, kernel_size=4, padding='same', data_format='channels_last')(x)
  model = Model(inputs=input_layer, outputs=output)
  return model

  
def compile_and_fit_model(model, Xtrain, Ytrain):
  
  # Callback for the early stopping
  callback = EarlyStopping(monitor='loss', patience=5)
  
  adam = Adam(learning_rate=0.001, beta_1=0.99, beta_2=0.999)
  model.compile(loss='mse', optimizer=adam, metrics=['RootMeanSquaredError'])

  # Fit model to the data
  history = model.fit(Xtrain, Ytrain, epochs=1, batch_size=64,
                      callbacks = [callback], verbose=2, shuffle=True)

  return history


# A function for sequence to sequence predictions
def prediction(model, Xtest): 
  
  # Remove the dimension of size 1
  Xtest = np.squeeze(Xtest)

  (lookback, n_features, n_cols) = Xtest.shape[1:]
  Predictions = np.zeros((Xtest.shape[0], n_features, n_cols, 1))
  first_sequence = Xtest[0, :].reshape((1, lookback, n_features, n_cols, 1))

  # Predict the first sequence and
  # store it in Predictions
  Predictions[0, :] = model.predict(first_sequence)

  # Now predict the rest    
  for i in range(1, Xtest.shape[0]):
      if i < lookback:
          ith_sequence = Xtest[i, :].reshape((1, lookback, n_features, n_cols, 1))

          # Replace the last one in the sequence with prediction 
          ith_sequence[0, (lookback - i):lookback, :] = Predictions[:i, :]

          # Now predict the new sequence 
          Predictions[i, :] = model.predict(ith_sequence)
      else:
          ith_sequence = Predictions[i - lookback:i, :].reshape((1, lookback, n_features, n_cols, 1))
          Predictions[i, :] = model.predict(ith_sequence)

  return Predictions




