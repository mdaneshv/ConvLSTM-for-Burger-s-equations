"""
Created on Mon Oct 7 2019

@author: Mohammad Daneshvar
"""

import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import GRU, LSTM, ConvLSTM2D
from tensorflow.keras.layers import Dense, Input, Activation, Dropout, BatchNormalization, MaxPooling3D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from keras import backend as K
np.random.seed(123)


# Fuction to create datasets
# for the ConvLSTM2d model
def create_dataset(original_dataset,
                   lookback, n_cols):
  n_features, n_samples = original_dataset.shape

  # data preprocessing
  for i in range(n_features):
      original_dataset[i, :] = preprocessing.scale(original_dataset[i, :])

  # Number of 2D-images(matrices)
  # to be created from the original dataset
  n_images = int(n_samples/n_cols)  
  print('n_images = ', n_images)  

  # Converting data into 2D-images(matrices).
  # Number of rows in the matrices equals 
  # the number of features in the original dataset
  Z = np.zeros((n_images, n_features, n_cols))  
  for i in range(n_images):
      Z[i, :] = original_dataset[:, i*n_cols:(i+1)*n_cols]  

   # The new dataset containing images    
  image_dataset = np.transpose(Z)  

  # Creating time-series data from images
  Znew = {}  
  for i in range(lookback):
      Znew[i] = image_dataset[:, :, i:n_images - (lookback - i - 1)]

  X = Znew[0]
  for i in range(lookback - 1):
      X = np.vstack([X, Znew[i + 1]])

  # A time-series dataset from images
  X = np.transpose(X) 
  # Target values for X
  Y = Z[lookback:, :]   

  # Train and test sets and the corresponding target values   
  train_size = int(0.8 * n_images)  
  test_size = n_images - train_size - lookback
  Xtrain = X[:train_size, :]
  Ytrain = Y[:train_size, :]
  Xtest = X[train_size:train_size + test_size, :]
  Ytest = Y[train_size:train_size + test_size, :]

  # Reshape the input and output to be fed into ConvLastm2D layers
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
def build_ConvLSTM_model(Xtrain):
  
  # A functional model 
  input_layer = Input(shape=Xtrain.shape[1:])
  x = ConvLSTM2D(filters=64, kernel_size=4, padding='same', return_sequences=True)(input_layer)
  x = BatchNormalization()(x)
  x = MaxPooling3D(pool_size=(5,1,1), padding='same')(x)
  output = ConvLSTM2D(filters=1, kernel_size=4, padding='same', data_format='channels_last')(x)
  model = Model(inputs=input_layer, outputs=output)
  return model
  
def compile_model(model, Xtrain, Ytrain):
  
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
  
  # Remove the dimension of size 1 (the channels)
  Xtest = np.squeeze(Xtest)

  (lookback, n_features, n_cols) = Xtest.shape[1:]
  Predictions = np.zeros((Xtest.shape[0], n_features, n_cols, 1))
  first_sequence = Xtest[0, :].reshape((1, lookback, n_features, n_cols, 1))

  # Predict the first sequence and store it in Predictions
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




