import numpy as np
from sklearn import preprocessing
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Model, Sequential
from keras.layers import GRU, LSTM, Dense, Input, Activation, Dropout
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras import optimizers
from keras import backend as K
import seaborn as sns
import pandas.util.testing as tm
from urllib.request import urlretrieve



Origin_data = np.genfromtxt('Burgers.dat', delimiter=' ')

for i in range(16):
        Origin_data[i, :] = preprocessing.scale(Origin_data[i, :])


# Create datasets for ConvLSTM2d
def ConvLSTM_dataset(dataset, time_window, rows, columns):
    features = dataset.shape[0] 
    m = dataset.shape[1] 
    samples = int(m / columns)   # number of images 
    test_size = 10
    train_size = samples - test_size - time_window
    

    # start converting dataset into 2D-images
    Z = np.zeros((samples, rows, columns))  
    for i in range(samples):
        Z[i, :, :] = dataset[:, i * columns:i * columns + columns] # We cut dataset into different parts: 2D-images

    image_data = np.transpose(Z)

    # Creating a sequence of data from 2D-images we have created above
    Znew = {}  
    for i in range(time_window):
        Znew[i] = image_data[:, :, i:samples - (time_window - i - 1)]

    X = Znew[0]
    for i in range(time_window - 1):
        X = np.vstack([X, Znew[i + 1]])

    X = np.transpose(X)
    Y = Z[time_window:, :, :]  # target values for X

    # creating train and test sets and corresponding target values from X and Y
              
    X_train = X[:train_size, :]
    Y_train = Y[:train_size, :]
    X_test = X[train_size:train_size + test_size, :]
    Y_test = Y[train_size:train_size + test_size, :]

    # reshape input and output to be fed into ConvLastm2D layers: input must be 5 dimensional 
    Xtrain_set = X_train.reshape((X_train.shape[0], time_window, rows, columns, 1))
    Xtest_set = X_test.reshape((X_test.shape[0], time_window, rows, columns, 1))
    Ytrain_set = Y_train.reshape((Y_train.shape[0], rows, columns, 1))
    Ytest_set = Y_test.reshape((Y_test.shape[0], rows, columns, 1))
  
    return Xtrain_set, Ytrain_set, Xtest_set, Ytest_set

# Define ConvLSTM2D model
def create_ConvLSTM_layers(X, Y, filters, kernel_size, batch_size, epochs, learning_rate):
    time_window, rows, columns = np.shape(X)[1], np.shape(X)[2], np.shape(X)[3]

    model = Sequential()

    model.add(ConvLSTM2D(filters=filters, kernel_size=kernel_size,
                         input_shape=(time_window, rows, columns, 1),
                         padding='same', return_sequences=True))
    
    
    # last layer has 1 filter
    model.add(ConvLSTM2D(filters=1, kernel_size=kernel_size,    
                         input_shape=(time_window, rows, columns, 1),
                         padding='same', data_format='channels_last'))
    

    adam = optimizers.Adam(lr=learning_rate, beta_1=0.99, beta_2=0.999)

    model.compile(loss='mse', optimizer=adam, metrics=['accuracy'])

    # fit model to the data
    history = model.fit(X, Y, epochs=epochs, batch_size=batch_size, verbose=2, shuffle=True)

    return model, history


# sequence to sequence predictions
def prediction(model, X, time_window, rows, columns):  

    time_window, rows, columns = np.shape(X)[1], np.shape(X)[2], np.shape(X)[3]

    Predictions = np.zeros((X.shape[0], rows, columns, 1))

    first_sequence = X[0, :, :, :, :].reshape((1, time_window, rows, columns, 1))
    # predict the first sequence and store it in Predictions
    Predictions[0, :, :, :] = model.predict(first_sequence)

    for i in range(1, X.shape[0]):
        if i < time_window:
            ith_sequence = X[i, :, :, :, :].reshape((1, time_window, rows, columns, 1))
            # replace the last one in sequence with prediction 
            ith_sequence[0, (time_window - i):time_window, :, :, :] = Predictions[:i, :, :, :]
            # now predict the new sequence 
            Predictions[i, :, :, :] = model.predict(ith_sequence)
        else:
            ith_sequence = Predictions[i - time_window:i, :, :, :].reshape((1, time_window, rows, columns, 1))
            Predictions[i, :, :, :] = model.predict(ith_sequence)

    return Predictions

# Paste columns of Predictions and true values to create 2 vectors for plotting
def make_plots(Predictions,Ytest_set, pred_steps):
    Ypred = np.zeros((rows,pred_steps*columns))
    Ytest = np.zeros((rows,pred_steps*columns))

    for i in range(rows):
        for j in range(pred_steps):
            Ypred[i,j*columns:j*columns+columns] =  Predictions[j, i, :, 0]
            Ytest[i,j*columns:j*columns+columns] =  Ytest_set[j, i, :, 0]
         

    # plots  
    for i in range(rows):
        plot1, = plt.plot(Ytest[i,:])

        plot2, = plt.plot(Ypred[i,:])

        plt.xlabel('prediction horizon')

        plt.ylabel('$x[%i]$' % i)

        plt.title('Prediction for feature $x[%i]$ using ConvLSTM' % i, fontsize=10)

        plt.legend([plot1, plot2], ["true_values", "prediction"])

        plt.savefig('predction for %i' % i)

        plt.show()

    return Ypred, Ytest, plot1, plot2


time_window = 2    # recurrent steps or tiem-steps
rows = 16          # I chose it to be the same as number of features
columns = 50       # columns of images 
pred_steps = 10    # prediction horizon shown on x_axis of plots which is : pred_steps * columns 
filters = 30       # number of filters in convolutional layres
kernel_size = (100,1)
batch_size = 512
epochs = 100
learning_rate = 0.003


#dataset = encoder_decoder(np.transpose(Origin_data), code_size)
Xtrain_set, Ytrain_set, Xtest_set, Ytest_set = ConvLSTM_dataset(Origin_data, time_window, rows, columns)
model, history = create_ConvLSTM_layers(Xtrain_set, Ytrain_set, filters, kernel_size, batch_size, epochs, learning_rate)
Predictions = prediction(model, Xtest_set, time_window, rows, columns)
Ypred, Ytest, plot1, plot2 = make_plots(Predictions, Ytest_set, pred_steps)

### prediction starts from element (samples-test_size)*columns from Origin_data
