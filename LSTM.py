import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Activation, BatchNormalization
from keras import optimizers
from keras.regularizers import l2
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects
import seaborn as sns

np.random.seed(0)


# Define a custom activation function
def custom_activation(x):
    return (K.tanh(x) * -0.02)

get_custom_objects().update({'custom_activation': Activation(custom_activation)})


# read csv data
data_df = pd.read_csv('GlobalTemperatures.csv')
data_df.head()

# We use the first two columns as our data (Average Temperature and its uncertainty)
data = data_df[['LandAverageTemperature', 'LandAverageTemperatureUncertainty']]

print('data =', data.shape)

# To see how many null values exist in data
data.isnull().sum()

# Remove null values
dataset = data.dropna()

# Number of features: here are Temperature and Uncertainty
features =dataset.shape[1]



# Normalizing data
'''
for i in range (features):

   dataset.values[:,i]=preprocessing.scale(dataset.values[:,i])
'''


# create new data sets for LSTM model
def LSTM_datasets(dataset, train_size, test_size, time_steps):
    samples = train_size + test_size
    features = dataset.shape[1]
    new_data = np.transpose(dataset.values)[:, :samples]
    
    # Creating a sequance of data 
    Xcut = {}

    for i in range(time_steps):
        Xcut[i] = new_data[:, i:samples - (time_steps - i - 1)]

    X = Xcut[0]
    for i in range(time_steps - 1):
        X = np.vstack([X, Xcut[i + 1]])
        
    # new data set and corresponding target values
    X = np.transpose(X)
    Y = np.transpose(new_data[:, time_steps:samples])
    
    # Creating train and test set
    X_train = X[:train_size, :]
    Y_train = Y[:train_size, :]
   
    X_test = X[train_size:train_size + test_size:, :]
    X_test = np.delete(X_test, (test_size - time_steps), axis=0)
    Y_test = Y[train_size:train_size + test_size:, :]

    # Reshape train and test set into 3-dimensional arrays for Keras layers
    X_train_set = X_train.reshape((X_train.shape[0], time_steps, features))
    X_test_set = X_test.reshape((X_test.shape[0], time_steps, features))
    print("Xtrain shape = ", X_train_set.shape, "Ytrain shape = ", Y_train.shape)
    print("Xtest shape =  ", X_test_set.shape, " Ytest shape =  ", Y_test.shape)

    return X_train_set, Y_train, X_test_set, Y_test, features


# Create LSTM architecture
def LSTM_model(X_train_set, Y_train_set, features, num_hidden, batch_size):
    model = Sequential()

    model.add(LSTM(num_hidden, input_shape=(X_train_set.shape[1], X_train_set.shape[2]),
                   activation='tanh', recurrent_activation='sigmoid', kernel_initializer='random_normal',
                   recurrent_initializer='random_normal', kernel_regularizer=None,
                   recurrent_regularizer=l2(l=0.0), activity_regularizer=None, dropout=0, recurrent_dropout=0,
                   return_sequences=True, stateful=False))

    model.add(LSTM(num_hidden, dropout=0, recurrent_dropout=0, return_sequences=False, stateful=False))

    # model.add(BatchNormalization())

    model.add(Dense(features, activation=None))

    # model.add(Activation(custom_activation, name='SpecialActivation'))

    adam = optimizers.Adam(lr=0.003, beta_1=0.99, beta_2=0.999)
    
    model.compile(loss='mse', optimizer=adam, metrics=['accuracy'])
    
    history = model.fit(X_train_set, Y_train_set, epochs=50, batch_size=batch_size, verbose=2, shuffle=True)

    return model, history


# sequence to sequence prediction
def Prediction(model, X_test_set, time_steps):
    Y_prediction = np.zeros((X_test_set.shape[0], features))

    for i in range(X_test_set.shape[0]):
        if i == 0:
            tt = X_test_set[0, :, :].reshape((1, time_steps, features))
            Y_prediction[i, :] = model.predict(tt)
        elif i < time_steps:
            tt = X_test_set[i, :, :].reshape((1, time_steps, features))
            u = Y_prediction[:i, :]
            tt[0, (time_steps - i):time_steps, :] = u
            Y_prediction[i, :] = model.predict(tt)
        else:
            tt = Y_prediction[i - time_steps:i, :].reshape((1, time_steps, features))
            Y_prediction[i, :] = model.predict(tt)
    return Y_prediction


train_size = 3000
test_size = 153 
time_steps = 3
num_hidden = 50
batch_size = 32

X_train_set, Y_train, X_test_set, Y_test, features = LSTM_datasets(dataset, train_size, test_size,time_steps)                                                                 
model, history = LSTM_model(X_train_set, Y_train, features, num_hidden, batch_size)
Y_prediction = Prediction(model, X_test_set, time_steps)

# save predictions
np.savetxt('Y_prediction.csv', Y_prediction, delimiter=',')
np.savetxt('Y_test.csv', Y_test, delimiter=',')

# Plots
for i in range(features):
    plot1, = plt.plot(Y_test[:, i])

    plot2, = plt.plot(Y_prediction[:, i])

    plt.xlabel('prediction horizon')

    plt.ylabel('x[%i]' % i)

    plt.title('prediction for feature x[%i] using LSTM' % i)

    plt.legend([plot1, plot2], ["true_values", "Prediction"])

    plt.savefig('%i' % i)

    plt.show()

    
    
# PDF plot
sns.distplot(Y_test[:, 0], kde=True, hist=False, kde_kws={"label": "true_values"})
sns.distplot(Y_prediction[:, 0], kde=True, hist=False, kde_kws={"label": "predictions"})
plt.title(' distribution plot:  Temperatures')
plt.xlabel('values')
plt.savefig('dist0')
plt.show()

sns.distplot(Y_test[:, 1], kde=True, hist=False, kde_kws={"label": "true_values"})
sns.distplot(Y_prediction[:, 1], kde=True, hist=False, kde_kws={"label": "predictions"})
plt.title(' distribution plot:  Uncertainty')
plt.xlabel('values')
plt.savefig('dist5')
plt.show()

# Plot heatmap for predictions and true values
sns.heatmap(Y_test[:, 0].reshape((30, 5)))
plt.title('true_values: Temperatures')
plt.show()
sns.heatmap(Y_prediction[:, 0].reshape((30, 5)))
plt.title('prediction: Temperatures')
plt.show()

sns.heatmap(Y_test[:, 1].reshape((30, 5)))
plt.title('true_values: Uncertainty')
plt.show()
sns.heatmap(Y_prediction[:, 1].reshape((30, 5)))
plt.title('prediction: Uncertainty')
plt.show()
