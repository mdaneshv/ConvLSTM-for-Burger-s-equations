

  
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras import optimizers
from keras.layers.convolutional_recurrent import ConvLSTM2D



# def custom_activation(x):
#     return(K.tanh(x)*-0.02)
# 
# get_custom_objects().update({'custom_activation':Activation(custom_activation)})    


X= np.genfromtxt('Useries.dat', dtype = np.float64, delimiter=' ')





def ConvLSTM_dataset(Origin_data, time_steps ,rows, columns):
    
    features=Origin_data.shape[0]
    m=Origin_data.shape[1]
    samples=int(m/columns)-1
    train_size=samples-4000
    test_size=4000
    
    for i in range (features):


       Origin_data[i,:]=preprocessing.scale(Origin_data[i,:])
    
    
    

    Z=np.zeros((samples,rows,columns)) # transform original data into 2D images

    for i in range(samples):
    
        
        Z[i,:,:]= Origin_data[:,i*columns:i*columns+columns]
        
    data = np.transpose(Z)    


    
  
    
    Zcut = {} # Creating a seuqnce of data 
    
    for i in range(time_steps):    
        Zcut[i] = data[:,:,i:samples-(time_steps-i-1)]

    Xnew = Zcut[0] 
    for i in range(time_steps-1):
        Xnew = np.vstack([Xnew,Zcut[i+1]])

    Xnew = np.transpose(Xnew)
    Y = Z[time_steps:,:,:]
    
   
    

    Xnew_train = Xnew[:train_size,:]    
    Ynew_train = Y[:train_size,:]
    Xnew_test = Xnew[train_size:train_size+test_size,:]
    Xnew_test = np.delete(Xnew_test, (test_size-time_steps), axis=0)
    Ynew_test = Y[train_size:train_size+test_size,:]
    
    
    

    #reshape input and output to be fed into ConvLastm2D layers 
    Xtrain_set = Xnew_train.reshape((Xnew_train.shape[0], time_steps,rows,columns,1))
    Xtest_set = Xnew_test.reshape((Xnew_test.shape[0], time_steps,rows,columns,1))
    Ytrain_set = Ynew_train.reshape((Ynew_train.shape[0],rows,columns,1))
    Ytest_set = Ynew_test.reshape((Ynew_test.shape[0],rows,columns,1))


    return Xtrain_set,Ytrain_set,Xtest_set,Ytest_set




def create_ConvLSTM_layers(X,Y,batch_size):
    
    time_steps,rows,columns=np.shape(X)[1],np.shape(X)[2],np.shape(X)[3]
    
    
    model = Sequential()
    
    model.add(ConvLSTM2D(filters=20, kernel_size=(20,3),
                   input_shape=(time_steps, rows, columns,1),
                   padding='same', return_sequences=True))
    
    model.add(ConvLSTM2D(filters=20, kernel_size=(20, 3),
                    input_shape=(time_steps, rows, columns,1),
                   padding='same', return_sequences=True))
     
    model.add(ConvLSTM2D(filters=20, kernel_size=(20, 3),
                    input_shape=(time_steps, rows, columns,1),
                    padding='same', return_sequences=True))
                          
    model.add(ConvLSTM2D(filters=1, kernel_size=(20, 3),
                    input_shape=(time_steps, rows, columns,1),
                    padding='same', data_format='channels_last'))                      


    
    adam=optimizers.Adam(lr=0.003,beta_1=0.99, beta_2=0.999)
    
    model.compile(loss='mse', optimizer=adam, metrics=['accuracy'])
    
    
    history = model.fit(X, Y, epochs=1, batch_size=batch_size, verbose=2, shuffle=True)
    
    return model,history




def prediction(model,X,time_steps, rows, columns): # sequence to sequence predictions
    
    time_steps,rows,columns=np.shape(X)[1],np.shape(X)[2],np.shape(X)[3]
    
    
    Predictions = np.zeros((X.shape[0],rows,columns,1))
       
    first_sequence = X[0,:,:,:,:].reshape((1,time_steps,rows,columns,1))
    Predictions[0,:,:,:] = model.predict(first_sequence) 
     
    for i in range(1,X.shape[0]):        
        if i < time_steps:
            ith_sequence = X[i,:,:,:,:].reshape((1,time_steps,rows,columns,1))
            ith_sequence[0,(time_steps-i):time_steps,:,:,:] = Predictions[:i,:,:,:]
            Predictions[i,:,:,:] = model.predict(ith_sequence)
        else:
            ith_sequence = Predictions[i-time_steps:i,:,:,:].reshape((1,time_steps,rows,columns,1))
            Predictions[i,:,:,:] = model.predict(ith_sequence)
           
    return Predictions




def make_plots(Predictions,Ytest_set, pred_steps):
   
    
   
   Ypred={}
   Ytest={}
   for i in range(rows):
     Ypred[i]=np.empty((1))
     Ytest[i]=np.empty((1))
   
   for i in range(rows):
       for j in range(0,pred_steps):
    
          Ypred[i]=np.append(Ypred[i],Predictions[j,i,:,0])
          Ytest[i]=np.append(Ytest[i],Ytest_set[j,i,:,0])

       Ypred[i] = np.delete(Ypred[i], 0, axis=0)
       Ytest[i] = np.delete(Ytest[i], 0, axis=0)
       
   for i in range(rows):
 
    plot1,=plt.plot(Ytest[i])
        
    plot2,=plt.plot(Ypred[i])
        
    plt.xlabel('steps')
        
    plt.ylabel('$X[%i]$' %i)
        
    plt.title('Prediction for $X[%i]$' %i, fontsize=10)
        
    plt.legend([plot1,plot2],["true_values","prediction"])
        
    plt.savefig('predction for %i'%i)
        
    plt.show()
    


   return Ypred,Ytest,plot1,plot2


time_steps=5
rows=16
columns=10
batch_size=32
pred_steps=10




Xtrain_set,Ytrain_set,Xtest_set,Ytest_set = ConvLSTM_dataset(X, time_steps ,rows, columns)
model,history = create_ConvLSTM_layers(Xtrain_set,Ytrain_set,batch_size)


Predictions = prediction(model,Xtest_set,time_steps, rows, columns)

Ypred,Ytest,plot1,plot2=make_plots(Predictions,Ytest_set,pred_steps)



       







