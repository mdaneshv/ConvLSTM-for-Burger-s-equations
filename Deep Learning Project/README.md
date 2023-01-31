## Introduction
Data-driven models are models where they use data to analyze specific problems where it’s difficult or sometimes impossible to find a analytic solution for them. They have been widely used in various area of mathematics such as prediction of trajectories of chaotic systems or spatio-temporal models driven by discretization of partial differential equations. 

We can apply different techniques such as recurrent neural networks to predict short-time or reproduce long-term statistics of these systems. In this documentation, I have used experiments and illustated that reshaping data and utilizing a ConvLSTM network can improve the results obtained by LSTM network.

## Inviscid Burgers-Hopf Equation: Data and Model
 The dynamical system that I have explored here is governed by the [inviscid Burgers equations](https://en.wikipedia.org/wiki/Burgers%27_equation): 

$$ \frac{\partial u}{\partial t} + u\frac{\partial u}{\partial x}=0. $$

Data is generated over a periodic domain of size $L$, by discretizing the equation in space using the difference method:

$$\frac{d}{dt}u_i + \frac{F_{i+\frac 12} - F_{i-\frac 12}}{\Delta x},$$

where $0 \leq i \leq N-1, \Delta x = \frac{L}{N}, x_i = i\Delta x, u_i(t)=u(x_i,t)$, and

$$ F_{i+\frac 12} = \frac 16 (u_i^2 + u_i u_{i+1} + u_{i+1}^2),$$

$$ F_{i-\frac 12} = \frac 16 (u_i^2 + u_i u_{i-1} + u_{i-1}^2).$$

Then by local averaging $u_i$ over $n$ neighbor sites, we can define a new smooth variable $x_i$ as


$$ x_i = \frac{1}{n} \sum_{j=ni}^{n(i+1)-1} u_j,$$

where $0 \leq i \leq \frac{N}{n}-1$. See [this paper](https://www.researchgate.net/publication/265481355_Subgrid-scale_closure_for_the_inviscid_Burgers-Hopf_equation) for more details. 

Data is generated using parameters $L = 100, N = 256, n = 16$, and $\Delta t = 0.02$. Therefore, the dataset includes samples of $x_i (t)$ for $0 \leq i \leq 15$ with *step_size* $\Delta t = 0.02$. This is a time-series dataset with $300,000$ sequence samples ( $0 \leq t \leq 6,000$ ) each of which has $16$ features.

First, I used a deep-layer LSTM network to predict (average) trajectories, $x_i(t)$  and long-term statistics of data. The results show that the LSTM network doesn’t do well on this data. I have used a $4$-layer LSTM with *lookback* = $3$ and $50$ hidden units. The train set includes $319500$ samples and the remaining $500$ samples are used in the test set.

The following plots are for features $x_0, x_4, x_5$ and $x_{12}$, where predictions are the worst among all features. In these plots, the x-axis shows the prediction horizon where the *step-size*, $\Delta t$, is $0.02$. Data is normalized before processing.


## LSTM Network Results 

<img width="860" alt="Screenshot 2023-01-31 at 4 20 51 AM" src="https://user-images.githubusercontent.com/58681626/215733625-e68df2c7-6e49-4ea3-a336-62c4ae5bf91e.png">


<img width="904" alt="Screenshot 2023-01-31 at 4 22 46 AM" src="https://user-images.githubusercontent.com/58681626/215733963-ee2b3abc-8e4c-49a4-b3de-c200ae0eb668.png">

## Explanation
LSTM neural networks are powerful on the text prediction where input vectors are one-hot vectors which are sparse matrices. So it might be a good idea if we create sparse matrices from the original data to see how the accuracy on the test-set changes. Moreover, if we make 2D-images or matrices with $c$ columns, where we put sequence of the original data, then we are actually feeding a big sequence of data (a sequence of length c) one at a time instead of only one. So, if LSTM learns these images, then an accurate one-step prediction for images is equivalent to an accurate c-step predictions for the original sequence data. We can increase the number $c$ to get more accurately-predicted steps, however it might reduces the number of images and hence the accuracy on the training set. 

For this task I used a ConvLSTM which I breifly explained here.

## ConvLSTM Neural Network

Convolutional LSTMs (or ConvLSTM) are one special type of LSTM networks where linear combination of input vectors or hidden states are replaced by convolutions. In this network input can be a series of images. They are useful for object detection and motion prediction in autonomous cars. 

The dynamics of the netwrok is as follows:

<img width="524" alt="Screenshot 2023-01-31 at 4 30 59 AM" src="https://user-images.githubusercontent.com/58681626/215735935-5ca9e1b2-48b6-48f6-abdf-63bf9bbbcd19.png">

where ∗ stands for the convolution operator as we have seen in CNN architecture and ⊙ stands for
Hadamard product.

<img width="906" alt="Screenshot 2023-01-31 at 4 33 29 AM" src="https://user-images.githubusercontent.com/58681626/215736442-9236dbf6-3ea7-4b47-8358-c91a6ec4c54e.png">

I have created one-channel images with $16$ rows. This is the number of features, so each row corresponds to one feature. We can change the number of columns but for this experiment I have used $50$ columns for each image (matrix). So the total number of images is $320000/50 = 6400$. This reduces the number of samples from $320000$ in LSTM to $6400$ in ConvLSTM, but still I observed a very low training error. I used one convolutional layer followed by a batch normalization layer and a max-pooling layer. I have used $30$ kernels with kernel size = (100, 1) for the convolutional layer. One important hyperparameter is *lookback* which is chosen empirically. I chose *lookback* = 2, because it provided the best result. I trained the network with $100$ epochs. The last $5$ epoch in training the network, are shown here:

Epoch 95/100
- 9s - loss: 3.7384e-05 Epoch 96/100
- 9s - loss: 3.6531e-05 Epoch 97/100
- 9s - loss: 3.5723e-05 Epoch 98/100
- 9s - loss: 3.4972e-05 Epoch 99/100
- 9s - loss: 3.4267e-05 Epoch 100/100
- 9s - loss: 3.3511e-05

## PLots for ConvLSTM
<img width="940" alt="Screenshot 2023-01-31 at 4 36 27 AM" src="https://user-images.githubusercontent.com/58681626/215737118-35166abf-8eb9-42f9-9568-965a21f906f2.png">
