# mnist_data_classification-using-ANN-and-CNN

In this post i am going to implement a simple python code for mnist data classification using ANN and CNN model in keras.

keras is a high level nueral network API, capable of running over tensorflow and theano. Keras allows easy and fast implementation. it supports deep neural networks and very much suitable for building deep neural network models.
 
 Lets start the step by step implementation by importing important libraries
 ```import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
import numpy as np
 `````
     
Next step is to load tha data. Here the data is imported from keras.dataset. Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
MNIST datset cosists of 28x28 size images of handwritten digits.We will load pre-shuffled data into training and testing sets.

```
(X_train, y_train) , (X_test, y_test) = keras.datasets.mnist.load_data()
print(len(X_train))
print(len(X_test))
````


We can plot and see any of the image from dataset image(let's say the fifth image)
```
plt.imshow(X_train[5])
plt.show()
````
![5](https://user-images.githubusercontent.com/58771064/98289933-ec1d4f80-1fce-11eb-9e5c-98ff71ee8eae.png)




Now we need to reshape our data set inorder to apply for ANN model. As the ANN model receive data data in vector form so the data is converted into vector form.
```
feature_vector_length = 784
X_train = X_train.reshape(X_train.shape[0], feature_vector_length)
X_test = X_test.reshape(X_test.shape[0], feature_vector_length)
input_shape = (feature_vector_length,)
X_train.shape

`````

Now we can build the model. The model type that we will be using is Sequential. Sequential is the easiest way to build a model in Keras. It allows you to build a model layer by layer.We use the ‘add()’ function that adds layers to our neural network. Dense specifies fully connected layers.Activation Function is used for introducing non-linearity

```
model = Sequential()
model.add(Dense(350, input_shape=input_shape, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(10, activation='softmax'))

`````


Now we need to specify the loss function and the optimizer, that is done by compile function available in keras. Here loss is cross entropy. Categorical cross-entropy states that we have multiple classes. The optimizer used is Adam. Metrics is used to specify how we want to judge our models performance. 

```
model.compile(optimizer='adam',
             loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

`````



Training step is simple in keras. model.fit function is used to train the model.

```
model.fit(X_train, y_train, epochs=10)

`````

Now lets see for CNN model. All the preprocessing taks will be sames as ANN, but difference is that CNN receive data in image form(2-D) form.so data is reshaped into 2-D form
```
img_width, img_height = 28, 28
X_train = X_train.reshape(X_train.shape[0],  img_width, img_height,1)
X_test = X_test.reshape(X_test.shape[0], img_width, img_height,1)
input_shape = ( img_width, img_height,1,)
X_train.shape

````
The model type that we will be using is Sequential. Sequential is the easiest way to build a model in Keras. It allows you to build a model layer by layer.We use the ‘add()’ function that adds layers to our neural network. The first 2 layers are Conv2D layers. These are convolution layers that will deal with 2-D inputs.Dense specifies fully connected layers. Dropout is a regularization technique used for reducing overfitting by randomly dropping output units in the network.Activation Function is used for  non-linearity and Flatten is used for converting output 2-D data into vector from.

```
odel = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(10, activation='softmax'))

````

Now we need to specify the loss function and the optimizer, that is done by compile function available in keras. Here loss is cross entropy. Categorical cross-entropy states that we have multiple classes. The optimizer used is Adam. Metrics is used to specify how we want to judge our models performance. Training step is simple in keras. model.fit function is used to train the model. 

```
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10)
              
              `````
              
              
              
              
              
              
              
             
              
              
          
              
              







              















 
 
 
 

 
 
