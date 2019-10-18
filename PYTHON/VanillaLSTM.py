import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import LSTM
from sklearn.metrics import mean_squared_error as mse

    
    
'''
Vanilla LSTM:

A Vanilla LSTM is an LSTM model that has a single hidden layer of LSTM units, and an output layer used to make a prediction.

We can define a Vanilla LSTM for univariate time series forecasting as follows.

'''
# Inicial loading 
dataset = pd.read_pickle('/PATH/filename.pkl')
labels = np.load('/PATH/filename_Labels.pkl', allow_pickle=True)
X = dataset.iloc[:, :31].values
y = labels.values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    
    
# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# define model
n_steps = 31 #number of columns
model = Sequential()
model.add(LSTM(31, activation='relu', input_shape=(n_steps, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
print(model.summary())


'''
Key in the definition is the shape of the input; that is what the model expects as input for each sample in terms of the number of time steps and the number of features.

We are working with a univariate series, so the number of features is one, for one variable.

The number of time steps as input is the number we chose when preparing our dataset as an argument to the split_sequence() function.

The shape of the input for each sample is specified in the input_shape argument on the definition of first hidden layer.

We almost always have multiple samples, therefore, the model will expect the input component of training data to have the dimensions or shape:

[samples, timesteps, features]

Our split_sequence() function in the previous section outputs the X with the shape [samples, timesteps], so we easily reshape it to have an additional dimension for the one feature.
'''

# reshape from [samples, timesteps] into [samples, timesteps, features]
n_features = 1
X = X.reshape((X.shape[0], X.shape[1], n_features))


'''
In this case, we define a model with 50 LSTM units in the hidden layer and an output layer that predicts a single numerical value.

The model is fit using the efficient Adam version of stochastic gradient descent and optimized using the mean squared error, or ‘mse‘ loss function.

Once the model is defined, we can fit it on the training dataset.
'''

# fit model
model.fit(X, y, epochs=200, verbose=0)


# demonstrate prediction
y_pred = model.predict(X_test, verbose=0)
print(y_pred)

# Visualising the Test set results
plt.scatter(y_test, yhat, color = 'blue')
plt.plot(np.arange(0,200), np.arange(0,200))
plt.title('(Regression Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()