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
Stacked LSTM:

Multiple hidden LSTM layers can be stacked one on top of another in what is referred to as a Stacked LSTM model.

An LSTM layer requires a three-dimensional input and LSTMs by default will produce a two-dimensional output as an interpretation from the end of the sequence.

We can address this by having the LSTM output a value for each time step in the input data by setting the return_sequences=True argument on the layer. This allows us to have 3D output from hidden LSTM layer as input to the next.

We can therefore define a Stacked LSTM as follows.

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
model = Sequential()
model.add(LSTM(31, activation='relu', return_sequences=True, input_shape=(n_steps, n_features)))
model.add(LSTM(31, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')


# fit model
model.fit(X, y, epochs=200, verbose=0)


n_steps = 31 #number of columns
n_features = 1
# demonstrate prediction
x_input = array([70, 80, 90])
x_input = x_input.reshape((1, n_steps, n_features))
yhat = model.predict(x_input, verbose=0)
print(yhat)


# Visualising the Test set results
plt.scatter(y_test, yhat, color = 'blue')
plt.plot(np.arange(0,200), np.arange(0,200))
plt.title('(Regression Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
