# Importing the Keras libraries and packages
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense


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

# Initialising the ANN
regressor = Sequential()

# Adding the input layer and the first hidden layer
regressor.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 31))

# Adding the second hidden layer
regressor.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
regressor.add(Dense(units = 1, kernel_initializer = 'uniform'))

# Compiling the ANN
regressor.compile(optimizer = 'adam', loss = 'mse')

# Fitting the ANN to the Training set
regressor.fit(X_train, y_train, batch_size = 10, epochs = 3)


# evaluating the performence
eval_model=regressor.evaluate(X_test, y_test)


# Predicting the result
y_pred = regressor.predict(X_test)


# Visualising the Test set results
plt.scatter(y_test, y_pred, color = 'blue')
plt.plot(np.arange(0,200), np.arange(0,200))
plt.title('(Regression Model)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
