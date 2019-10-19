import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras import optimizers
from sklearn.metrics import mean_squared_error as mse


# Inicial loading 
dataset = pd.read_pickle('/PATH/filename.pkl')
labels = np.load('/PATH/filename_Labels.pkl', allow_pickle=True)
X = dataset.iloc[:, :31].values
y = labels.values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Example:

# Normally, X has the form 
# array([[19., 15., 10., ..., 24.,  9., 17.],
#        [15., 10., 16., ...,  9., 17., 15.],
#        [10., 16., 14., ..., 17., 15., 17.],
#        ...,
#        [68., 76., 73., ..., 41., 63., 59.],
#        [76., 73., 66., ..., 63., 59., 74.],
#        [73., 66., 49., ..., 59., 74., 62.]])
# CNN ALWAYS requiers 3D shape of data. We need to transfrom X to have each element as a independent array:

X  = X.reshape(-1, 31, 1)

# array([[[19.],
#         [15.],
#         [10.],
#         ...,
#         [24.],
#         [ 9.],
#         [17.]],

#        [[15.],
#         [10.],
#         [16.],
#         ...,
#         [ 9.],
#         [17.],
#         [15.]],
#         ...
    
    
# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# fit and evaluate a model
def evaluate_model(X_train, y_train, X_test, y_test):
    verbose, epochs, batch_size = 2, 1, 32
    n_timesteps, n_features, n_outputs = X_train.shape[1], X_train.shape[2], X_train.shape[1]
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(n_timesteps,n_features)))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    # fit network
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose)
    # evaluate model
    _, accuracy = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=0)
    # get the prediction
    y_pred = model.predict(X_test)
    return (y_pred, accuracy, model)


# summarize scores
def summarize_results(scores):
    print(scores)
    m, s = mean(scores), std(scores)
    print('Accuracy: %.3f%% (+/-%.3f)' % (m, s))
    
    
def run_experiment(repeats=10):
    # load data
    X_train, y_train, X_test, y_test = load_dataset()
    # repeat experiment
    scores = list()
    for r in range(repeats):
        score = evaluate_model(X_train, y_train, X_test, y_test)[1]
        score = score * 100.0
        print('>#%d: %.3f' % (r+1, score))
        scores.append(score)
    # summarize results
    summarize_results(scores)
    

# Make a prediction    
y_pred = evaluate_model(X_train, y_train, X_test, y_test)[0]



# Visualising the Test results
plt.scatter(y_test, y_pred, color = 'blue')
plt.plot(np.arange(0,200), np.arange(0,200))
plt.title('CNN Model')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()