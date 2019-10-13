from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

# Inicial loading
dataset = pd.read_pickle('/PATH/filename.pkl')
labels = np.load('/PATH/filename_Labels.pkl', allow_pickle = True)
X = dataset.iloc[:, :31].values
y = labels.values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# Feature Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


#define the model
def build_regressor():
  regressor = Sequential()
  regressor.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 31))
  regressor.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
  regressor.add(Dense(units = 1, kernel_initializer = 'uniform'))
  regressor.compile(optimizer = 'adam', loss = 'mse', metrics = ['accuracy'])

  return regressor

# evaluate model
estimator = KerasRegressor(build_fn = build_regressor, epochs = 5, batch_size = 5, verbose = 0)
kfold = KFold(n_splits = 2)
results = cross_val_score(estimator, X_train, y_train, cv = kfold)
print("Baseline: %.2f (%.2f) MSE" % (results.mean(), results.std()))