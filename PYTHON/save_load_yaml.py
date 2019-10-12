from keras.models import model_from_yaml

# Save NN model to yaml

# serialize model named regressor to YAML
model_yaml = regressor.to_yaml()
with open("model.yaml", "w") as yaml_file:
  yaml_file.write(model_yaml)
# serialize weights to HDF5
regressor.save_weights("model.h5")
print("Saved model ANN to disk")

# load YAML and create model
yaml_file = open('model.yaml', 'r')
loaded_model_yaml = yaml_file.read()
yaml_file.close()
loaded_model = model_from_yaml(loaded_model_yaml)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")


# evaluate loaded model on test data
loaded_model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
score = loaded_model.evaluate(X_test, y_test, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))
