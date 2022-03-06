import tensorflow as tf
from tensorflow import keras
from keras import layers
import keras_tuner as kt
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

# get dataset
dataset = pd.read_csv("Data/LLVM_AllNumeric.csv")
dataset = dataset.sample(frac=1)
dataset_features = dataset.copy()
dataset_labels = dataset_features.pop('PERF')

# normalize dataset (MinMaxScale)
features_max = dataset_features.max()
labels_max = dataset_labels.max()
dataset_features /= features_max
dataset_labels /= dataset_labels.max()

# split dataset train (2/3) test (1/3)
x, x_test, y, y_test = train_test_split(dataset_features, dataset_labels, test_size=0.33)

def build_model(hp):
    model = keras.Sequential()
    model.add(keras.layers.Flatten())
    for i in range(hp.Int("num_layers", 1, 11)):
        model.add(
            keras.layers.Dense(
                units=hp.Int(f"units_{i}", min_value=8, max_value=256, step=8),
                activation=hp.Choice("activation", ["relu", "tanh", "sigmoid"])
            )
        )
    if hp.Boolean("dropout"):
        model.add(keras.layers.Dropout(rate=hp.Choice("dr", [0.25, 0.5])))
    model.add(keras.layers.Dense(1))
    learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-1, sampling="log")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mean_squared_error",
        metrics=[keras.metrics.MeanSquaredError()]
    )
    return model

tuner = kt.BayesianOptimization(
    hypermodel = build_model,
    objective="mean_squared_error",
    max_trials=10,
    overwrite=True,
    directory="my_tuner",
    project_name="feature_degradation",
)

es = keras.callbacks.EarlyStopping(
    monitor="mean_squared_error",
    patience=5,
    restore_best_weights=True
)

tuner.search_space_summary()

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2)

tuner.search(x_train, y_train, epochs=10, validation_data=(x_val, y_val), callbacks=[es])

tuner.results_summary()

best_hps = tuner.get_best_hyperparameters(5)
model = build_model(best_hps[0])
history = model.fit(x_train, y_train, batch_size=20, epochs=10, validation_data=(x_val, y_val),)

keras.utils.plot_model(model, "model.png", show_shapes=True)

print("Evaluate on test data")
results = model.evaluate(x_test, y_test, batch_size=5)
print("test loss, test msqe:", results)

print("Generate predictions for 3 samples")
test = x_test[:3]
print(test, predictions*labels_max)
predictions = model.predict(test)


