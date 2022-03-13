#!/usr/bin/env python3
import tensorflow as tf
from tensorflow import keras
from keras import layers
import keras_tuner as kt
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import argparse

class HyperModel:

    def __init__(self, path, label):
        self.provide_dataset(path, label)

    def provide_dataset(self, path, label):
        # get dataset
        self._dataset = pd.read_csv(path)
        self._dataset = self._dataset.sample(frac=1)
        self._dataset_features = self._dataset.copy()
        self._dataset_labels = self._dataset_features.pop(label)

        # normalize dataset (MinMaxScale)
        self._features_max = self._dataset_features.max()
        self._labels_max = self._dataset_labels.max()
        self._dataset_features /= self._features_max
        self._dataset_labels /= self._dataset_labels.max()

        # split dataset train (2/3) test (1/3)
        self._x, self._x_test, self._y, self._y_test = train_test_split(self._dataset_features, self._dataset_labels, test_size=0.33)

    def build_model(self, hp):
        self._model = keras.Sequential()
        self._model.add(keras.layers.Flatten())
        for i in range(hp.Int("num_layers", 1, 5)):
            self._model.add(
                keras.layers.Dense(
                    units=hp.Int(f"units_{i}", min_value=8, max_value=256, step=8),
                    activation=hp.Choice("activation", ["relu", "tanh", "sigmoid"])
                )
            )
        
        self._model.add(keras.layers.Dropout(rate=hp.Choice("dr", [0.25, 0.5])))
        self._model.add(keras.layers.Dense(1))
        learning_rate = hp.Float(
            "lr", 
            min_value=1e-4, 
            max_value=1e-2, 
            sampling="log"
        )
        self._model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
            loss="mean_squared_error",
            metrics=[keras.metrics.MeanSquaredError()]
        )

        return self._model

    def get_best_hyperparams(self):
        self._tuner = kt.BayesianOptimization(
            hypermodel = self.build_model,
            objective="mean_squared_error",
            max_trials=10,
            overwrite=True,
            directory="my_tuner",
            project_name="feature_degradation",
        )
        self._es = keras.callbacks.EarlyStopping(
            monitor="mean_squared_error",
            patience=5,
            restore_best_weights=True
        )
        self._tuner.search_space_summary()
        self._x_train, self._x_val, self._y_train, self._y_val = train_test_split(self._x, self._y, test_size=0.2)
        self._tuner.search(
            self._x_train, 
            self._y_train, 
            epochs=40, 
            validation_data=(self._x_val, self._y_val), 
            callbacks=[self._es]
        )
        self._tuner.results_summary()
        self._best_hps = self._tuner.get_best_hyperparameters(5)

        return self._best_hps[0]
    
    def plot_the_loss_curve(self, epochs, mse):
        plt.figure()
        plt.xlabel("Epoch")
        plt.ylabel("Mean Squared Error")

        plt.plot(epochs, mse, label="Loss")
        plt.legend()
        plt.ylim([mse.min()*0.95, mse.max() * 1.03])
        plt.savefig("mse_per_epoch.png")

    def main(self):
        model = self.build_model(self.get_best_hyperparams())
        history = model.fit(
            self._x_train, 
            self._y_train, 
            batch_size=5, 
            epochs=30, 
            validation_data=(self._x_val, self._y_val)
        )

        epochs = history.epoch

        hist = pd.DataFrame(history.history)
        mse = hist["mean_squared_error"]

        self.plot_the_loss_curve(epochs, mse)
        keras.utils.plot_model(model, "model.png", show_shapes=True)

        print("Evaluate on test data")
        results = model.evaluate(
            self._x_test, 
            self._y_test, 
            batch_size=5
        )
        print("test loss, test msqe:", results)

        print("Generate predictions for 3 samples")
        test = self._x_test[:3]
        predictions = model.predict(test)
        print("Testing on:\n",test)
        print("Predictions:\n", predictions*self._labels_max)
        print("real values \n", np.array([self._dataset['PERF'][test.index]]).transpose())
        model.save("generated_model", overwrite=True)
        with open('generated_model/max_y.txt', 'w') as f:
            f.write('%d' % self._labels_max)
            f.close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="This program either generates a model to predict the provided label or uses an exisiting model to predict the passed input"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-p", "--path", action="store", dest="csv_path", type=str, help="Path to csv file")
    group.add_argument("-c", "--conf", action="store", dest="conf", type=str, help="The configuration to use as input for the model")
    parser.add_argument("-l", "--label", action="store", dest="label", type=str, help="The label to predict")
    parser.add_argument("-mp", "--model-path", action="store", dest="model_path", type=str, help="Path to the saved model")

    args = parser.parse_args()
    
    if args.csv_path and args.label is None:
        parser.error("-p requires -l")
    elif args.csv_path and args.label:
        hyper_model = HyperModel(args.csv_path, args.label)
        hyper_model.main()
    elif args.conf and args.model_path is None:
        parser.error("-c requires -mp")
    elif args.conf and args.model_path:
        model = keras.models.load_model(args.model_path)
        x_pred = np.array([list(map(int,args.conf.split()))])
        y_pred = model.predict(x=x_pred)
        with open(f'{args.model_path}/max_y.txt', 'r') as f:
            y_max = float(f.readlines()[-1])
        print(y_pred*y_max)