import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.sequence import TimeseriesGenerator

import pickle

def make_window_dataset(ds, window_size=5, shift=1, stride=1):
  windows = ds.window(window_size, shift=shift, stride=stride)

  def sub_to_batch(sub):
    return sub.batch(window_size, drop_remainder=True)

  windows = windows.flat_map(sub_to_batch)
  return windows

def dense_7_step(batch):
  # Shift features and labels one step relative to each other.
  return batch[:-7], batch[-7:]

def predictions(y):

    # dates to be predicted
    dates = pd.date_range(start=y.index[-1],periods=8)[1:]

    # normalize data with MinMaxScaler
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(y)
    
    max_ospiti = y["Ospiti"].max()
    min_ospiti = y["Ospiti"].min()
    
    tf_dataset = tf.data.Dataset.from_tensor_slices(scaled_data)
    ds = make_window_dataset(tf_dataset, window_size=37, shift = 1, stride=1)

    dense_labels_ds = ds.map(dense_7_step)

    #create a tensors for holding the labels and features
    all_inputs = []
    all_labels = []

    for inputs,labels in dense_labels_ds:
      all_inputs.append(inputs)
      all_labels.append(labels)
   
    X = tf.concat(all_inputs, axis=0)
    X = tf.reshape(X, (-1, 30, 29))

    Y = tf.concat(all_labels, axis=0)
    Y = tf.reshape(Y, (-1, 7, 29))
    Y = Y[:,:,0]

    # build a forecasting model using the Keras Sequential API
    model = keras.Sequential(
      [
          layers.Flatten(input_shape=(30, 29)),
          layers.Dense(870, activation="relu", name="layer0"),
          # add dropout to prevent overfitting
          layers.Dropout(0.2),
          layers.Dense(435, activation="relu", name="layer1"),
          layers.Dropout(0.2),
          layers.Dense(128, activation="relu", name="layer2"),
          layers.Dropout(0.2),
          layers.Dense(64, activation="relu", name="layer3"),
          layers.Dropout(0.2),
          layers.Dense(7, name="layer4"),
      ]
    )

    model.compile(optimizer='adam', loss="mse")

    model.fit(x=X, y=Y ,epochs=500, verbose=0)

    y_predict = scaled_data.iloc[-30:]
    y_predict_scaled = scaler.fit_transform(y_predict)
    y_predict_scaled = y_predict_scaled.reshape(1,30,29)
    
    # predict the next 7 days
    pred = model.predict(y_predict_scaled)

    # inverse transform the predicted values
    pred_unscaled = pred*(max_ospiti - min_ospiti) + min_ospiti
    pred_unscaled
    
    results = pd.DataFrame(pred_unscaled[0], columns=["Dense NN forecasts"], index=dates)
    
    return results

