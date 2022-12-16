import pandas as pd
import numpy as np
import tensorflow as tf

import pickle

def predictions(y):
    # load model from pickle file
    model = pickle.load(open('models/my_dense_model.pkl', 'rb'))
    # load scaler from pickle file
    scaler = pickle.load(open('models/scaler.pkl', 'rb'))
    
    # scale the data
    y = scaler.transform(y.values.reshape(-1, 1))

    # dates to be predicted
    dates = pd.date_range(start=y["Data"].iloc[-1],periods=8)[1:]

    # predict the next 7 days
    pred = model.predict(y)
    pred_unscaled = scaler.inverse_transform(pred)

    # construct results dataframe
    results = pd.DataFrame(pred_unscaled[0], columns=["Ospiti"], index=dates)
    
    return results

