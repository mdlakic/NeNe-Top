#!/usr/bin/env python
# coding: utf-8

import warnings
warnings.filterwarnings('ignore')
import os
#os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'   # see issue #152 https://github.com/keras-team/keras/issues/152
#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from datetime import datetime
import pandas as pd
import numpy as np
np.random.seed()
import tensorflow as tf
tf.get_logger().setLevel('ERROR')
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error
import joblib
from Bio import SeqIO
import argparse

parser = argparse.ArgumentParser(
    description=
    '\n Neural network prediction of optimal growth temperature from di-peptide composition.\n',
    epilog='\n \n',
    formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument(
    'input_file', help='path to the input .csv file')

def timer(start_time=None, action="Activity"):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print(
            " "
            + action
            + ": %i hours %i minutes and %s seconds." % (thour, tmin, round(tsec, 2))
        )
#        print(" " + str(datetime.now().strftime("%Y-%m-%d %H:%M:%S")) + "\n")

def rmse_score(actual, predictions):
    assert len(actual) == len(predictions)
    return np.sqrt(mean_squared_error(actual, predictions))


args = parser.parse_args()

if os.access(args.input_file, os.R_OK):
    CSVFile = pd.read_csv(args.input_file)
else:
    parser.error(
        '\n !!! Input file "%s" does not exist in this directory !!!\n' %
        args.input_file)

id_code = args.input_file.rsplit('.', 1)[0] + '_ProtT5'

df_plm = CSVFile.drop(['Assembly_ID'], axis=1)
df_ids = CSVFile['Assembly_ID']
encoder = joblib.load('models/StandardScaler_NeNe-Top-ProtT5.joblib')
features = df_plm.columns.tolist()
df_plm[features] = encoder.transform(df_plm[features].values)

folds = 10
bags = 3
val_batchsize = 8192
batchsize = 32

fpred = []
starttime = timer(None)
for i in range(folds):
    start_time = timer(None)
    for bag in range(bags):
        nnet = load_model(
            "models/NN-ProtT5-model-fold-"
            + str("%02d" % (i + 1))
            + "-bag-"
            + str("%02d" % (bag + 1))
            + ".h5"
        )

        y_pred_bag = nnet.predict(df_plm, verbose=0, batch_size=val_batchsize).flatten()
        if bag > 0:
            y_pred = y_pred + y_pred_bag
        else:
            y_pred = y_pred_bag

    y_pred = y_pred / bags
    timer(start_time, "Fold %d" % (i + 1))

    if i > 0:
        fpred = pred + y_pred
    else:
        fpred = y_pred
    pred = fpred

timer(starttime, "Complete prediction")
mpred = pred / folds

result = pd.DataFrame(mpred, columns=["prediction"])
result["Assembly_ID"] = id_code
result = result[["Assembly_ID", "prediction"]]
print("\n 10-fold average prediction for %s: %.2f" % (id_code, mpred))
sub_file = id_code + ".csv"
result.to_csv(sub_file, index=False, float_format="%.4f")

if (mpred > 45):
    print('\n !!! As the general prediction is above 45 degrees, we are making a high temperature-based prediction with pLM !!!')

    CSVFile = pd.read_csv(args.input_file)
    id_code = args.input_file.rsplit('.', 1)[0] + '_ProtT5_HT'
    df_plm = CSVFile.drop(['Assembly_ID'], axis=1)
    df_ids = CSVFile['Assembly_ID']
    encoder = joblib.load('models/StandardScaler_NeNe-above45.joblib')
    features = df_plm.columns.tolist()
    df_plm[features] = encoder.transform(df_plm[features].values)

    fpred = []
    starttime = timer(None)
    for i in range(folds):
        start_time = timer(None)
        for bag in range(bags):
            nnet = load_model(
                "models/NN-ProtT5-above45-model-fold-"
                + str("%02d" % (i + 1))
                + "-bag-"
                + str("%02d" % (bag + 1))
                + ".h5"
            )

            y_pred_bag = nnet.predict(df_plm, verbose=0, batch_size=val_batchsize).flatten()
            if bag > 0:
                y_pred = y_pred + y_pred_bag
            else:
                y_pred = y_pred_bag

        y_pred = y_pred / bags
        timer(start_time, "Fold %d" % (i + 1))

        if i > 0:
            fpred = pred + y_pred
        else:
            fpred = y_pred
        pred = fpred

    timer(starttime, "Complete prediction")
    mpred = pred / folds

    result = pd.DataFrame(mpred, columns=["prediction"])
    result["Assembly_ID"] = id_code
    result = result[["Assembly_ID", "prediction"]]
    print("\n 10-fold average prediction for %s: %.2f\n" % (id_code, mpred))
    sub_file = id_code + ".csv"
    result.to_csv(sub_file, index=False, float_format="%.4f")
