#!/usr/bin/env python
# coding: utf-8

import warnings
warnings.filterwarnings('ignore')
import os
import re
from tqdm import tqdm
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
from sklearn.svm import SVR
import joblib
from Bio import SeqIO
import argparse

pd.set_option('display.max_rows', 5000)
pd.set_option('display.max_columns', 5000)

#############################################################################################
# https://www.geeksforgeeks.org/python-count-overlapping-substring-in-a-given-string/
#############################################################################################
def CountOccurrences(string, substring):

    """
    ########################################################################
    str.count fails to count overlapping occurencies of the same word
    This function fixes that.
    ########################################################################
    """
    # Initialize count and start to 0
    count = 0
    start = 0

    # Search through the string till
    # we reach the end of it
    while start < len(string):

        # Check if a substring is present from
        # 'start' position till the end
        pos = string.find(substring, start)

        if pos != -1:
            # If a substring is present, move 'start' to
            # the next position from start of the substring
            start = pos + 1

            # Increment the count
            count += 1
        else:
            # If no further substring is present
            break
    # return the value of count
    return count

AALetter=["A","C","D","E","F","G","H","I","K","L","M","N","P","Q","R","S","T","V","W","Y"]
#############################################################################################
def CalculateDipeptideComposition(ProteinSequence):
	"""
	########################################################################
	Calculate the composition of dipeptidefor a given protein sequence.
	Usage:
	result=CalculateDipeptideComposition(protein)
	Input: protein is a pure protein sequence.
	Output: result is a dict form containing the composition of
	400 dipeptides.
	########################################################################
	"""

	len(ProteinSequence)
	Result={}
	for i in AALetter:
		for j in AALetter:
			Dipeptide=i+j
			Result[Dipeptide]=CountOccurrences(ProteinSequence,Dipeptide)
	return Result

#############################################################################################

parser = argparse.ArgumentParser(
    description=
    '\n Neural network prediction of optimal growth temperature from di-peptide composition.\n',
    epilog='\n \n',
    formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument(
    'input_file', help='path to the input .faa file')

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
    FastaFile = open(args.input_file, 'r')
else:
    parser.error(
        '\n !!! Input file "%s" does not exist in this directory !!!\n' %
        args.input_file)

id_code = args.input_file.rsplit('.', 1)[0]

# Create a list of all sequence records
records = list(SeqIO.parse(FastaFile, 'fasta'))
# Get the length of the list
sequence_count = len(records)
FastaFile.seek(0)

if sequence_count < 20000:
    w = 0
    for rec in SeqIO.parse(FastaFile, 'fasta'):
        w = w + 1
        name = rec.id
        seq = str(rec.seq)
        # delete * signs prodigal puts at the end of sequence
        seq = re.sub("\*", "", seq)
        DIP=CalculateDipeptideComposition(seq)
        if w == 1:
            df_dip = pd.DataFrame.from_dict(DIP,orient='index',dtype=np.int,columns=['count'])
        else:
            df_dipp = pd.DataFrame.from_dict(DIP,orient='index',dtype=np.int,columns=['count'])
            df_dip['count'] = df_dip['count'] + df_dipp['count']
else:
    w = 0
    with tqdm(total=sequence_count, desc="  Counting dipeptides", ncols=150, mininterval=2) as pbar:
        for rec in SeqIO.parse(FastaFile, 'fasta'):
            w = w + 1
            name = rec.id
            seq = str(rec.seq)
            # delete * signs prodigal puts at the end of sequence
            seq = re.sub("\*", "", seq)
            DIP=CalculateDipeptideComposition(seq)
            if w == 1:
                df_dip = pd.DataFrame.from_dict(DIP,orient='index',dtype=np.int,columns=['count'])
            else:
                df_dipp = pd.DataFrame.from_dict(DIP,orient='index',dtype=np.int,columns=['count'])
                df_dip['count'] = df_dip['count'] + df_dipp['count']
            pbar.update()

FastaFile.close()
aa_dip_sum = df_dip['count'].sum()
df_dip[id_code] = df_dip['count'] / aa_dip_sum

df_dip.drop(['count'], axis=1, inplace=True)
df_dip.sort_index(ascending=True, inplace=True)
df_dip = df_dip.transpose(copy=True)
df_dip.index.name = 'Assembly_ID'
df_dip.to_csv(id_code + "_DIpep_freq.csv", index=True, float_format='%.9f')
encoder = joblib.load('models/StandardScaler_NeNe-Top-DIpep.joblib')
features = df_dip.columns.tolist()
df_dip[features] = encoder.transform(df_dip[features].values)

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
            "models/NN-DIpep-model-fold-"
            + str("%02d" % (i + 1))
            + "-bag-"
            + str("%02d" % (bag + 1))
            + ".h5"
        )

        y_pred_bag = nnet.predict(df_dip, verbose=0, batch_size=val_batchsize).flatten()
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
result["Assembly_ID"] = id_code + '_DIpep'
result = result[["Assembly_ID", "prediction"]]
print("\n 10-fold average NN prediction for %s: %.2f" % (id_code + '_DIpep', mpred))
sub_file = id_code + "_DIpep.csv"
result.to_csv(sub_file, index=False, float_format="%.4f")

if (mpred > 45):
    print('\n !!! As the NN prediction is above 45 degrees, we are making a high temperature-based prediction with NNs !!!')
    df_dip = pd.read_csv(id_code + "_DIpep_freq.csv")
    df_dip.drop(['Assembly_ID'], axis=1, inplace=True)

    encoder = joblib.load('models/StandardScaler_SVR-above45.joblib')
    features = df_dip.columns.tolist()
    df_dip[features] = encoder.transform(df_dip[features].values)

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
                "models/NN-DIpep-above45-model-fold-"
                + str("%02d" % (i + 1))
                + "-bag-"
                + str("%02d" % (bag + 1))
                + ".h5"
            )

            y_pred_bag = nnet.predict(df_dip, verbose=0, batch_size=val_batchsize).flatten()
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
    test_prediction = pred / folds

    result = pd.DataFrame(test_prediction, columns=["prediction"])
    result["Assembly_ID"] = id_code + '_DIpep_HT'
    result = result[["Assembly_ID", "prediction"]]
    print("\n 10-fold average NN HT prediction for %s: %.2f\n" % (id_code + '_DIpep_HT', test_prediction))
    sub_file = id_code + "_DIpep_HT.csv"
    result.to_csv(sub_file, index=False, float_format="%.4f")
