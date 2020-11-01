################################################################################
#                                                                              #
# This section of the script sends the data to a neural model                  #
#                                                                              #
################################################################################

# Import libraries
import os
import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Logging
import logging
logging.basicConfig(level=logging.INFO)

# Deep Learning
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, BatchNormalization

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

################################################################################
#                                                                              #
#                        Loading the data from the DB                          #
#                                                                              #
################################################################################

# Find the current working directory
path = os.getcwd()

# Create a folder that contains all data files
if os.path.isdir(os.path.join(path, 'Data')) == False:
    raise Exception('Data directory does not exist, run retrieve script')
data_dir_path = os.path.join(path, 'Data')

# Load the data
df = pd.read_csv(os.path.join(data_dir_path, 'data.csv'))

# remove nan
df = df.dropna()

# balance the classes
g = df.groupby('accident')
df = g.apply(lambda x: x.sample(g.size().min()).reset_index(drop=True))

# print classes
print(df['accident'].value_counts())

# Keep certain columns
X = df.loc[:, df.columns[~df.columns.isin(['accident_datetime', 'accident_geometry', 'road_segment_geometry', 'accident_id', 'accident'])]]
Y = df.loc[:, 'accident']

# Split the data into training, development and test sets
train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.1, shuffle=True, stratify=Y)
train_X, dev_X, train_Y, dev_Y = train_test_split(train_X, train_Y, test_size=0.1, shuffle=True, stratify=train_Y)

# Normalize the data
scaler = StandardScaler()
train_XN = scaler.fit_transform(train_X)
dev_XN = scaler.transform(dev_X)
test_XN = scaler.transform(test_X)

################################################################################
#                                                                              #
# Test Model.                                                                  #
#                                                                              #
################################################################################

# load model
model = load_model('model.h5')

# Evaluate the model
evaluation = model.evaluate(x=test_XN, y=test_Y)

# Create the confusion matrix
prediction = (model.predict(test_XN) > 0.5).astype("int32")
confusion = confusion_matrix(test_Y, prediction, labels=[0,1], normalize='all')
print(f'True Negative : {np.round(confusion[0][0], 4)}')
print(f'False Negative : {np.round(confusion[1][0], 4)}')
print(f'True Positives : {np.round(confusion[1][1], 4)}')
print(f'False Positives : {np.round(confusion[0][1], 4)}')
