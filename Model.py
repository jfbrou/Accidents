################################################################################
#                                                                              #
# This section of the script imports libraries and creates directories.        #
#                                                                              #
################################################################################

# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.models import load_model, Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

# Find the current working directory
path = os.getcwd()

# Create a folder that contains all data files
if os.path.isdir(os.path.join(path, 'Data')) == False:
    os.mkdir('Data')
data = os.path.join(path, 'Data')

# Create a folder that contains all figures
if os.path.isdir(os.path.join(path, 'Figures')) == False:
    os.mkdir('Figures')
figures = os.path.join(path, 'Figures')

################################################################################
#                                                                              #
# This section of the script splits our data into training and test sets.      #
#                                                                              #
################################################################################

# Load the data
df = pd.read_csv(os.path.join(data, 'data.csv'))

# Keep certain columns
df = df.dropna()
X = df.loc[:, df.columns[~df.columns.isin(['datetime', 'segmentid', 'geometry', 'accident'])]]
Y = df.loc[:, 'accident']

# Split the data into training, development and test sets
train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.05, shuffle=True, stratify=Y)
train_X, dev_X, train_Y, dev_Y = train_test_split(train_X, train_Y, test_size=0.05, shuffle=True, stratify=train_Y)

# Normalize the data
scaler = StandardScaler()
train_XN = scaler.fit_transform(train_X)
dev_XN = scaler.transform(dev_X)
test_XN = scaler.transform(test_X)

################################################################################
#                                                                              #
# This section of the script defines, compiles and trains the model.           #
#                                                                              #
################################################################################

# Define the model instance
model = Sequential()
model.add(Dense(256, input_dim=train_XN.shape[1], activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.1))
model.add(Dense(32, activation='relu'))
model.summary()

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[tf.keras.metrics.Recall(), tf.keras.metrics.AUC(curve='PR'), tf.keras.metrics.AUC(curve='ROC')])

# Fit the model
history = model.fit(x=train_XN, y=train_Y, validation_data=(dev_XN, dev_Y), epochs=100, batch_size=128)

# Save the model
model.save(os.path.join(path, 'model.h5'))

# Evaluate the model
evaluation = model.evaluate(x=test_XN, y=test_Y)

# Create the confusion matrix
prediction = model.predict_classes(test_XN)
confusion = confusion_matrix(test_Y, prediction)
