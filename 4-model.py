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
from joblib import dump, load

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

# print features
for i, feature in enumerate(train_X.columns):
    print(f'Feature #{i} : {feature}')


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
model.add(BatchNormalization())
model.add(Dense(1, activation='sigmoid'))
model.summary()

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fit the model
history = model.fit(x=train_XN, y=train_Y, validation_data=(dev_XN, dev_Y), epochs=100, batch_size=128)

# Save the model
model.save(os.path.join(path, 'model.h5'))

# Evaluate the model
evaluation = model.evaluate(x=test_XN, y=test_Y)

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
