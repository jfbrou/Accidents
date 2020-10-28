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
from keras.layers import Input, Dense, Dropout, BatchNormalization
from keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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

# Split the data into training and test sets
train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.05, random_state=0, shuffle=True, stratify=Y)

# Cast to numpy arrays
test_Y = np.array(test_Y)
test_X = np.array(test_X)

# Normalize the data
scaler = StandardScaler()
train_XN = scaler.fit_transform(train_X)
test_XN = scaler.transform(test_X)

################################################################################
#                                                                              #
# This section of the script defines, compiles and trains the model.           #
#                                                                              #
################################################################################

# Define the model instance
model = Sequential()
model.add(Dense(128, input_dim=train_XN.shape[1], activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fit the model
history = model.fit(x=train_XN, y=train_Y, validation_data=(test_XN, test_Y), epochs=100, batch_size=128)

# Evaluate the model
predictions = model.evaluate(x=test_XN, y=test_Y)
