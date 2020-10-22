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

# Import database helper
import db_helper


# Deep Learning
from joblib import dump, load

from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


################################################################################
#                                                                              #
#                        Loading the data from the DB                          #
#                                                                              #
################################################################################

# get the processed accidents
positive = db_helper.get_accidents(LIMIT=20000)


################################################################################
#                                                                              #
# This section of the script randomly samples the negative examples.           #
#                                                                              #
################################################################################

# Create a data frame of randomly sampled negative examples
negative = positive.sample(n=4*positive.shape[0], replace=True, random_state=0)

# Randomly alter the time of negative examples
np.random.seed(0)
negative.loc[:, 'randomhours'] = np.random.normal(scale=positive.accident_datetime.dt.hour.std(), size=negative.shape[0]).astype('int64')
negative.loc[:, 'accident_datetime'] = negative.accident_datetime+negative.randomhours.transform(lambda x: datetime.timedelta(hours=x))
negative = negative.drop('randomhours', axis=1)

# Randomly alter the date of negative examples
np.random.seed(0)
negative.loc[:, 'randomdays'] = np.random.normal(scale=positive.accident_datetime.transform(lambda x: x.timetuple().tm_yday).std(), size=negative.shape[0]).astype('int64')
negative.loc[:, 'accident_datetime'] = negative.accident_datetime+negative.randomdays.transform(lambda x: datetime.timedelta(days=x))
negative = negative.drop('randomdays', axis=1)

# Drop the dates that preceed the year 2012 or exceed the year 2019 in UTC time
negative = negative.loc[(negative.accident_datetime > pd.Timestamp(2012, 1, 1, 4)) & (negative.accident_datetime < pd.Timestamp(2020, 1, 1, 5)), :]

# Drop duplicated observations
negative = negative.drop_duplicates()

# Create the positive and negative labels
positive.loc[:, 'accident'] = 1
negative.loc[:, 'accident'] = 0

# Append the positive and negative example data frames
df = positive.append(negative, ignore_index=True)

# Drop duplicated negative examples
df.loc[:, 'duplicate'] = df.duplicated(keep=False)
df = df.loc[(df.duplicate == False) | (df.accident == 1), :].drop('duplicate', axis=1)

# log
print("Length of set : ", len(df))
print("Number of positives : ", len(df.loc[df['accident'] == 1]))
print("Number of negatives : ", len(df.loc[df['accident'] == 0]))

################################################################################
#                                                                              #
# This section of the script split our dataframe into a training set.          #
#                                                                              #
################################################################################

# keep only certain columns
features = ['accident', 'temperature', 'dewpoint', 'humidity', 'wdirection', 'wspeed', 'visibility', 'pressure', 'risky']
df = df[features]

# drop rows with null values
df = df.dropna()

# Split features/label
features = ['temperature', 'dewpoint', 'humidity', 'wdirection','wspeed', 'visibility', 'pressure', 'risky']
label = ['accident']
X = df[features]
y = df[label]

# Split the data
train_X, valid_X, train_Y, valid_Y = train_test_split(X, y, test_size=0.15, random_state=42, shuffle=True, stratify=y)
print("Length of training set : ", len(train_X))
print("Length of validation set : ", len(valid_X))

# cast to np
valid_Y = np.array(valid_Y)
valid_X = np.array(valid_X)

# Normalize data
scaler = StandardScaler()
train_X_n = scaler.fit_transform(train_X)
valid_X_n = scaler.transform(valid_X)



################################################################################
#                                                                              #
# Random Forest                                                                #
#                                                                              #
################################################################################

# from sklearn.ensemble import RandomForestClassifier

# rdf_classifier = RandomForestClassifier(n_estimators=30, random_state=0)
# rdf_classifier.fit(train_X, train_Y)

# rdf_predictions = rdf_classifier.predict(valid_X)

# success = 0
# for i, pred in enumerate(rdf_predictions):
#     if(pred == valid_Y[i]):
#         success += 1

# print("RDF Validation Accuracy = " + str(success/len(valid_X)))



################################################################################
#                                                                              #
# CNN                                                                          #
#                                                                              #
################################################################################

# define model
model = Sequential()
model.add(Dense(128, input_dim=len(features), activation= "relu"))
model.add(Dense(64, activation= "relu"))
model.add(Dropout(rate=0.5))
model.add(Dense(64, activation= "relu"))
model.add(Dense(32, activation= "relu"))
model.add(Dense(1, activation='sigmoid'))
model.summary() #Print model Summary

# Compile model
model.compile(loss="binary_crossentropy" , optimizer="adam", metrics=["accuracy"])

# Fit Model
history = model.fit(train_X_n, train_Y, batch_size=128, epochs=10, verbose=0)

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

# evaluate predict
cnn_predictions = model.predict(valid_X_n)
success = 0
for i, pred in enumerate(cnn_predictions):
    if(round(pred[0]) == valid_Y[i]):
        success += 1

print("Validation Accuracy = " + str(success/len(valid_X_n)))

# save model
model.save('model_cnn.h5')
