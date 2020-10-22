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


# Find the current working directory
path = os.getcwd()

# Create a folder that contains all figures
if os.path.isdir(os.path.join(path, 'Figures')) == False:
    os.mkdir('Figures')
figures_dir_path = os.path.join(path, 'Figures')


def draw_as_pdf(geometries, out_path):
    """
        Function to draw geometries on a pdf
    """

    if os.path.exists(out_path) == False:

        # plot
        fig, ax = plt.subplots()
        geometries.plot(ax=ax, color='teal', markersize=0.1)
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(out_path, format='pdf')
        plt.close()

        # log
        logging.info(f'File created {out_path}')

    else:
        logging.info(f'Already exists {out_path}')



################################################################################
#                                                                              #
#                        Loading the data from the DB                          #
#                                                                              #
################################################################################

# get the processed accidents
positive = db_helper.get_accidents()


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
logging.info(f'Sampling of negative events done')
print(df.columns)


################################################################################
#                                                                              #
# This section of the script split our dataframe into a training set.          #
#                                                                              #
################################################################################

# Split features/label
features = ['road_segment_class','road_segment_direction',
            'temperature', 'dewpoint', 'humidity', 'wdirection','wspeed', 'visibility', 'pressure', 'risky']

label = ['accident']
X = df[features]
y = df[label]

# Split the data
train_X, valid_X, train_Y, valid_Y = train_test_split(X, y, test_size=0.15, random_state=42, shuffle=True, stratify=y)

# cast to np
valid_Y = np.array(valid_Y)
valid_X = np.array(valid_X)

print("Length of training set : ", len(train_X))
print("Length of validation set : ", len(valid_X))

# Normalize data
scaler = StandardScaler()
train_X_n = scaler.fit_transform(train_X)
valid_X_n = scaler.transform(valid_X)


# define model
model = Sequential()
model.add(Dense(100, input_dim=len(features), activation= "relu"))
model.add(Dense(60, activation= "relu"))
model.add(Dropout(rate=0.3))
model.add(Dense(60, activation= "relu"))
model.add(Dense(30, activation= "relu"))
model.add(Dense(1, activation='sigmoid'))
model.summary() #Print model Summary

# Compile model
model.compile(loss="binary_crossentropy" , optimizer="adam", metrics=["accuracy"])

# Fit Model
history = model.fit(train_X_n, train_Y, epochs=10, verbose=0)

# eval
score = model.evaluate(valid_X_n, valid_Y)
print('Test Accuracy: {}'.format(score[1]))

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# predict
cnn_predictions = model.predict(valid_X_n)

success = 0
for i, pred in enumerate(cnn_predictions):
    if(round(pred[0]) == valid_Y[i]):
        success += 1

print("Validation Accuracy = " + str(success/len(valid_X_n)))

model.save('model_cnn.h5')
























################################################################################
#                                                                              #
#                                 Produce PDFs                                 #
#                                                                              #
################################################################################

# write
out_path = os.path.join(figures_dir_path, 'accidents_2019.pdf')
draw_as_pdf(accidents, out_path)

# write
out_path = os.path.join(figures_dir_path, 'segments.pdf')
draw_as_pdf(road_segments, out_path)
