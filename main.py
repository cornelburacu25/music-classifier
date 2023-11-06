import os
import random

import librosa
import numpy as np
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.layers import BatchNormalization, Activation, GlobalAveragePooling2D, Conv1D, MaxPooling1D, Bidirectional, \
    LSTM
from keras.optimizers import Adam
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.regularizers import l2


DATA_PATH = "D:/Projects/Python/musicClassifier/genres"
dataset = []
genres = {'blues': 0, 'classical': 1, 'country': 2, 'disco': 3, 'hiphop': 4,
        'jazz': 5, 'metal': 6, 'pop': 7, 'reggae': 8, 'rock': 9}

segment_duration = 2  # 5-second segments
# (64, 431, 1) -> 5s    (64, 173, 1) -> 2s

for genre, genre_number in genres.items():
    for filename in os.listdir(os.path.join(DATA_PATH, genre)):
        songname = os.path.join(DATA_PATH, genre, filename)
        total_duration = librosa.get_duration(path=songname)
        num_segments = int((total_duration - segment_duration) / segment_duration) + 1

        for index in range(num_segments):
            y, sr = librosa.load(songname, mono=True, duration=segment_duration, offset=index*segment_duration)
            ps = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=256, n_fft=512, n_mels=64)
            ps = librosa.power_to_db(ps**2)

            dataset.append((ps, genre_number))

# Assuming the dataset has changed in size, let's derive the split indices dynamically
total_samples = len(dataset)
train_end = int(total_samples * 0.8)
valid_end = train_end + int(total_samples * 0.1)  # 10% of total for validation

random.shuffle(dataset)
train = dataset[:train_end]
valid = dataset[train_end:valid_end]
test = dataset[valid_end:]
X_train, Y_train = zip(*train)
X_valid, Y_valid = zip(*valid)
X_test, Y_test = zip(*test)

# Ensure that the reshaping reflects the new dimensions if they have changed
X_train = np.array([x.reshape((*x.shape, 1)) for x in X_train])
X_valid = np.array([x.reshape((*x.shape, 1)) for x in X_valid])
X_test = np.array([x.reshape((*x.shape, 1)) for x in X_test])

Y_train = to_categorical(Y_train, 10)
Y_valid = to_categorical(Y_valid, 10)
Y_test = to_categorical(Y_test, 10)

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Dropout, Flatten, Dense
from keras.regularizers import l2

cnn_model = Sequential()

# First Convolutional Block
cnn_model.add(Conv2D(32, (3, 3), input_shape=(64, 173, 1), activation="relu"))
cnn_model.add(BatchNormalization())
cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
cnn_model.add(Dropout(0.3))

# Second Convolutional Block
cnn_model.add(Conv2D(64, (3, 3), activation="relu"))
cnn_model.add(BatchNormalization())
cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
cnn_model.add(Dropout(0.3))

# Third Convolutional Block
cnn_model.add(Conv2D(128, (3, 3), activation="relu"))
cnn_model.add(BatchNormalization())
cnn_model.add(MaxPooling2D(pool_size=(2, 2)))
cnn_model.add(Dropout(0.3))

# Fully Connected Layers
cnn_model.add(Flatten())
cnn_model.add(Dense(256, activation='relu', kernel_regularizer=l2(0.001)))
cnn_model.add(BatchNormalization())
cnn_model.add(Dropout(0.5))
cnn_model.add(Dense(10, activation='softmax'))

cnn_model.summary()

lstm_model = Sequential()

# First Convolutional Block
lstm_model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
lstm_model.add(MaxPooling1D(pool_size=2))
lstm_model.add(BatchNormalization())
lstm_model.add(Dropout(0.3))

# Second Convolutional Block
lstm_model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
lstm_model.add(MaxPooling1D(pool_size=2))
lstm_model.add(BatchNormalization())
lstm_model.add(Dropout(0.3))

# Third Convolutional Block
lstm_model.add(Conv1D(filters=256, kernel_size=3, activation='relu'))
lstm_model.add(MaxPooling1D(pool_size=2))
lstm_model.add(BatchNormalization())
lstm_model.add(Dropout(0.3))

# First Bidirectional LSTM layer
lstm_model.add(Bidirectional(LSTM(128, return_sequences=True)))
lstm_model.add(BatchNormalization())
lstm_model.add(Dropout(0.3))

# Second Bidirectional LSTM layer
lstm_model.add(Bidirectional(LSTM(64, return_sequences=True)))
lstm_model.add(BatchNormalization())
lstm_model.add(Dropout(0.3))

# Final LSTM layer
lstm_model.add(LSTM(32))
lstm_model.add(BatchNormalization())
lstm_model.add(Dropout(0.3))

# Fully Connected Layer
lstm_model.add(Dense(10, activation='softmax'))

# Model summary
lstm_model.summary()




# Compile the model (use appropriate optimizer, loss, and metrics)
lstm_model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.001), loss="categorical_crossentropy", metrics=['accuracy'])
cnn_model.compile(optimizer=Adam(lr = 1e-3), loss='categorical_crossentropy', metrics=['accuracy'])

# Learning rate scheduler
early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)
checkpoint = ModelCheckpoint("best_weights.h5", monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')


# Fit the model using training data and validation data
#callbacks=[reduce_lr, checkpoint, early_stopping
cnn_model.fit(X_train, Y_train, validation_data=(X_valid, Y_valid), epochs=90, batch_size=64)
lstm_history = lstm_model.fit(X_train, Y_train, validation_data=(X_valid, Y_valid), epochs=100, batch_size=64)

cnn_loss, cnn_accuracy = cnn_model.evaluate(X_test, Y_test)
print(f"Test Loss: {cnn_loss}")
print(f"Test Accuracy: {cnn_accuracy}")

lstm_loss, lstm_accuracy = lstm_model.evaluate(X_test, Y_test)
print(f"Test Loss: {lstm_loss}")
print(f"Test Accuracy: {lstm_accuracy}")

