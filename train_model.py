# train_model.py
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import cv2

# Load FER2013 Dataset CSV
df = pd.read_csv("fer2013.csv")  # download from Kaggle

# Prepare data
X = []
y = []
for i in range(len(df)):
    pixels = np.array(df['pixels'][i].split(), dtype='float32').reshape(48,48)
    X.append(pixels)
    y.append(df['emotion'][i])

X = np.array(X) / 255.0
X = np.expand_dims(X, axis=-1)
y = to_categorical(np.array(y), num_classes=7)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

# Build CNN Model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dense(7, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=15, batch_size=64)

# Save trained model
model.save("model/custom_emotion_model.h5")
