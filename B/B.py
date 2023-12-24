import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models


# Load datasets
data = np.load('Datasets/pathmnist.npz')

# Data preprocessing
X_train = data['train_images']
y_train = data['train_labels']
y_train = y_train.reshape(-1,)

X_val = data['val_images']
y_val = data['val_labels']
y_val = y_val.reshape(-1,)

X_test = data['test_images']
y_test = data['test_labels']
y_test = y_test.reshape(-1,)

# Data preprocssing
X_train  = X_train  / 255.0
X_val = X_val / 255.0
X_test = X_test/ 255.0


# CNN
cnn = models.Sequential([

        # Convolutional Layers
        layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=(28,28,3), padding='same'),
        layers.MaxPooling2D((2,2), padding='same'),
        #layers.BatchNormalization(),

        layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same', strides=1),
        layers.MaxPooling2D((2,2), padding='same'),

        layers.Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding='same', strides=1),
        layers.MaxPooling2D((2,2), padding='same'),
        #layers.BatchNormalization(),
        layers.Dropout(0.6),

        # Dense layers
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        #layers.BatchNormalization(),
        layers.Dense(256, activation='relu'),
        #layers.BatchNormalization(),
        layers.Dense(9, activation='softmax')

])

cnn.compile(optimizer='adam', 
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

cnn.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
cnn.evaluate(X_val, y_val)
cnn.evaluate(X_test, y_test)


"""
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()


# Data preprocssing
train_images = train_images / 255.0
val_images = val_images / 255.0
test_images = test_images / 255.0


plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(train_labels[train_labels[i]])
plt.show()
"""

