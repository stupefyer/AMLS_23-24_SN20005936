import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import models
from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import f1_score
tf.keras.utils.set_random_seed(0)


def data_process():
    # Load datasets
    data = np.load('Datasets/pathmnist.npz')

    # Split the data to training, validation and tests according to original sets.
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
    # Normalize the pixel values
    X_train  = X_train  / 255.0
    X_val = X_val / 255.0
    X_test = X_test/ 255.0

    return X_train, y_train, X_val, y_val, X_test, y_test

def CNN_model():
    cnn = models.Sequential([

        # Model B
        # Convolutional Layer 1
        layers.Conv2D(filters=32, kernel_size=(3,3), activation='relu', input_shape=(28,28,3), padding='same'),

        # Max pooling layer
        layers.MaxPooling2D((2,2), padding='same'),

        # Convolutional Layer 2
        layers.Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same', strides=1),

        # Max pooling layer
        layers.MaxPooling2D((2,2), padding='same'),

        # Convolutional Layer 3
        layers.Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding='same', strides=1),

        # Max pooling layer
        layers.MaxPooling2D((2,2), padding='same'),

        # Dropout layer
        layers.Dropout(0.8),

        layers.Flatten(),

        # Fully connected layer 1
        layers.Dense(128, activation='relu'),

        # Fully connected layer 2
        layers.Dense(256, activation='relu'),

        # Output layer
        layers.Dense(9, activation='softmax')

    ])

    optimizer = keras.optimizers.Adam(lr=10e-6)

    cnn.compile(optimizer=optimizer, 
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
    
    return cnn

def plot_training_curves(history):
    plt.plot(history['accuracy'])
    plt.plot(history['val_accuracy'])
    plt.title('Accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'])
    plt.show()

    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('Loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'])
    plt.show()

def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred, normalize='true')
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8])
    cm_display.plot(cmap='seismic')
    plt.show()

def calculate_metrcis(y_test, y_pred):
    precision = metrics.precision_score(y_test, y_pred, average='macro')
    recall = metrics.recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    return precision, recall, f1

def get_filters(cnn_saved_model):
    filters, biases = cnn_saved_model.layers[0].get_weights()

    f_min, f_max = filters.min(), filters.max()
    filters = (filters - f_min) / (f_max - f_min)

    n_filters, ix = 32, 1

    for i in range(n_filters):
        f = filters[:, :, :, i]    
        ax = plt.subplot(4, 8, ix)
        ax.set_xticks([])
        ax.set_yticks([])

        plt.imshow(f[:, :, 0], cmap='gray')
        ix += 1  
    plt.show()

def get_convoluted_images(cnn, x):
    layer_outputs = [layer.output for layer in cnn.layers[:8]] 
    activation_model = models.Model(inputs = cnn.input, outputs = layer_outputs) 
    activations = activation_model.predict(x)

    plt.figure(figsize=(20,20))
    for i in range(0, 32):
        plt.subplot(8, 8, i+1)
        plt.axis('off')
        plt.matshow(activations[1][0, :, :, i], cmap ='viridis', fignum=0) 
    cax = plt.axes((0.95, 0.53, 0.03, 0.35))
    plt.colorbar(cax=cax)
    plt.show()


# Load dataset
X_train, y_train, X_val, y_val, X_test, y_test = data_process()

# Define CNN model
cnn = CNN_model()

# Train the model
history = cnn.fit(X_train, y_train, epochs=100, batch_size=128, validation_data=(X_val, y_val), shuffle=True)

# Save the model and model's training history
#cnn.save('A/CNN_Model_A.h5')
#np.save('A/CNN_Model_A_History.npy',cnn_history.history)

# Load saved model and its training history
cnn = tf.keras.models.load_model('B/CNN_Model_B.h5')
history = np.load('B/CNN_Model_B_History.npy',allow_pickle='TRUE').item()
cnn.summary()

# Visualise filters used 
filters = get_filters(cnn)

#Visualise convoluted images
conv = get_convoluted_images(cnn, np.expand_dims(X_test[3974], axis=0))

# Plot training curves
trainig_curves = plot_training_curves(history)

# Evaluate the model's performance on test set
cnn.evaluate(X_test, y_test)

# Use the model to classifiy on test set
y_pred = cnn.predict(X_test)
y_pred = np.argmax(y_pred, axis=1)

# Plot confusion matrix on test set
cm = plot_confusion_matrix(y_test, y_pred)

# Calculate precision, recall and f-1 score on test set
precision, recall, f1 = calculate_metrcis(y_test, y_pred)
print("Macro Precision = ", + precision)
print("Macro Recall = ", + recall)
print("Macro F1 score = ", + f1)