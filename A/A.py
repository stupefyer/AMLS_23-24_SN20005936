import numpy as np
import matplotlib.pyplot as plt

# Load datasets
data = np.load('Datasets/pneumoniamnist.npz')

# Data preprocessing
train_images = data['train_images']
train_labels = data['train_labels']

val_images = data['val_images']
val_labels = data['val_labels']

test_images = data['test_images']
test_labels = data['test_labels']

"""
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()
"""

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