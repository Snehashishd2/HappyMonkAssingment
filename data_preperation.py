import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report, confusion_matrix
import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

EPOCHS = 10
BATCH_SIZE = 128

#Reading the train dataset
data = pd.read_csv('train.csv')

#checking unique numbers in train label column
unique = data['label'].unique()
print("Unique Numbers :", unique)

#countine the unique number of digits for classification
n_classes = len(unique)
print("Number of classes :", n_classes)

#Filtering the dataset between X and Y
x = data.drop(labels=["label"], axis=1)
y = data['label']

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.20, random_state=42, stratify=y)
# # normalize pixel values
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
print(x_train.to_numpy().shape, x_test.to_numpy().shape)
# steps = x_train.to_numpy().shape[0] // BATCH_SIZE


def dense_to_one_hot(labels_dense, num_classes=10):
    """Convert class labels from scalars to one-hot vectors"""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1

    return labels_one_hot


y_coded = dense_to_one_hot(y_train.to_numpy(), 10)

steps_train = x_train.to_numpy().shape[0] // BATCH_SIZE
steps_test = x_test.to_numpy().shape[0] // BATCH_SIZE
for epoch in range(EPOCHS):
    for step in range(steps_train):
        x_batch = x_train[step*BATCH_SIZE:(step+1)*BATCH_SIZE]
        y_batch = y_train[step*BATCH_SIZE:(step+1)*BATCH_SIZE]

    for step in range(steps_test):
        x_batch = x_train[step*BATCH_SIZE:(step+1)*BATCH_SIZE]
        y_batch = y_train[step*BATCH_SIZE:(step+1)*BATCH_SIZE]
    print(x_batch.to_numpy(), x_batch.to_numpy().shape)
    print(y_batch.to_numpy(), y_batch.to_numpy().shape)

    break
