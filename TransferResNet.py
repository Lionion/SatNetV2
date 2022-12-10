import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
import splitfolders
import shutil
import pandas as pd

test_dir = "input/test"

BATCH_SIZE = 32
IMG_SIZE = (160, 160)
test_dataset = tf.keras.utils.image_dataset_from_directory(test_dir,
                                                           shuffle=True,
                                                           batch_size=BATCH_SIZE,
                                                           image_size=IMG_SIZE)

IMG_SHAPE = IMG_SIZE + (3,)

model50 = load_model("model/SatNet50.h5")
model101 = load_model("model/SatNet101.h5")

#loss, accuracy = model50.evaluate(test_dataset)
#print('Test accuracy :', accuracy)

#loss, accuracy = model101.evaluate(test_dataset)
#print('Test accuracy :', accuracy)

y_true = np.concatenate([y for x, y in test_dataset], axis=0)
y_pred = np.argmax(model50.predict(test_dataset), axis=1)
print(y_pred)
con_mat = tf.math.confusion_matrix(labels=y_true, predictions=y_pred).numpy()

classes=[0,1,2,3,4,5,6,7,8,9]

con_mat_df = pd.DataFrame(con_mat,
                     index = classes,
                     columns = classes)

figure = plt.figure(figsize=(9, 9))
sns.heatmap(con_mat_df, annot=True,cmap=plt.cm.Blues)
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

for i in range(len(con_mat)):
    sum = 0
    precision = con_mat[i][i]
    for j in range(len(con_mat[i])):
        sum += con_mat[i][j]
    precision /= sum

    print("The precision of SatNet50 for class " + str(i) + " is " + str(precision))

for i in range(len(con_mat)):
    sum = 0
    recall = con_mat[i][i]
    for j in range(len(con_mat[i])):
        sum += con_mat[j][i]
    recall /= sum

    print("The recall of SatNet50 for class " + str(i) + " is " + str(recall))

y_pred = np.argmax(model101.predict(test_dataset), axis=1)
con_mat = tf.math.confusion_matrix(labels=y_true, predictions=y_pred).numpy()

classes=[0,1,2,3,4,5,6,7,8,9]

con_mat_df = pd.DataFrame(con_mat,
                     index = classes,
                     columns = classes)

figure = plt.figure(figsize=(9, 9))
sns.heatmap(con_mat_df, annot=True,cmap=plt.cm.Blues)
plt.tight_layout()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

for i in range(len(con_mat)):
    sum = 0
    precision = con_mat[i][i]
    for j in range(len(con_mat[i])):
        sum += con_mat[i][j]
    precision /= sum

    print("The precision of SatNet101 for class " + str(i) + " is " + str(precision))

for i in range(len(con_mat)):
    sum = 0
    recall = con_mat[i][i]
    for j in range(len(con_mat[i])):
        sum += con_mat[j][i]
    recall /= sum

    print("The recall of SatNet101 for class " + str(i) + " is " + str(recall))

loss, accuracy = model50.evaluate(test_dataset)
print('Test accuracy :', accuracy)

loss, accuracy = model101.evaluate(test_dataset)
print('Test accuracy :', accuracy)
