

import matplotlib
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#from tensorflow.keras.layers.convolutional import *
from sklearn.metrics import confusion_matrix
import itertools
import matplotlib.pyplot as plt
import os
from tensorflow.keras.metrics import AUC
import focal_loss
from focal_loss import BinaryFocalLoss
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.utils import class_weight


tf.keras.backend.clear_session()


train_path = 'D:/Users/14oka/Downloads/jpeg/trainresized2/'
test_path = 'D:/Users/14oka/Downloads/jpeg/test/'

df = pd.read_csv('D:/Users/14oka/Downloads/jpeg/trainresized2/train22.csv')
df['image_name'] = train_path + df['image_name'] + '.jpg'

train_df = df[0:30000]
val_df = df[30000:]
test_df = df[30000:]


train_df.target = train_df.target.astype(str)
test_df.target = test_df.target.astype(str)
val_df.target = val_df.target.astype(str)



train_batches = ImageDataGenerator(rescale=1/255.0,
                                   rotation_range = 30,
                                   zoom_range = 0.20,
                                   fill_mode = "nearest",
                                   shear_range = 0.20,
                                   horizontal_flip = True,
                                   width_shift_range = 0.1,
                                   height_shift_range = 0.1).flow_from_dataframe(train_df,
                                                                                x_col = 'image_name',
                                                                                y_col = 'target',
                                                                                target_size=(224, 224),
                                                                                class_mode="categorical")
validation_batches = ImageDataGenerator(rescale=1/255.0,
                                        rotation_range = 30,
                                        zoom_range = 0.20,
                                        fill_mode = "nearest",
                                        shear_range = 0.20,
                                        horizontal_flip = True,
                                        width_shift_range = 0.1,
                                        height_shift_range = 0.1).flow_from_dataframe(val_df,
                                                                                      x_col = 'image_name',
                                                                                      y_col = 'target',
                                                                                      target_size=(224, 224),
                                                                                      class_mode="categorical",
                                                                                      shuffle=True)




class_weights = class_weight.compute_class_weight(
           'balanced',
            np.unique(train_batches.classes),
            train_batches.classes)

epochs = 100
batch_size = 32

#**********************************************************************************

#***********************************************************************************

from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.layers import GlobalAveragePooling2D
learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
                                            patience=5,
                                            verbose=1,
                                            factor=0.5,
                                            min_lr=1e-7)


model = Sequential()
# VGG16 KULLANILDI 0.8356
base_model = tf.keras.applications.VGG16(weights='imagenet', input_shape=(224,224,3), include_top = False)

model = Sequential([base_model,GlobalAveragePooling2D(), Dense(2, activation='softmax')])

model.compile(Adam(lr=.00001), loss=[BinaryFocalLoss(gamma=0.25)], metrics=[AUC()])
model.fit(train_batches,batch_size=32,
                    validation_data=validation_batches, validation_batch_size=32, epochs=10,workers=4, verbose=1,callbacks=[learning_rate_reduction])
model.save("Transfer-Learning.model")



def load_image(img_path):

    img = image.load_img(img_path, target_size=(224, 224))
    img2 = image.img_to_array(img)
    img2 = np.expand_dims(img2, axis=0)
    img2 /= 255.
    return img2

csv = pd.read_csv('C:/Users/14oka/Desktop/test.csv')
f = open('submission.csv', 'w')
lines= []
for img in csv['image_name']:
  imj = 'D:/Users/14oka/Downloads/jpeg/test/' + img + '.jpg'
  imaj = load_image(imj)
  pred = model.predict(imaj)
  print(pred)
  lines.append(img + ',' + str(pred[0][1])+ '\n')

f.writelines(lines)
f.close()