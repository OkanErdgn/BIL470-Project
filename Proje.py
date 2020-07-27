""""

OLUSTURULMUS MODELLER EN ALT KISIMDADIR
COMMENT ICINDEKI KISIMLAR ONCEKI DENEMELERIMIZ ICIN OLUSTURULMUS MODELLERDIR
SONRASINDA PROJE-2 VE PROJE-3 E GECILDI ORADA TEST YAPILDI
YAPILAN CALISMALAR BU KODDA BASLADI


"""

import math

import matplotlib.pyplot as plt



plt.style.use('fivethirtyeight')
class Datanode:
    def __init__(self, dataval=None): # csv value leri icin olusturdugum bir node
        self.value = dataval
        self.count = 0
        self.poscount = 0
class Node: # Tree de kullanilan node
    def __init__(self):
        self.data = []
        self.depth = -1
        self.leafdata = -1
        self.isleaf = 0
        self.dataval = -1
        self.parent = []
        self.child = []


def Read(File): # Bir veri parcasında her bir predictor un alabilecegi degerleri bir array olusturur ve o arrayi doner
    values = []
    values2 = []
    for i in range(len(File[0])):
        values.append(values2)
        values2 = []
    for i in range(len(File[0])-1):

        for k in range(0,len(File)):

            if(k == 0):
                dat = Datanode(File[k][i])
                values[i].append(dat)
                values[i][k].count += 1
                if File[k][len(File[0])-1] == 1:
                    values[i][k].poscount += 1
            else:
                tmp = 0
                for z in range(0,len(values[i])):
                    if(values[i][z].value == File[k][i]):
                        tmp = 1
                        values[i][z].count += 1
                        if File[k][len(File[0])-1] == 1:
                            values[i][z].poscount += 1
                        break
                if tmp == 0:
                    dat = Datanode(File[k][i])
                    dat.count += 1
                    if File[k][len(File[0])-1] == 1:
                        dat.poscount += 1
                    values[i].append(dat)
    return values


def newarr(S,attr,val): # bir S datasinin istenilen bir predictor icin o predictorun istebilen bir value oldugu durumlari yeni array olarak doner
    tmpdat = []
    for i in range(len(S)):
        if S[i][attr] == val:
            tmpdat.append(S[i])
    return tmpdat


def Entropy(S): # Entropi hesaplama
    count = len(S)
    poscount = 0
    for i in range(len(S)):
        if S[i][len(S[i])-1] == 1:
            poscount += 1
    if poscount == 0 or poscount == count:
        return 0
    return ((-poscount/count)*math.log(poscount/count,2)) - (((count-poscount)/count)*math.log((count-poscount)/count,2))


def Gain(S,attr,values): # bir predictor icin Gain i hesaplama
    if(len(S) == 0):
        return 0

    ES = Entropy(S)
    cs = 0
    for i in range(len(values[attr])):
        tmpS = newarr(S,attr,values[attr][i].value)
        cs += (len(tmpS)/len(S))*Entropy(tmpS)

    return ES-cs


def getx(S): # Aldigi data 2d arrayinin degerler kismini 2 array olarak doner (logisticreggression icin)
    tmp = []
    tmp2 = []
    for i in range(len(S)):
        for k in range(len(S[i])-1):
            tmp2.append(S[i][k])
        tmp.append(tmp2)
        tmp2 = []
    return tmp

def gety(S): # Aldigi data 2d arrayinin target kismini 1d array olarak doner (logisticreggression icin)
    tmp = []
    for i in range(len(S)):
        tmp.append(S[i][len(S[i])-1])
    return tmp

count0 = 0
count1 = 0

def ID3(S,node,attr,dpth,v): # DTLog seklinde tree'yi olusturuyor
    pos = 0
    node.depth = dpth


    #print(len(S))


    if len(attr) == 0: # attribute lar bitti ise most common olani atiyor
        node.data = S.copy()
        node.isleaf = 1
        return
    gain = -1
    gval = -1
    if len(attr) > 1:
        for i in range(len(attr)): # Gain i max olani buduran dongu
            g = Gain(S,attr[i],v)
        if g > gain:
            gain = g
            gval = attr[i]
    else:
        gval = attr[0]
    tmpid = []
    tmpid = attr.copy()
    tmpid.remove(gval)
    node.dataval = gval
    k = 0
    for i in range(len(v[gval])): # Gaini max olani bulduktan sonra o node icin child larin olusturulmasi

       # print(i," ",len(tmpid))
        tmparr = newarr(S, gval, v[gval][i].value)
        if len(tmparr) > 0:
            k = 1
            n = Node()
            n.parent.append(node)
            node.child.append(n)
            ID3(tmparr,node.child[i],tmpid,dpth+1,v)
        else:
            n = Node()
            n.parent.append(node)
            node.child.append(n)


    if k == 0:
        node.isleaf = 1
        node.data = S.copy()



def printtree(k,a): # Bir islevi yok agac olusturulurken test amacli kullanildi
    for i in range(k.depth):
        print("*", end=" ")
    print(k.depth, "  dataval:",train_data_features[0][k.dataval], "  isleaf",k.isleaf)
    for i in range(len(k.child)):
        printtree(k.child[i],a+1)

def printleaf(k): # Bir islevi yok agac olusturulurken test amacli kullanildi
    if k.isleaf == 1:
        print(len(k.data))
        return len(k.data)
    sum = 0
    for i in range(len(k.child)):
       sum += printleaf(k.child[i])
    return sum


file = open("C:/Users/14oka/Desktop/train.csv","r", errors='ignore')
a = file.readlines()
train_data = []
data2 = []


for x in a: # csv verisinin okunmasi
    #x = x[x.index(",")+1:len(x)]
    while len(x) > 0:
        if x.find(",") >= 0:
            if x.index((",")) == 0:
                tmp = "NULL"
            else:
                tmp = x[0:x.index(",")]
            data2.append(tmp)
            x = x[x.index(",") + 1:len(x)]
        else:
            tmp = x
            data2.append(tmp)
            break
    train_data.append(data2)
    data2 = []
print(len(train_data))

train_data_features = []
data2 = []
for i in range(1,len(train_data)):
    data2.append(train_data[i][2])
    data2.append(train_data[i][3])
   # data2.append(train_data[i][4])
    data2.append(train_data[i][len(train_data[i])-1])
    train_data_features.append(data2)
    data2 = []

"""
for k in range(len(train_data)):   # Train datayi bastirmak istersen
    print(k)
    for i in range(len(train_data[k])):
        print(train_data[k][i], end=" ")
"""
"""
values = Read(train_data_features)
unused_attr = []
for i in range(len(values)-1):
    unused_attr.append(i)
"""
"""
for k in range(len(values)):  # Train Datadaki Featurelarin alabilecegi degerler
    print()
    for i in range(len(values[k])):
        print(values[k][i].value, end=" | ")
"""


root = Node()


#ID3(train_data_features,root,unused_attr,0,values)
print()
#a = printleaf(root)
#printtree(root,0)
#print(a)
"""
dsize = (256,256)

train_target = []
tar = len(train_data_features[0])-1
for i in range(len(train_data_features)):
    train_target.append(train_data_features[i][tar])

image_arr = []

"""

"""
for i in range(1,len(train_data)):
    print(i)
    a = "D:/Users/14oka/Downloads/jpeg/train/"
    a2 = "D:/Users/14oka/Downloads/jpeg/trainresized/"
    b = ".jpg"
    c = a+train_data[i][0]+b
    d =a2+train_data[i][0]+b
    Img_1 = cv2.imread(c,1)
    #image_arr.append(Img_1)
    output = cv2.resize(Img_1, dsize)
    cv2.imwrite(d,output);
"""

"""
asd = "asd"
cv2.namedWindow(asd)        # Create a named window
cv2.moveWindow(asd, 40,30)
cv2.imshow(asd,output)
cv2.waitKey(0)
print(output)
"""



"""
import csv
import sklearn

array = []
with open('C:/Users/14oka/Desktop/train.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        array.append(row)
        #print(row)


#print(array[0][7])

patient = []

count = 1
while count < len(array):
    if 1 == int(array[count][7]):
        patient.append(array[count])
        #print(array[count][0])
    count = count + 1

print("how many patient")
print(len(patient))

#print(patient)
######
nopatient = []

count = 1
countPatient = 0
while count <= len(array) and countPatient < len(patient):
    if 0 == int(array[count][7]):
        countPatient = countPatient + 1
        nopatient.append(array[count])
        #print(array[count][0])
    count = count + 1

print("how many no patient")
print(len(nopatient))

sumarr = []

for i in range(len(patient)):
    sumarr.append(patient[i])
    sumarr.append(nopatient[i])
#print(sumarr)

#training_set = pd.DataFrame(sumarr,columns=array[0])
training_set = pd.read_csv("C:/Users/14oka/Desktop/train.csv")

#training_set = pd.read_csv("C:/Users/14oka/Desktop/train.csv")


training_imgs = ["D:/Users/14oka/Downloads/jpeg/trainresized/{}.jpg".format(x) for x in list(training_set.image_name)]
#print(list(training_imgs))

training_labels_1 = list(training_set['target'])
training_labels_2 = list(training_set['sex'])
training_labels_3 = list(training_set['age_approx'])
training_set = pd.DataFrame( {'Images': training_imgs,'target': training_labels_1})

#sklearn.utils.shuffle(training_set)
#training_x = pd.DataFrame.copy(training_set)
#training_y = pd.DataFrame.copy(training_set)
#training_x = pd.DataFrame({'Images': training_imgs,'Sex': training_labels_2,'Age': training_labels_3})
#training_y = pd.DataFrame({'target': training_labels_1})


#print(training_x)
#print(training_y)

#Changing the type  to str
training_set.target = training_set.target.astype(str)

print(training_set.head())
print()

train_dataGen = ImageDataGenerator()

train_generator = train_dataGen.flow_from_dataframe(
                                        dataframe = training_set,
                                        directory="",x_col="Images",
                                        y_col="target",
                                        class_mode="categorical",
                                        target_size=(256,256)
                                        )

model = Sequential()
model.add(Conv2D(filters=96, kernel_size=(5, 5), activation='relu', input_shape=(256,256,3)))
model.add(Dense(128, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=128, kernel_size=(7, 7), activation='relu'))
model.add(Dense(128, activation='relu'))

model.add(Conv2D(filters=256, kernel_size=(7, 7), activation='relu'))
model.add(Dense(128, activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(filters=256, kernel_size=(9, 9), activation='relu'))
model.add(Dense(128, activation='relu'))

model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(2, activation='softmax'))

######
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['categorical_accuracy','accuracy'])
model.summary()
model.fit(train_generator,
          batch_size=32,
          epochs=20,
          steps_per_epoch =10,
          workers=8
          )
"""
"""
model = Sequential()
model.add(Conv2D(filters=96, input_shape=(256,256,3), kernel_size = (5,5), strides=(2,2), activation='relu' ))
model.add(Dense(4096))
model.add(Activation('relu'))
model.add(Conv2D(filters=256, kernel_size = (7,7), strides=(2,2), activation='relu') )
model.add(Conv2D(filters=512, kernel_size = (7,7), strides=(2,2), activation='relu' ))
model.add(Dense(4096))
model.add(Activation('relu'))
model.add(Conv2D(filters=512, kernel_size = (9,9), strides=(2,2), activation='relu' ))
model.add(Flatten())
model.add(Dense(2))
model.add(Activation('softmax'))


# Compile the model
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['categorical_accuracy','accuracy'])
model.summary()
model.fit(train_generator,
          batch_size=256,
          epochs=10,
          workers=8,
          steps_per_epoch =10
          )

"""
"""
model = Sequential()
model.add(Conv2D(filters=64, kernel_size=(5, 5), activation='relu', input_shape=(256,256,3)))
model.add(Dense(128, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Dropout(0.25))
#model.add(Conv2D(128, kernel_size=(5, 5), activation='relu'))
model.add(Dense(128, activation='relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))

#model.add(Dropout(0.25))
model.add(Flatten())
#model.add(Dense(64, activation='relu'))
model.add(Dense(2, activation='sigmoid'))

# Compile the model
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['categorical_accuracy','accuracy'])
model.summary()
model.fit(train_generator,
          batch_size=256,
          epochs=10,
          steps_per_epoch = 10,
          workers=8
          )
"""



#model.save("testmodel6.model")



"""
classifier = Sequential()
classifier.add(Conv2D(filters = 32,kernel_size = (3,3), activation = 'relu', input_shape = (256,256,3)))
classifier.add(MaxPooling2D(pool_size = (2,2)))
classifier.add(Conv2D(64,(3,3),activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))
classifier.add(Flatten())
classifier.add(Dense(units = 64, activation = 'relu'))
classifier.add(Dense(units = 2 , activation = 'softmax'))
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['categorical_accuracy','accuracy'])
classifier.summary()

classifier.fit(train_generator,batch_size=1024, epochs = 20, steps_per_epoch = 100,workers=8)
classifier.save("testmodel.model")
"""




 #LOAD AND TEST
"""
new_model = tf.keras.models.load_model('testmodel6.model')
new_model.summary()

training_set = pd.read_csv("C:/Users/14oka/Desktop/test.csv")

training_imgs = ["D:/Users/14oka/Downloads/jpeg/test/{}.jpg".format(x) for x in list(training_set.image_name)]

nullarr = []
for i in range(len(training_imgs)):
    nullarr.append("")

training_set = pd.DataFrame( {'Images': training_imgs})




#Changing the type  to str


dsize = (256,256)

training_set = pd.read_csv("C:/Users/14oka/Desktop/test.csv")


training_imgs = ["D:/Users/14oka/Downloads/jpeg/test/{}.jpg".format(x) for x in list(training_set.image_name)]
training_set = pd.DataFrame( {'Images': training_imgs})





for i in range(1,len(train_data)):
    if int(train_data[i][len(train_data[i])-1]) == 1:
        Img_1 = cv2.imread("D:/Users/14oka/Downloads/jpeg/train/"+train_data[i][0]+".jpg", 1)
        # image_arr.append(Img_1)
        output = cv2.resize(Img_1, dsize)


        X = output.reshape([1] + list(output.shape))

        prediction = new_model.predict_classes(X)



        print("KANSERLI  ",prediction)

    
    if int(train_data[i][len(train_data[i]) - 1]) == 0:
        Img_2 = cv2.imread("D:/Users/14oka/Downloads/jpeg/train/" + train_data[i][0] + ".jpg", 1)
        # image_arr.append(Img_1)
        output = cv2.resize(Img_2, dsize)

        X = output.reshape([1] + list(output.shape))
        prediction = new_model.predict_classes(X)
        print("kansersiz  ",prediction)
  
"""
"""
for i in range(len(training_imgs)):
    Img_1 = cv2.imread(training_imgs[i], 1)
    # image_arr.append(Img_1)
    output = cv2.resize(Img_1, dsize)

    X = output.reshape([1] + list(output.shape))
    prediction = new_model.predict(X)
    print(prediction[0][1])

"""



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

epochs = 100
batch_size = 32
train_path = 'D:/Users/14oka/Downloads/jpeg/trainresized2/'
test_path = 'D:/Users/14oka/Downloads/jpeg/test/'

df = pd.read_csv('D:/Users/14oka/Downloads/jpeg/trainresized2/train22.csv')
df['image_name'] = train_path + df['image_name'] + '.jpg'

train_df = df[0:30000]
val_df = df[30000:]
test_df = df[30000:]
print(df)


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
                                                                                 target_size=(256, 256),
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
                                                                                      target_size=(256, 256),
                                                                                      class_mode="categorical",
                                                                                      shuffle=True)





class_weights = class_weight.compute_class_weight(
           'balanced',
            np.unique(train_batches.classes),
            train_batches.classes)



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



# 71 ALAN
"""
model_ann.add(Conv2D(filters = 32,kernel_size = (5,5), activation = 'relu', input_shape = (256,256,3)))
model_ann.add(MaxPooling2D(pool_size = (2,2)))
model_ann.add(Conv2D(64,(3,3),activation = 'relu'))
model_ann.add(MaxPooling2D(pool_size = (2,2)))
model_ann.add(Flatten())
model_ann.add(Dense(units = 64, activation = 'relu'))
model_ann.add(Dense(units = 2 , activation = 'softmax'))
model_ann.compile(Adam(lr=.0001), loss=[BinaryFocalLoss(gamma=2)], metrics=[AUC()])
"""


# Submission NU1


"""
model.add(Conv2D(64, (5,5),activation='relu', input_shape=(256,256,3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3,3) ,activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(256, (3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(256, (3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.4))
model.add(Dense(2))
model.add(Activation('softmax'))


model.compile(Adam(lr=.0001), loss=[BinaryFocalLoss(gamma=2)], metrics=[AUC()])
#mc=ModelCheckpoint('classifier.h5',monitor='val_loss',save_best_only=True,verbose=1,period=1)
model.fit(train_batches,steps_per_epoch=100,
                    validation_data=validation_batches, validation_steps=16, epochs=32,workers=4, verbose=1,class_weight={0:0.2, 1:0.8})

"""

"""
model.add(Conv2D(32, (5,5),activation='relu', input_shape=(256,256,3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3,3),activation='relu',))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(32, (3,3),activation='relu',))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3,3),activation='relu',))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(32, (3,3),activation='relu',))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3,3),activation='relu',))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.4))
model.add(Dense(2))
model.add(Activation('softmax'))

"""



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