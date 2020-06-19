#!/usr/bin/env python
# coding: utf-8

# Loading data!

# In[ ]:


import pandas as pd

train_csv_path="" #Use your csv path here

train = pd.read_csv(train_csv_path)

train.rename(columns = {'x2' : 'y1', 'y1' : 'x2'}, inplace = True)

train.reset_index(drop=True, inplace=True)

drop_index=train[train['name']=='5392.jpg'].index

train=train.drop(train.index[drop_index])

"""
Do dropping 1861.jpg as long as it fully goes away cos it has incorrect bbox coordinates
"""
drop_index=train[train['name']=='1861.jpg'].index

train=train.drop(train.index[drop_index])

drop_index=train[train['name']=='1861.jpg'].index

train=train.drop(train.index[drop_index])

train.reset_index(drop=True,inplace=True)


# In[ ]:


drop_index=train[train['name']=='1861.jpg'].index

train=train.drop(train.index[drop_index])

train.reset_index(drop=True,inplace=True)

# I chose the following classes from the dataset : "face_with_mask" ,"face_with_mask_incorrect" , "face_no_mask"

#You can choose other classes if you want

# In[ ]:


wanted_samples=pd.DataFrame(train[train['classname'].isin(['face_with_mask','face_with_mask_incorrect','face_no_mask'])])

# I have put "face_with_mask_incorrect" and "face_no_mask" under a common category called "Without_mask"

# I have put "face_with_mask" as "With_mask"

# However , you can choose the classes which you wish to work with

# In[ ]:


for i in range(len(wanted_samples)):
    if wanted_samples.iloc[i]['classname'] =='face_with_mask':
        wanted_samples['classname'].replace([wanted_samples.iloc[i]['classname']],['With_mask'],inplace=True)
    elif wanted_samples.iloc[i]['classname'] =='face_with_mask_incorrect'or wanted_samples.iloc[i]['classname'] =='face_no_mask':
        wanted_samples['classname'].replace([wanted_samples.iloc[i]['classname']],['Without_mask'],inplace=True)


# Splitting into train and validation sets

# In[ ]:


import cv2 
import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np

ws=wanted_samples
ws = ws.iloc[np.random.permutation(len(ws))]
ws.reset_index(drop=True, inplace=True)

#Using LabelBinarizer to encode class labels

#In this supervised learning problem , I have chosen two classes : "With_mask" and "Without_mask"
labels = ws['classname']
labels = np.array(labels)
lb = LabelBinarizer()
labels = lb.fit_transform(labels)

(train_df,test_df,train_labels,test_labels) = train_test_split(ws,labels,test_size=0.20, stratify=labels, random_state=42)

train_df.reset_index(drop=True, inplace=True)
test_df.reset_index(drop=True, inplace=True)

points = train_df.drop(['name','classname'], axis=1)
points = points.to_numpy() 

image_dir="" #Use your image diretory here

train_data = []
for i,j in enumerate(train_df['name']):
    direc = os.path.sep.join([image_dir,'Medical Mask/images/%s'%j])
    img1 = cv2.imread(direc)
    img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
    x1= int(points[i][0])
    y1= int(points[i][1])
    x2= int(points[i][2])
    y2= int(points[i][3])
    cropped = img1[y1:y2,x1:x2]
    image = cv2.resize(cropped,(224,224))
    image=img_to_array(image)
    image=preprocess_input(image)
    train_data.append(image)
    
train_data = np.array(train_data, dtype="float32")
train_labels = np.array(train_labels)
train_labels = to_categorical(train_labels,num_classes=2)

test_points = test_df.drop(['name','classname'], axis=1)
test_points = test_points.to_numpy() 

test_data = []
for i,j in enumerate(test_df['name']):
    direc = os.path.sep.join([image_dir,'Medical Mask/images/%s'%j])
    img1 = cv2.imread(direc)
    img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
    x1= int(test_points[i][0])
    y1= int(test_points[i][1])
    x2= int(test_points[i][2])
    y2= int(test_points[i][3])
    cropped = img1[y1:y2,x1:x2]
    image = cv2.resize(cropped,(224,224))
    image=img_to_array(image)
    image=preprocess_input(image)
    test_data.append(image)

test_data = np.array(test_data, dtype="float32")
test_labels = np.array(test_labels)
test_labels = to_categorical(test_labels,num_classes=2)

# In[ ]:


final_train_labels=train_labels
final_test_labels=test_labels


# Data Augmentation to increase out-of-sample performance

# In[ ]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen=ImageDataGenerator(rotation_range=20,
zoom_range=0.15,
width_shift_range=0.2,
height_shift_range=0.2,
shear_range=0.15,
horizontal_flip=True,
fill_mode="nearest")


# In[ ]:


features=train_data
test_features=test_data


# Displaying shapes of the train-test feature matrices and train-test label matrices

# In[ ]:


features.shape , train_labels.shape ,test_features.shape , test_labels.shape


# Loading ResNet pre-trained on ImageNet for transfer learning

# In[ ]:


baseModel = ResNet50(weights="imagenet", include_top=False,input_tensor=Input(shape=(224, 224, 3)))


# Setting hyperparameters for training

# In[ ]:


INIT_LR = 1e-3
EPOCHS = 30
BS = 32


# Below classes will help in implementing cosine annealing learning rate schedule (for training)

# In[ ]:


import tensorflow
class SnapshotCallbackBuilder:
    def __init__(self, nb_epochs, nb_snapshots, init_lr=0.1):
        self.T = nb_epochs
        self.M = nb_snapshots
        self.alpha_zero = init_lr

    def get_callbacks(self, model_prefix='Model'):

        callback_list = [
            tensorflow.keras.callbacks.LearningRateScheduler(schedule=self._cosine_anneal_schedule)
        ]

        return callback_list

    def _cosine_anneal_schedule(self, t):
        cos_inner = np.pi * (t % (self.T // self.M))  # t - 1 is used when t has 1-based indexing.
        cos_inner /= self.T // self.M
        cos_out = np.cos(cos_inner) + 1
        return float(self.alpha_zero / 2 * cos_out)

class SWA(tensorflow.keras.callbacks.Callback):
    
    def __init__(self, filepath, swa_epoch):
        super(SWA, self).__init__()
        self.filepath = filepath
        self.swa_epoch = swa_epoch 
    
    def on_train_begin(self, logs=None):
        self.nb_epoch = self.params['epochs']
        print('Stochastic weight averaging selected for last {} epochs.'
              .format(self.nb_epoch - self.swa_epoch))
        
    def on_epoch_end(self, epoch, logs=None):
        
        if epoch == self.swa_epoch:
            self.swa_weights = self.model.get_weights()
            
        elif epoch > self.swa_epoch:    
            for i in range(len(self.swa_weights)):
                self.swa_weights[i] = (self.swa_weights[i] * 
                    (epoch - self.swa_epoch) + self.model.get_weights()[i])/((epoch - self.swa_epoch)  + 1)  
        else:
            pass
        
    def on_train_end(self, logs=None):
        self.model.set_weights(self.swa_weights)
        print('Final model parameters set to stochastic weight average.')
        self.model.save_weights(self.filepath)
        print('Final stochastic averaged weights saved to file.')

snapshot = SnapshotCallbackBuilder(nb_epochs=EPOCHS,nb_snapshots=1,init_lr=INIT_LR)


# Base model and Head model joined and the basemodel's top layers are frozen (to preserve pre-trained weights)

# In[ ]:


x = baseModel.output
x = AveragePooling2D(pool_size=(7, 7))(x)
x = Flatten(name="flatten")(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(2, activation="softmax")(x)
model = Model(inputs=baseModel.input, outputs=x)
for layer in baseModel.layers:
    layer.trainable = False


# Training the above defined network with callbacks

# In[ ]:


import tensorflow

opt = Adam(lr=1e-7)
model.compile(loss="binary_crossentropy", optimizer=opt,metrics=["accuracy"])

chk=tensorflow.keras.callbacks.ModelCheckpoint(
 filepath='model.h5',
 monitor='val_loss',
verbose=0,
save_best_only=True,
mode='min')

reduce=tensorflow.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.1,
    patience=3,
    verbose=0,
    mode="min",
    min_delta=0.0001,
    cooldown=0,
    min_lr=1e-9,
)

H = model.fit(
datagen.flow(features, train_labels, batch_size=BS),
steps_per_epoch=len(train_data) // BS,
validation_data=(test_features, test_labels),
validation_steps=len(test_features) // BS,
epochs=30,
callbacks=[chk,reduce]+snapshot.get_callbacks())


# Loading model with least val_loss

# In[ ]:


from tensorflow.keras.models import load_model
model=load_model('model.h5')


# Running Inference on Validation data

# In[ ]:


model.evaluate(test_features,final_test_labels)


# Loading Face detection model available in OpenCV

# In[ ]:


import cv2 as cv
prototxtpath='../input/prototxt/deploy.prototxt.txt'
weights_path='../input/face-files/res10_300x300_ssd_iter_140000.caffemodel'
face_net=cv.dnn.readNet(prototxtpath,weights_path)
mask_model=model


# In[ ]:

test_csv_path="" #Add your absolute path here where test.csv exists
import pandas as pd
submit=pd.read_csv(test_csv_path)


# In[ ]:


submit.drop_duplicates(subset='name',keep='first',inplace=True)
submit.reset_index(drop=True,inplace=True)


# Running human face detection and mask recognition on unseen test set!

# The predicted Bounding box coordinates and the predicted class in stored in the lists x1,y1,x2,y2 and classes respectively

# In[ ]:


import matplotlib.pyplot as plt
import numpy as np
classes = []
filename   = []
x1 = []
y1 = []
x2 = []
y2 = []
for n,i in enumerate(submit['name']):
    direc = os.path.sep.join([image_dir,'/Medical Mask/images/%s'%i])
    image = cv2.imread(direc)
    image2 =cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    image3 = image2.copy()
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1.0,(300, 300), (104.0, 117.0, 123.0))
    face_net.setInput(blob)
    detections = face_net.forward()
    for j in range(0, detections.shape[2]):

        confidence = detections[0, 0, j, 2]

        if confidence > 0.5:
            box = detections[0, 0, j, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
            if startX > w or startY > h:
                break
            face = image2[startY:endY, startX:endX]
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face=preprocess_input(face)
            face=np.expand_dims(face,axis=0)
            
            (mask, withoutMask) = mask_model.predict(face)[0]
            
            label = "With_mask" if mask > withoutMask else "Without_mask"
        
            classes.append(label)
            filename.append(i)
            x1.append(startX)
            y1.append(startY)
            x2.append(endX)
            y2.append(endY)


# Saving the above result in a csv file.

# In[ ]:


name = pd.Series(filename)
x1 = pd.Series(x1)
y1 = pd.Series(y1)
x2 = pd.Series(x2)
y2 = pd.Series(y2)
classes =  pd.Series(classes)
submission = pd.DataFrame({ 'name': name, 'x1': x1,'y1': y1, 'x2': x2,'y2': y2,'classname': classes })

#This csv file will hold the predictions for all the unlabelled data in test.csv file
# In[ ]:


submission.to_csv('submission.csv')    


