import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import *
from tensorflow.keras.models import * 
from tensorflow.keras.preprocessing import image

import cv2 #For Image processing 
import numpy as np #For converting Images to Numerical array 
import os #To handle directories 
from PIL import Image #Pillow lib for handling images 

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()

Face_ID = -1 
pev_person_name = ""
y_ID = []
x_train = []

Face_Images = os.path.join(os.getcwd(), "dataset/train") #Tell the program where we have saved the face images 
print (Face_Images)


for root, dirs, files in os.walk(Face_Images): #go to the face image directory 
	for file in files: #check every directory in it 
		if file.endswith("jpeg") or file.endswith("jpg") or file.endswith("png"): #for image files ending with jpeg,jpg or png 
			path = os.path.join(root, file)
			person_name = os.path.basename(root)
			print(path, person_name)

			
			if pev_person_name!=person_name: #Check if the name of person has changed 
				Face_ID=Face_ID+1 #If yes increment the ID count 
				pev_person_name = person_name

			
			Gery_Image = Image.open(path).convert("L") # convert the image to greysclae using Pillow
			Crop_Image = Gery_Image.resize( (550,550) , Image.ANTIALIAS) #Crop the Grey Image to 550*550 (Make sure your face is in the center in all image)
			Final_Image = np.array(Crop_Image, "uint8")
			#print(Numpy_Image)
			faces = face_cascade.detectMultiScale(Final_Image, scaleFactor=1.5, minNeighbors=5) #Detect The face in all sample image 
			print (Face_ID,faces)

			for (x,y,w,h) in faces:
				roi = Final_Image[y:y+h, x:x+w] #crop the Region of Interest (ROI)
				x_train.append(roi)
				y_ID.append(Face_ID)

recognizer.train(x_train, np.array(y_ID)) #Create a Matrix of Training data 
recognizer.save("face-trainner.yml") #Save the matrix as YML file 


import tensorflow as tf
#Training model
model = Sequential()   ## creating a blank model
model.add(Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=(224,224,3)))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))    ### reduce the overfitting

model.add(Flatten())    ### input layer
model.add(Dense(256,activation='relu'))    ## hidden layer of ann
model.add(Dropout(0.5))
model.add(Dense(2,activation=tf.math.softmax))   ## output layer

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

model.summary()

#Moulding train images
train_datagen = image.ImageDataGenerator(rescale = 1./255, shear_range = 0.2,zoom_range = 0.2, horizontal_flip = True)

test_dataset = image.ImageDataGenerator(rescale=1./255)

#Reshaping test and validation images 
train_generator = train_datagen.flow_from_directory(
    directory = 'dataset/train',
    target_size = (224,224),
    batch_size = 15,
    class_mode = 'categorical')
validation_generator = test_dataset.flow_from_directory(
    directory = 'dataset/val',
    target_size = (224,224),
    batch_size = 15,
    class_mode = 'categorical')
	
#### Train the model
history = model.fit_generator(
    train_generator,
    steps_per_epoch=7,
    epochs = 10,
    validation_data = validation_generator)
	
print("Training Ended")
model.save("training.h5")
plt.figure(figsize=(20,10))
plt.subplot(1, 2, 1)
plt.suptitle('Optimizer : Adam', fontsize=10)
plt.ylabel('Loss', fontsize=16)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend(loc='upper right')

plt.subplot(1, 2, 2)
plt.ylabel('Accuracy', fontsize=16)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend(loc='lower right')
plt.show()
