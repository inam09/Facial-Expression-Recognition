# 888 percent avg accuracy on jafee
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import numpy as np
from sklearn.metrics import confusion_matrix,classification_report
import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle
IMG_SIZE = 227
new_path='C:/Users/inam.qadir/AppData/Local/Continuum/anaconda3/Library/etc/haarcascades/'
face_cascade = cv2.CascadeClassifier(new_path + 'haarcascade_frontalface_default.xml')
DATADIR="jaffe"
CARTEGORIES=["0","1","2","3","4","5","6"]
training_data=[]
for catagory in CARTEGORIES:
	path=os.path.join(DATADIR,catagory) ## path of every expression
	class_num = CARTEGORIES.index(catagory)
	for img in os.listdir(path):
		img_array=cv2.imread(os.path.join(path,img))
		faces = face_cascade.detectMultiScale(img_array, 1.3, 5)
		for (x,y,w,h) in faces:
			#print(x,y,w,h)
			face=img_array[y:y+w, x:x+h]
		new_array = cv2.resize(face, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
		training_data.append([new_array, class_num])  # add this to our training_data
		random.shuffle(training_data)
x_train=[]
y_train=[]
x_test=[]
y_test=[]
print(len(training_data))
train_Size =int(len(training_data) * 0.8)
i=0

for features,label in training_data:
    if (i<train_Size):
        x_train.append(features)
        y_train.append(label)
        i=i+1
    else:
        x_test.append(features)
        y_test.append(label)
        i=i+1
plt.imshow(x_train[0],cmap="gray")
plt.show()
print("size of training instanses",len (x_train))
print("size of training instanses",len (x_test))
x_train=np.array(x_train)
x_test=np.array(x_test)

print("shape of training set ", np.shape(x_train))
print("shape of test set ", np.shape(x_test))
pickle_out = open("x_train.pickle","wb")
pickle.dump(x_train, pickle_out)
pickle_out.close()
pickle_out = open("y_train.pickle","wb")
pickle.dump(y_train, pickle_out)
pickle_out.close()
pickle_out = open("x_test.pickle","wb")
pickle.dump(x_test, pickle_out)
pickle_out.close()
pickle_out = open("y_test.pickle","wb")
pickle.dump(y_test, pickle_out)
pickle_out.close()
#print(train_Size)
pickle_in = open("x_train.pickle","rb")
x_train = pickle.load(pickle_in)

pickle_in = open("y_train.pickle","rb")
y_train = pickle.load(pickle_in)
pickle_in = open("x_test.pickle","rb")
x_test = pickle.load(pickle_in)
pickle_in = open("y_test.pickle","rb")
y_test = pickle.load(pickle_in)

x_train = x_train/255.0
NAME="Alex net"
model = Sequential()

model.add(Conv2D(filters=96, input_shape=(x_train.shape[1:]), kernel_size=(11,11), strides=(4,4), padding='valid'))
model.add(Activation('relu'))
model.add(Conv2D(filters=128, kernel_size=(3,3), strides=(2,2), padding='valid'))
model.add(Activation('relu'))
model.add(Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding='same'))
model.add(Activation('relu'))
# Max Pooling
model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid'))
# 2nd Convolutional Layer
model.add(Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(filters=256, kernel_size=(3,3), strides=(2,2), padding='same'))
model.add(Activation('relu'))
# Max Pooling
model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid'))
# 3rd Convolutional Layer
model.add(Flatten())

# 1st Fully Connected Layer

# 2nd Fully Connected Layer
model.add(Dense(512))
model.add(Activation('relu'))
# Add Dropout
model.add(Dropout(0.4))
model.add(Dense(256))
model.add(Activation('relu'))
# Add Dropout
model.add(Dropout(0.4))
# 3rd Fully Connected Layer

# Add Dropout
#model.add(Dropout(0.4))
# Output Layer
model.add(Dense(7))
model.add(Activation('softmax'))
model.summary()
#tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))
model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=["accuracy"])
model.fit(x_train, y_train,
          batch_size=5,
          epochs=500,
          validation_split=0.2)
model.save('227x227 papernet on alex with 88.7 percent accuracy.model')
pickle_in = open("x_test.pickle","rb")
x_test = pickle.load(pickle_in)
pickle_in = open("y_test.pickle","rb")
y_test = pickle.load(pickle_in)
#plt.imshow(x_test[0],cmap="gray")
#plt.show()
#print(np.shape(x_test))
y_pred=model.predict(x_test)
y_pred=np.argmax(y_pred, axis=1)
print('Confusion Matrix')
target_names = ['neutral','Angry','disgust','fear','happy','sad','surprise']
print(target_names)
print(confusion_matrix(y_test, y_pred))