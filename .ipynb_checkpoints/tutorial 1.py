import tensorflow as tf #tensorflow installed 01.26.20
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt #loading of all APIs were successful yay!
import random

data = keras.datasets.fashion_mnist #name mnist fashion data as simply as 'data'

#"it is important to separate traning and testing data"

(train_images, train_labels), (test_images, test_labels) = data.load_data() #thanks to 'keras' we can just skip this as just data.load_data()

class_names = ['T-shirt/top', 'Pants', 'Hoodies', 'Dress', 'Coat',
 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images/255.0 #to simplify to 0.0~1.0(decimal values) instead of 0~255, we divide them by 255.0
test_images = test_images/255.0 #same as above

#print(train_images[7]) #these are pixel values of the image 0~255

#plt.imshow(train_images[7]) #this shows the just images
#plt.imshow(train_images[0], cmap=plt.cm.binary) #this shows an actual image
#plt.show() #this shows the images


#######################load and look at data complete#############################

model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(10, activation="softmax") #'softmax': add up all the values of the neuron so that they add up to '1'
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]) #they are premade optimizers, loss, metric like relu and softmax

model.fit(train_images, train_labels, epochs=5) #epochs: how many time you will see the images in a different order



prediction = model.predict(test_images)

#image_number = 13
#print(class_names[np.argmax(prediction[image_number])]) #np.argmax: makes the whicheever neuron that has highest value the thing

#plt.imshow(train_images[image_number], cmap=plt.cm.binary) #this shows an actual image
#plt.show() #this shows the images

randomlist = random.sample(range(0, 99), 5)

for n in range(4):
    i = randomlist[n]
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel("Actual: " + class_names[test_labels[i]])
    plt.title("Prediction: " + class_names[np.argmax(prediction[i])])
    plt.show()

#test_loss, test_acc = model.evaluate(test_images, test_labels)
#print("Tested Acc:", test_acc)

#######################neural network modeled#############################

