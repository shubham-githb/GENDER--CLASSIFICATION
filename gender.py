from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
import numpy as np
import matplotlib.pyplot as plt
import random
import cv2
import os
import glob
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.layers import BatchNormalization, Conv2D,MaxPooling2D,Activation,Flatten,Dense,Dropout
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam

#initial parameters
epochs = 100
lr = 1e-3
batch_size = 64
img_dims = (96,96,3)

data = []
labels = []

# loading the data sets
image_files = [f for f in glob.glob(r'C:\Users\Shubham\Desktop\files\gender_dataset_face'+ "/**/*", recursive=True) if not os.path.isdir(f)]
random.shuffle(image_files)

#converting the image into np array format for keras to process it
for img in image_files:
    # running all over the loop
    image = cv2.imread(img)
    # reading the file of images
    # resizing the images in order to bring all of it in one dimensional values
    image = cv2.resize(image,(img_dims[0],img_dims[1]))
    # converting images to img_to_array (numpy)
    image = img_to_array(image)
    data.append(image)

#     seperating the labels to target the images file with os.path.sep()

    label = img.split(os.path.sep)[-2]
    if label == "woman" :
        label = 1      #giving labels a numerical forms
    else:
        label = 0

    labels.append([label])

###Pre-Processing
data = np.array(data, dtype="float")/255.0
labels = np.array(labels)

#############Train-Test-spliting#############
(trainX,testX,trainY,testY) = train_test_split(data,labels,test_size=0.2,random_state=42)

trainY = to_categorical(trainY,num_classes=2)
testY = to_categorical(testY,num_classes=2)

#####Data-Augmentation ##########################

datagen = ImageDataGenerator(rotation_range=25,
                             width_shift_range=0.1,
                             height_shift_range=0.2,
                             zoom_range=0.2,
                             horizontal_flip=True,
                             fill_mode="nearest")

####DEFINING THE MODEl#######

def build(width, height, depth, classes):
    model = Sequential()
    inputShape = (height, width, depth)
    chanDim = -1

    if K.image_data_format() == "channels_first":
        inputShape = (depth, height,width)
        chanDim = 1

    model.add(Conv2D(32,(3,3),padding="same",input_shape=inputShape))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(3,3)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))

    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))

    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(axis=chanDim))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation("relu"))
    model.add(BatchNormalization())
    model.add(Dropout(0.25))

    model.add(Dense(classes))
    model.add(Activation("sigmoid"))

    return model


######### build model ################################################

model = build(width=img_dims[0], height=img_dims[1], depth=img_dims[2],
                            classes=2)

#####complie the model

opt = Adam(lr=lr,decay=lr/epochs)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])


######### Train the model ########

H = model.fit_generator(datagen.flow(trainX,trainY,batch_size=batch_size),
                        validation_data=(testX,testY),
                        steps_per_epoch=len(trainX) // batch_size,
                        epochs=epochs,
                        verbose=1)

## saving the model to the disk
model.save('gender_detection.model')



# save the model to disk
model.save('gender_detection.model')

# plot training/validation loss/accuracy
plt.style.use("ggplot")
plt.figure()
N = epochs
plt.plot(np.arange(0,N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0,N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0,N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0,N), H.history["val_acc"], label="val_acc")

plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper right")

# save plot to disk
plt.savefig('plot.png')





























































