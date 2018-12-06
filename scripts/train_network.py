# set the matplotlib backend so figures can be saved in the background
import matplotlib

matplotlib.use("Agg")

# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import cv2
from classifier import Woke

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=False,
	help="path to input dataset"
                ,default="data/")
ap.add_argument("-m", "--model", required=False,
	help="path to output model"
                ,default='woke.model')
ap.add_argument("-p", "--plot", type=str, default="plot.png",
	help="path to output accuracy/loss plot")
ap.add_argument("-l", "--download",
                help="boolean whether or not to pull images from s3 to local",
                default=False)
args = vars(ap.parse_args())


# first pull down data
if args['download']:
    import subprocess
    import logging
    def simple_read_s3(bkt, directory):
        try:
            subprocess.run(['aws',
                            's3',
                            'sync',
                            f'{bkt}',
                            f'{directory}',
                            # '-- recursive'
                            ],
                           # check=True,
                           shell=True,
                           # stdout=subprocess.PIPE
                           )
            return 1
        except Exception as e:
            logging.exception(e)
            return 0

    for state in ['full', 'empty']:
        bkt = r's3://woke-coffee-pot/{}/'.format(state)
        directory = 'data/{}'.format(state)
        simple_read_s3(bkt, directory)

# initialize the number of epochs to train for, initial learning rate,
# and batch size
EPOCHS = 25
INIT_LR = 1e-3
BS = 1
img_size = 28

# initialize the data and labels
print("[INFO] loading images...")
data = []
labels = []

# grab the image paths and randomly shuffle them
imagePaths = sorted(list(paths.list_images(args["dataset"])))
random.seed(42)
random.shuffle(imagePaths)

# loop over the input images
for imagePath in imagePaths:
    # load the image, pre-process it, and store it in the data list
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (img_size, img_size))
    image = img_to_array(image)
    data.append(image)

    # extract the class label from the image path and update the
    # labels list
    label = imagePath.split('/')[1].split('\\')[0]  # TODO: CLEAN THIS UP...
    label = 1 if label == "full" else 2 if label == "half" else 0
    labels.append(label)

# scale the raw pixel intensities to the range [0, 1]
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# partition the data into training and testing splits using 75% of
# the data for training and the remaining 25% for testing
(trainX, testX, trainY, testY) = train_test_split(data,
                                                  labels, test_size=0.25, random_state=42)

# convert the labels from integers to vectors
trainY = to_categorical(trainY, num_classes=3)
testY = to_categorical(testY, num_classes=3)

# construct the image generator for data augmentation
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, fill_mode="nearest")

# initialize the model
print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
woke = Woke()
woke.build(img_size=img_size)

# Compiling the CNN
woke.compile(optimizer = opt, loss = 'categorical_crossentropy', metrics = ['accuracy'])


# train the network
print("[INFO] training network...")
print(
len(trainX) // BS,
    len(trainX),
    len(trainY)
)
H = woke.classifier.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
                        validation_data=(testX, testY), steps_per_epoch=len(trainX) // BS,
                        epochs=EPOCHS, verbose=1)
print(H)
# save the model to disk
print("[INFO] serializing network...")
woke.classifier.save(args["model"])

# plot the training loss and accuracy
plt.style.use("ggplot")
plt.figure()
N = EPOCHS
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy on WOKE Coffee Pot")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])