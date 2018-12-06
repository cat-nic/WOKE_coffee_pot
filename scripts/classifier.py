from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

class Woke:
    def __init__(self):
        self.classifier = Sequential()

    def build(self, img_size=64):
        self.classifier.add(Conv2D(img_size//2, 3, 3, input_shape = (img_size, img_size, 3), activation = 'relu'))
        # Step 2 - PooliPooling
        self.classifier.add(MaxPooling2D(pool_size = (2, 2)))
        # Adding a seconsecond convolutional layer
        self.classifier.add(Conv2D(img_size//2, 3, 3, activation = 'relu'))
        self.classifier.add(MaxPooling2D(pool_size = (2, 2)))
        # Step 3 - FlattFlattening
        self.classifier.add(Flatten())
        # Step 4 - Full Full connection
        self.classifier.add(Dense(output_dim = img_size, activation = 'relu'))
        self.classifier.add(Dense(output_dim = 3, activation = 'softmax'))

    def compile(self,optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy']):
        self.classifier.compile(optimizer = optimizer, loss = loss, metrics = metrics)
