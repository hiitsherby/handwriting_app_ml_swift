import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten

pixel_width = 28
pixel_height = 28
batch_size = 32
epochs = 10

num_of_classes = 10

(features_train, labels_train), (features_test, labels_test) = mnist.load_data()

print(features_train.shape)

features_train = features_train.reshape(features_train.shape[0], pixel_width, pixel_height, 1)
features_test = features_test.reshape(features_test.shape[0], pixel_width, pixel_height, 1)

input_shape = (pixel_width, pixel_height, 1)

features_train = features_train.astype('float32')
features_test = features_test.astype('float32')

features_train /= 255
features_test /= 255

labels_train = keras.utils.to_categorical(labels_train, num_of_classes)
labels_test = keras.utils.to_categorical(labels_test, num_of_classes) #binary

model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3), activation = 'relu', input_shape = input_shape))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(num_of_classes, activation='softmax'))
print(model.output_shape)

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(features_train, labels_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(features_test, labels_test))

score = model.evaluate(features_test, labels_test, verbose=0)

model.save('/Users/apple/Desktop/handwriting-classifier.h5')#h5 type

import coremltools
coreml_model = coremltools.converters.keras.convert(model, input_names=['image'], image_input_names='image')

coreml_model.author = 'me'
coreml_model.license = 'MIT'
coreml_model.short_description = 'des'
coreml_model.input_description['image'] = '28x28 greyscale image'
coreml_model.output_description['output1'] = 'des' 
coreml_model.save('/Users/apple/Desktop/Handwriting.mlmodel')






















