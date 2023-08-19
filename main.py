from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from sklearn.metrics import classification_report
import os
import matplotlib.pyplot as plt
import numpy as np
from keras.optimizers import Adam 


train_data_dir=r"D:\ai project\Facial Emotion\data\test"
validation_data_dir=r"D:\ai project\Facial Emotion\data\train"

train_datagen =ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest")

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    color_mode='grayscale',
    target_size=(48, 48),
    batch_size=32,
    class_mode='categorical',
    shuffle=True)

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    color_mode='grayscale',
    target_size=(48, 48),
    batch_size=32,
    class_mode='categorical',
    shuffle=True)

class_labels=['angry','disgusted', 'fearful', 'happy','neutral','sad','surprised']

img, label = train_generator.__next__()

model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))

model.add(Conv2D(512, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))


model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(7, activation='softmax'))

INIT_LR = 1e-4
epochs=30
#BS = 32

opt = Adam(lr=INIT_LR, decay=INIT_LR / epochs)
model.compile(optimizer = 'adam', loss='categorical_crossentropy', metrics=['accuracy'])
print(model.summary())

train_path = r"D:\ai project\Facial Emotion\data\test"
test_path = r"D:\ai project\Facial Emotion\data\train"

num_train_imgs = 0
for root, dirs, files in os.walk(train_path):
    num_train_imgs += len(files)
    
num_test_imgs = 0
for root, dirs, files in os.walk(test_path):
    num_test_imgs += len(files)

print(num_train_imgs)
print(num_test_imgs)



history = model.fit(train_generator,
                    steps_per_epoch=num_train_imgs//32,
                    epochs=epochs,
                    validation_data=validation_generator,
                    validation_steps=num_test_imgs//32)

model.save('model_file_30epochs.h5')

# Get predictions for validation data
Y_pred = model.predict(validation_generator, num_test_imgs // 32 + 1)
y_pred = np.argmax(Y_pred, axis=1)

# Get true labels for validation data
validation_generator.reset()
Y_true = validation_generator.classes

# Generate classification report
report = classification_report(Y_true, y_pred, target_names=class_labels)
print(report)
