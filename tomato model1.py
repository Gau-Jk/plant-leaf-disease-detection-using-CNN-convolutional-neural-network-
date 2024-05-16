import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator

train_dir = "C:/Users/gauth/OneDrive/Desktop/SPLITdata/train"
test_dir = "C:/Users/gauth/OneDrive/Desktop/SPLITdata/val"

data_augmentation = tf.keras.Sequential([
  tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal",
    input_shape=(128, 128, 3)),
  tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
  tf.keras.layers.experimental.preprocessing.RandomZoom(0.2),
  tf.keras.layers.experimental.preprocessing.RandomHeight(0.2),
  tf.keras.layers.experimental.preprocessing.RandomWidth(0.2),
  tf.keras.layers.RandomContrast(0.5),
  tf.keras.layers.RandomBrightness(0.5),
  tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
], name ="data_augmentation")


IMG_SIZE = (128, 128)
BATCH_SIZE = 32
train_data = tf.keras.preprocessing.image_dataset_from_directory(
    directory = train_dir,
    image_size = IMG_SIZE,
    label_mode = 'categorical',
    batch_size = BATCH_SIZE,
    shuffle = True,
    seed=42
).cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)




test_data = tf.keras.preprocessing.image_dataset_from_directory(
    directory = test_dir,
    image_size = IMG_SIZE,
    label_mode = 'categorical',
    batch_size = BATCH_SIZE
)

class_names = test_data.class_names
valid_data = test_data.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
checkpoint_path = "CheckPoint/cp.ckpt"
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    checkpoint_path,
    save_weights_only=True,
    monitor='val_accuracy',
    save_best_only=True
)

classifier = Sequential()
classifier.add(data_augmentation)
classifier.add(Conv2D(128,(5,5), input_shape = (128,128,3),activation = 'relu'))
classifier.add(Conv2D(96,(3,3), input_shape = (128,128,3),activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))
classifier.add(Conv2D(48,(3,3), input_shape = (128,128,3),activation = 'relu'))
classifier.add(Conv2D(64,(3,3), input_shape = (128,128,3),activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))
classifier.add(Conv2D(32,(3,3), input_shape = (128,128,3),activation = 'relu'))
classifier.add(Conv2D(48,(5,5), input_shape = (128,128,3),activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2,2)))
classifier.add(Flatten())
classifier.add(Dense(units = 112, activation = 'relu'))
classifier.add(Dense(units = 10, activation = 'softmax'))
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

history = classifier.fit(
    train_data,
    epochs=30,
    steps_per_epoch=len(train_data),
    validation_data = test_data,
    validation_steps = len(test_data),
    callbacks = [
        checkpoint_callback
    ]
)

#Saving the trained model
model.save('pretrain_modelT1.h5')



IMG_SIZE = (112, 112)
BATCH_SIZE = 32
class_names=['Apple___Apple_scab',
 'Apple___Black_rot',
 'Apple___Cedar_apple_rust',
 'Apple___healthy',]
# Load the model (Replace 'load_model()' with your actual method of loading the model)
load_model = tf.keras.models.load_model("C:/Users/gauth/Documents/prototypes/pretrain_model2.h5")
test_data = tf.keras.preprocessing.image_dataset_from_directory(
    directory="C:/Users/gauth/OneDrive/Desktop/test",
    image_size=IMG_SIZE,
    label_mode='categorical',
    batch_size=BATCH_SIZE,
    shuffle=False  # Set shuffle to False to keep the order of images
)
for image_batch1, label_batch in test_data.take(1):
    break
predictions = load_model.predict(np.expand_dims(image_batch1[0], axis=0))
predicted_label_index = np.argmax(predictions, axis=-1)[0]
predicted_class = class_names[predicted_label_index]
# Visualization
plt.imshow(image_batch1[0].numpy().astype("uint32"))
plt.title(f"Predict: {predicted_class}", color='g')
plt.axis("off")
plt.show()