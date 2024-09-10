import tensorflow as tf 
import pandas as pd
import numpy as np
import plotly.express as px
import os
from keras import layers, models
from glob import glob
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.preprocessing import LabelEncoder

path = "C:/Users/User/Documents/Jason/ML project/data set 2/tomato"
print(os.listdir(path))

train_path = os.path.join(path, "train")
print(os.listdir(train_path))
print("*"*100)
test_path = os.path.join(path, "val")
print(os.listdir(test_path))

folders = glob("C:/Users/User/Documents/Jason/ML project/data set 2/tomato/train/*")
print(folders)

# Load and display an image
image = Image.open("C:/Users/User/Documents/Jason/ML project/data set 2/tomato/train/Tomato___Septoria_leaf_spot/663f412f-6bb7-4e89-aac6-683d87e92638___JR_Sept.L.S 8376.JPG")
plt.imshow(image)
plt.show()

# Define the CNN model
cnn = models.Sequential()
cnn.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(128, 128, 3)))
cnn.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same'))
cnn.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))

cnn.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
cnn.add(layers.Conv2D(64, (3, 3), activation='relu', padding='same'))
cnn.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))

cnn.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
cnn.add(layers.Conv2D(128, (3, 3), activation='relu', padding='same'))
cnn.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
cnn.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))

cnn.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
cnn.add(layers.Conv2D(256, (3, 3), activation='relu', padding='same'))
cnn.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
cnn.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))

cnn.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
cnn.add(layers.Conv2D(512, (3, 3), activation='relu', padding='same'))
cnn.add(layers.MaxPooling2D((2, 2), strides=(2, 2)))

cnn.add(layers.Flatten())
cnn.add(layers.Dense(1024, activation='relu'))
cnn.add(layers.Dropout(0.5))
cnn.add(layers.Dense(512, activation='relu'))
cnn.add(layers.Dropout(0.5))
cnn.add(layers.Dense(10, activation='softmax'))

cnn.summary()

# Data generators
train_datagen_vg19 = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
test_datagen_vg19 = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

trainning_set_vg19 = train_datagen_vg19.flow_from_directory(train_path,
                                                            target_size=(128, 128),
                                                            batch_size=20,
                                                            class_mode="sparse", 
                                                            shuffle=True)

testing_set_vg19 = test_datagen_vg19.flow_from_directory(test_path,
                                                         target_size=(128, 128),
                                                         batch_size=20,
                                                         class_mode="sparse", 
                                                         shuffle=False)

# Encode the labels using LabelEncoder
label_encoder = LabelEncoder()
label_encoder.fit(trainning_set_vg19.classes)

trainning_set_vg19.classes = label_encoder.transform(trainning_set_vg19.classes)
testing_set_vg19.classes = label_encoder.transform(testing_set_vg19.classes)

# Compile the model
learning_rate = 1e-5
adam_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
cnn.compile(loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
            optimizer=adam_optimizer)

# Callbacks
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)

# Train the model
r_vg19 = cnn.fit(trainning_set_vg19,
                 validation_data=testing_set_vg19,
                 epochs=50,
                 callbacks=[callback])

# Plotting results
accuracy = r_vg19.history['accuracy']
val_accuracy = r_vg19.history['val_accuracy']
loss = r_vg19.history['loss']
val_loss = r_vg19.history['val_loss']
epochs = range(len(accuracy))

plt.title("Model Accuracy")
plt.plot(epochs, accuracy, "b", label="Training Accuracy")
plt.plot(epochs, val_accuracy, "r", label="Validation Accuracy")
plt.legend()
plt.show()

plt.title("Model Loss")
plt.plot(epochs, loss, "b", label="Training Loss")
plt.plot(epochs, val_loss, "r", label="Validation Loss")
plt.legend()
plt.show()



# Save the model
model_path = 'C:/Users/User/Documents/Jason/ML project/save/my_model2_new.h5'
cnn.save(model_path)

# Predictions
predictions = cnn.predict(testing_set_vg19)

# Decode predictions
class_labels = label_encoder.inverse_transform(np.arange(10))
predicted_class_indices = np.argmax(predictions, axis=1)
predicted_labels = label_encoder.inverse_transform(predicted_class_indices)

print(predicted_labels)
