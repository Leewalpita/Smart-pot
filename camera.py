import tensorflow as tf
import cv2
import numpy as np
import urllib.request  # Import urllib.request for Python 3
import os
import numpy as np

url = "http://192.168.55.39:8080/photo.jpg"
imgResp = urllib.request.urlopen(url)  # Fetch the image from the URL
imgNp = np.array(bytearray(imgResp.read()), dtype=np.uint8)  # Convert the image to a numpy array
img = cv2.imdecode(imgNp, -1)  # Decode the image

# Display the image
cv2.imshow("test", img)
cv2.waitKey(10)  # Wait for 10 milliseconds

# Save the image to your project directory
cv2.imwrite("C:/Users/User/Documents/Jason/ML project/real time check/captured_image.jpg", img)

# Clean up and close the window
cv2.destroyAllWindows()

# Load the trained model
model_path = 'C:/Users/User/Documents/Jason/ML project/save/my_model2_new.h5'
model = tf.keras.models.load_model(model_path)
print("....................loaded.......................")
input_folder = "C:/Users/User/Documents/Jason/ML project/real time check"
image_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.jpg')]

# Ensure the classes are mapped correctly
train_datagen_vg19 = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
train_path = "C:/Users/User/Documents/Jason/ML project/data set 2/tomato/train"
trainning_set_vg19 = train_datagen_vg19.flow_from_directory(train_path,
                                                            target_size=(128, 128),
                                                            batch_size=20,
                                                            class_mode="sparse", 
                                                            shuffle=True)

class_indices = {v: k for k, v in trainning_set_vg19.class_indices.items()}
class_names = [class_indices[i] for i in range(len(class_indices))]

def load_and_preprocess_image(img_path, img_height=128, img_width=128):
    # Load the image and resize it
    img = tf.keras.utils.load_img(img_path, target_size=(img_height, img_width))
    
    # Convert the image to a numpy array
    img_array = tf.keras.utils.img_to_array(img)
    
    # Expand dimensions to create a batch of size 1
    img_array = np.expand_dims(img_array, axis=0)
    
    # Rescale the image (normalize pixel values to [0, 1])
    img_array = img_array / 255.0
    
    return img_array
# Iterate over images and predict
for img_path in image_files:
    img_array = load_and_preprocess_image(img_path)
    predictions = model.predict(img_array)
    
    # Find the predicted class index
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    
    # Convert the index to the actual class name
    predicted_label = class_names[predicted_class_index]
    
    # Print the image path and the prediction
    print(f"Image Path: {img_path}")
    print(f"Predicted Label: {predicted_label}\n")




