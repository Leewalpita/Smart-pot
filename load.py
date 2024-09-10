import tensorflow as tf
import cv2
import numpy as np
import os

model_path = 'C:/Users/User/Documents/Jason/ML project/save/my_model1_new.h5'
model = tf.keras.models.load_model(model_path)
model.summary()
print("loaded")

path = "C:/Users/User/Documents/Jason/ML project/data set 2/tomato"
os.listdir(path)
train_path = os.path.join(path, "train")
print(os.listdir(train_path))
print("*" * 100)
test_path = os.path.join(path, "val")
print(os.listdir(test_path))

def prepare(filepath):
    img_array = cv2.imread(filepath, cv2.IMREAD_COLOR)
    if img_array is None:
        print(f"Error loading image: {filepath}")
        return None
    img_array = img_array / 255.0  # Normalize the pixel values to [0, 1]
    new_array = cv2.resize(img_array, (128, 128))  # Resize to the input size expected by the model
    return new_array.reshape(-1, 128, 128, 3)

class_dict = {
    'Tomato___Bacterial_spot': 0,
    'Tomato___Early_blight': 1,
    'Tomato___healthy': 2,
    'Tomato___Late_blight': 3,
    'Tomato___Leaf_Mold': 4,
    'Tomato___Septoria_leaf_spot': 5,
    'Tomato___Spider_mites Two-spotted_spider_mite': 6,
    'Tomato___Target_Spot': 7,
    'Tomato___Tomato_mosaic_virus': 8,
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 9
}

def prediction_cls(prediction):
    for key, clss in class_dict.items():
        if np.argmax(prediction) == clss:
            return key

image_paths = ["C:/Users/User/Documents/Jason/ML project/archive/tomato/val/Tomato___Early_blight/0bd357fe-1e54-4c65-979c-e894e0b8a3aa___RS_Erly.B 8328.JPG",
    "C:/Users/User/Documents/Jason/ML project/archive/tomato/val/Tomato___Septoria_leaf_spot/Tomato___Septoria_leaf_spot_original_0cba2d69-c73b-472b-9fc1-beca9b3f02a0___JR_Sept.L.S 8369.JPG_45f2453b-4703-401a-a75f-c17b918649a9.JPG",
    "C:/Users/User/Documents/Jason/ML project/archive/tomato/val/Tomato___healthy/0aacdad5-c9b9-4309-96e3-0797bbed1375___RS_HL 9836.JPG",
    "C:/Users/User/Documents/Jason/ML project/archive/tomato/val/Tomato___Bacterial_spot/0b13b997-9957-4029-b2a4-ef4a046eb088___UF.GRC_BS_Lab Leaf 0595.JPG",
    "C:/Users/User/Documents/Jason/ML project/archive/tomato/val/Tomato___Late_blight/0ab1cab4-a0c9-4323-9a64-cdafa4342a9b___GHLB2 Leaf 8918.JPG",
    "C:/Users/User/Documents/Jason/ML project/archive/tomato/val/Tomato___Leaf_Mold/0b943ada-01a9-4ce0-a607-e799394856de___Crnl_L.Mold 7008.JPG",
    "C:/Users/User/Documents/Jason/ML project/archive/tomato/val/Tomato___Tomato_mosaic_virus/Tomato___Tomato_mosaic_virus_original_01b32f27-2b9b-4961-805b-8066bf4d90f1___PSU_CG 2417.JPG_d0f26ba8-c956-4de8-94b5-72adc5fcf232.JPG",
    "C:/Users/User/Documents/Jason/ML project/archive/tomato/val/Tomato___Tomato_Yellow_Leaf_Curl_Virus/1b80b934-9b73-4c3b-8306-8058b59e766b___YLCV_GCREC 2027.JPG",
    "C:/Users/User/Documents/Jason/ML project/archive/tomato/val/Tomato___Spider_mites Two-spotted_spider_mite/0b5cda10-da2f-4647-b159-69647b42212f___Com.G_SpM_FL 1784.JPG"
    "C:U/sers/User/Documents/Jason/ML project/archive/tomato/val/Tomato___Target_Spot/0b126ce6-af82-477f-8f4e-1de79d84a6dd___Com.G_TgS_FL 8294.JPG"
]

# Loop through each image path and make prediction
for image_path in image_paths:
    prepared_image = prepare(image_path)
    if prepared_image is not None:
        prediction = model.predict(prepared_image)
        print(prediction_cls(prediction))
    else:
        print(f"Skipping image: {image_path} due to loading error.")

print(model.classes)