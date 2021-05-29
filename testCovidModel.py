import os

import numpy as np
from tensorflow import keras
from keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model
import tensorflow as tf

def predict_image(class_names, model, filename, img_height, img_width):
    try:
        img = load_img(filename, target_size=(img_height, img_width))
    except:
        return 0
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch
    predictions = model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
            .format(class_names[np.argmax(score)], 100 * np.max(score))
    )


class_names = ['COVID', 'NORMAL', 'PNEUMONIA']
model = load_model('preTrainedCovidModel.h5')

for root, dirs, files in os.walk("./chest_xray3/test/NORMAL/"):
    for filename in files:
        predict_image(class_names, model, "./chest_xray3/test/NORMAL/" + filename, 180, 180)

