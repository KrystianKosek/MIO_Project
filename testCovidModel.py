import os

import numpy as np
from tensorflow import keras
from keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model
import tensorflow as tf

def predict_image(filename, img_height, img_width):
    class_names = ['COVID', 'NORMAL', 'PNEUMONIA']

    try:
        img = load_img(filename, target_size=(img_height, img_width))
    except:
        return 0
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch
    model = load_model('preTrainedCovidModel.h5')
    predictions = model.predict(img_array)
    print(predictions)
    score = tf.nn.softmax(predictions[0])

    print(
        "This image most likely belongs to {} with a {:.2f} percent confidence."
            .format(class_names[np.argmax(score)], 100 * np.max(score))
    )

for root, dirs, files in os.walk("./chest_xray3/test/COVID/"):
    for filename in files:
        predict_image("./chest_xray3/test/COVID/" + filename, 180, 180)
