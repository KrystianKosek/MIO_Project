import numpy as np
from matplotlib import pyplot as plt
from tensorflow import keras
from keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model
import tensorflow as tf

def predict_image(filename, img_height, img_width):
    img = load_img(filename, target_size=(img_height, img_width))
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch
    model = load_model('model2.h5')
    predictions = model.predict(img_array)
    print(predictions[0])
    if predictions[0] > 0.5:
        print("predicted: PNEUMONIA")
    else:
        print("predicted: NORMAL")
    plt.imshow(img)
    plt.show()

predict_image("./chest_xray/test/NORMAL/IM-0023-0001.jpeg", 180, 180)