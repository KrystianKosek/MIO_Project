import tensorflow as tf
from keras.preprocessing.image import load_img
from matplotlib import pyplot as plt
from tensorflow import keras
from tensorflow.keras.models import load_model


def predict_image(filename, img_height, img_width):
    img = load_img(filename, target_size=(img_height, img_width))
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch
    model = load_model('model3.h5')

    predictions = model.predict_on_batch(img_array).flatten()
    # Apply a sigmoid since our model returns logits
    predictions = tf.nn.sigmoid(predictions)
    print(predictions)
    predictions = tf.where(predictions < 0.5, 0, 1)
    if predictions == 1:
        print("predicted: PNEUMONIA")
    else:
        print("predicted: NORMAL")
    plt.imshow(img)
    plt.show()


predict_image("./chest_xray/test/NORMAL/NORMAL2-IM-0373-0001.jpeg", 180, 180)
