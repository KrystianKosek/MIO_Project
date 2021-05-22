from keras.engine.saving import load_model
from matplotlib import pyplot as plt
from tensorflow import keras
from keras.preprocessing.image import load_img

def predict_image(filename, img_height, img_width):
    img = load_img(filename, target_size=(img_height, img_width))
    image = keras.preprocessing.image.img_to_array(img)
    image = image / 255.0
    image = image.reshape(1, 180, 180, 3)
    model = load_model('model.h5')
    prediction = model.predict(image)
    plt.imshow(img)
    if (prediction[0] > 0.5):
        print("predicted: PNEUMONIA")
    else:
        print("predicted: NORMAL")
    plt.show()

predict_image("./chest_xray/train/PNEUMONIA/person5_bacteria_15.jpeg", 180, 180)