import pandas as pd
import seaborn as sn
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from tensorflow.python.keras.models import load_model

main_dir = "./chest_xray2/"
train_data_dir = main_dir + "train/"
test_data_dir = main_dir + "test/"

batch_size = 16
img_height, img_width = 180, 180
input_shape = (img_height, img_width, 3)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    test_data_dir,
    shuffle=False,
    image_size=(img_height, img_width))

model = load_model('modelForChestXray2.h5')
class_names = test_ds.class_names

y_pred = model.predict(test_ds)
predicted_categories = tf.argmax(y_pred, axis=1)

true_categories = tf.concat([y for x, y in test_ds], axis=0)

cm = confusion_matrix(true_categories, predicted_categories)

df_cm = pd.DataFrame(cm, index=[i for i in class_names],
                     columns=[i for i in class_names])
plt.figure(figsize=(10, 7))
sn.heatmap(df_cm, annot=True, cmap='Blues', fmt='g')
plt.xlabel("Prediction classes", fontsize=25)
plt.ylabel("True classes", fontsize=25)
plt.show()
