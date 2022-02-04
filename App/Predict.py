import tensorflow as tf
import cv2
import numpy as np
IMG_SIZE=(224,224)


Model=tf.keras.models.load_model ('Ensamble_Model(Ori).h5')
path='C:/Users/mariotiara/Desktop/GUI/Sampel/ffc04fed30e6_(0)_Normal.jpeg'

img=tf.keras.preprocessing.image.load_img(path,target_size=IMG_SIZE)

x = tf.keras.preprocessing.image.img_to_array(img)
x = np.expand_dims(x, axis=0)

Predict=Model.predict(x)

print("Normal:",Predict[0][0]*100)
print("Mild:",Predict[0][1]*100)
print("Moderate:",Predict[0][2]*100)
print("Severe:",Predict[0][3]*100)

