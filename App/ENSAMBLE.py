import tensorflow as tf
import cv2
import numpy as np

input_shape = (224,224,3)
model_input =tf.keras.Input(shape=input_shape)

Base_model1 =tf.keras.applications.DenseNet201(input_shape=input_shape, input_tensor=model_input, include_top=False, weights=None)
for layer in Base_model1.layers:
    layer.trainable = True
Base_model1_last_layer = Base_model1.get_layer('relu')
print('last layer output shape:',Base_model1_last_layer.output_shape)
Base_model1_last_output = Base_model1_last_layer.output
x1 =tf.keras.layers.GlobalAveragePooling2D()(Base_model1_last_output)
x1 =tf.keras.layers.Dropout(0.25)(x1)
x1 =tf.keras.layers.Dense(512, activation='relu')(x1)
x1 =tf.keras.layers.Dropout(0.25)(x1)
final_output1 =tf.keras.layers.Dense(4, activation='softmax', name='final_output')(x1)
DensNet201_model =tf.keras.models.Model(model_input, final_output1)
metric_list = ["accuracy"]
optimizer =tf.keras.optimizers.Adam(lr= 1.2500e-05)
DensNet201_model.compile(optimizer=optimizer, loss="categorical_crossentropy",  metrics=metric_list)
DensNet201_model.load_weights('WEIGHT/Weight_DensNet201_Optimal_Ori.h5')



Base_model2 =tf.keras.applications.InceptionV3(input_shape=input_shape, input_tensor=model_input, include_top=False, weights=None)
for layer in Base_model2.layers:
    layer.trainable = True
Base_model2_last_layer = Base_model2.get_layer('mixed10')
print('last layer output shape:', Base_model2_last_layer.output_shape)
Base_model2_last_output = Base_model2_last_layer.output
x2 =tf.keras.layers.GlobalAveragePooling2D()(Base_model2_last_output)
x2 =tf.keras.layers.Dropout(0.25)(x2)
x2 =tf.keras.layers.Dense(1024, activation='relu')(x2)
x2 =tf.keras.layers.Dropout(0.25)(x2)
final_output2 =tf.keras.layers.Dense(4, activation='softmax', name='final_output2')(x2)
InceptionV3_model =tf.keras.models.Model(model_input, final_output2)
metric_list = ["accuracy"]
optimizer = tf.keras.optimizers.Adam(1.0000e-06)
InceptionV3_model.compile(optimizer=optimizer, loss="categorical_crossentropy",  metrics=metric_list)
InceptionV3_model.load_weights('WEIGHT/Weight_InceptionV3_Optimal_Ori.h5')


Base_model3 =tf.keras.applications.MobileNetV2(input_shape=input_shape, input_tensor=model_input, include_top=False, weights=None)
for layer in Base_model3.layers:
    layer.trainable = True
Base_model3_last_layer = Base_model3.get_layer('out_relu')
print('last layer output shape:', Base_model3_last_layer.output_shape)
Base_model3_last_output = Base_model3_last_layer.output
x3 =tf.keras.layers.GlobalAveragePooling2D()(Base_model3_last_output)
x3 =tf.keras.layers.Dropout(0.5)(x3)
x3 =tf.keras.layers.Dense(512, activation='relu')(x3)
x3 =tf.keras.layers.Dropout(0.5)(x3)
final_output3 =tf.keras.layers.Dense(4, activation='softmax', name='final_output3')(x3)
MobileNetV2_model =tf.keras.models.Model(model_input, final_output3)
metric_list = ["accuracy"]
optimizer = tf.keras.optimizers.Adam(lr= 6.2500e-06)
MobileNetV2_model.compile(optimizer=optimizer, loss="categorical_crossentropy",  metrics=metric_list)
MobileNetV2_model.load_weights('WEIGHT/Weight_MobileNetV2_Optimal_(Ori).h5')

def ensemble(models, model_input):
    outputs = [model.outputs[0] for model in models]
    y =tf.keras.layers.Average()(outputs)
    model =tf.keras.Model(model_input,y,name='ensemble')
    return model

ensemble_model = ensemble([DensNet201_model,InceptionV3_model,MobileNetV2_model], model_input)
metric_list = ["accuracy"]
optimizer =tf.keras.optimizers.Adam(lr=2.5000e-05)
ensemble_model.compile(optimizer=optimizer, loss="categorical_crossentropy",  metrics=metric_list)

IMG_SIZE=(224,224)
path='C:/Users/mariotiara/Desktop/GUI/Sampel/ffc04fed30e6_(0)_Normal.jpeg'
img=tf.keras.preprocessing.image.load_img(path,target_size=IMG_SIZE)
x = tf.keras.preprocessing.image.img_to_array(img)
x = np.expand_dims(x, axis=0)
Predict=ensemble_model.predict(x)

print("Normal:",Predict[0][0]*100)
print("Mild:",Predict[0][1]*100)
print("Moderate:",Predict[0][2]*100)
print("Severe:",Predict[0][3]*100)

