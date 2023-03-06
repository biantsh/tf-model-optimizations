import time

import cv2 as cv
import numpy as np
import tensorflow as tf

model_path = 'pcqat_model_2.tflite'
image_path = 'resources/image_cat_2.jpg'

interpreter = tf.lite.Interpreter(model_path)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
input_shape = input_details[0]['shape']
input_dtype = input_details[0]['dtype']
input_index = input_details[0]['index']

output_details = interpreter.get_output_details()

# Load and preprocess image
image = cv.imread(image_path)
image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
image = cv.resize(image, input_shape[2:0:-1])
image = np.expand_dims(image, 0)
image = image.astype(input_dtype)
image = image / 127.5 - 1

# Inference
time_before = time.time()
interpreter.set_tensor(input_index, image)
interpreter.invoke()
time_taken = time.time() - time_before

output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)
print('Time taken in seconds: ', time_taken)
