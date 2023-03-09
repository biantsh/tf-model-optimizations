"""Unzip a zipped TFLite model into memory and run inference

Example usage:
    python3 tflite_inference.py  \
      --model_path model.zip  \
      --images_dir images/
"""

import argparse
import glob
import os
import zipfile

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

CLASS_NAMES = ['cat', 'dog']


def main(model_path: str, images_dir: str) -> None:
    model_name, _ = os.path.splitext(model_path)
    unzipped_path = f'{model_name}.tflite'

    with zipfile.ZipFile(model_path, 'r') as zip_file:
        model_content = zip_file.read(unzipped_path)

    interpreter = tf.lite.Interpreter(model_content=model_content)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    input_shape = input_details[0]['shape']
    input_dtype = input_details[0]['dtype']
    input_index = input_details[0]['index']

    output_details = interpreter.get_output_details()

    for image_path in glob.glob(f'{images_dir}/*.jpg'):
        # Load and preprocess image
        image = cv.imread(image_path)
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        tensor = cv.resize(image, input_shape[2:0:-1])
        tensor = np.expand_dims(tensor, 0)
        tensor = tensor.astype(input_dtype)
        tensor = tensor / 127.5 - 1

        # Inference
        interpreter.set_tensor(input_index, tensor)
        interpreter.invoke()

        output_data = interpreter.get_tensor(output_details[0]['index'])
        output_data = np.squeeze(output_data)

        confidence = max(output_data)
        prediction = CLASS_NAMES[np.argmax(output_data)]

        # Prettify and display results
        confidence = round(confidence * 100, 2)
        prediction = prediction.title()

        plt.title(f'Prediction: {prediction}, Confidence: {confidence}%')
        plt.imshow(image)
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--images_dir', type=str)

    args = parser.parse_args()
    main(args.model_path, args.images_dir)
