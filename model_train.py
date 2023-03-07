"""Train a Keras model with PCQAT on the Asirra dataset and convert to TFLite.

Example usage:
    python3 model_train.py  \
      --output_file model.zip
      --image_width 224
      --image_height 224
      --train_split 0.8
      --batch_size 32
      --num_epochs 50
"""

import argparse
import logging
import zipfile

import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_model_optimization as tfmot
from keras import layers
from tensorflow import keras
from tensorflow_model_optimization.python.core. \
    clustering.keras.experimental import cluster

CLASS_NAMES = ['cat', 'dog']
PRUNING_PARAMS = {
    'pruning_schedule':
        tfmot.sparsity.keras.
        ConstantSparsity(
            target_sparsity=0.5,
            begin_step=0,
            frequency=100)
}
CLUSTERING_PARAMS = {
    'cluster_centroids_init':
        tfmot.clustering.keras.
        CentroidInitialization.KMEANS_PLUS_PLUS,
    'number_of_clusters': 8,
    'preserve_sparsity': True
}


def one_hot_encode(label: int) -> list[int, int]:
    return [1 - label, label]


def get_preprocessor(target_height, target_width) -> keras.Sequential:
    return keras.Sequential([
        layers.Resizing(target_height, target_width),
        layers.Rescaling(1. / 127.5, offset=-1)
    ])


def get_data_augmenter() -> keras.Sequential:
    return keras.Sequential([
        layers.RandomFlip('horizontal_and_vertical'),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
        layers.RandomContrast(0.2),
        layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
    ])


def get_model(image_height: int, image_width: int) -> keras.Sequential:
    input_shape = image_height, image_width, 3

    return keras.Sequential([
        layers.Conv2D(15, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D(pool_size=(4, 4)),

        layers.Conv2D(15, (3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(4, 4)),
        layers.Dropout(0.2),

        layers.Conv2D(2, (3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(4, 4)),

        layers.GlobalMaxPool2D(),
        layers.Activation('softmax')
    ])


def load_dataset(image_height: int,
                 image_width: int,
                 train_split: float,
                 batch_size: int
                 ) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    preprocess = get_preprocessor(image_height, image_width)
    augment = get_data_augmenter()

    train, val = tfds.load('cats_vs_dogs',
                           split=[f'train[:{train_split * 100:.0f}%]',
                                  f'train[{train_split * 100:.0f}%:]'],
                           as_supervised=True)

    train = train.map(lambda x, y: (preprocess(x), one_hot_encode(y)))
    val = val.map(lambda x, y: (preprocess(x), one_hot_encode(y)))

    train = train.batch(batch_size).map(lambda x, y: (augment(x), y))
    val = val.batch(batch_size)

    return train, val


def main(output_file: str,
         image_width: int,
         image_height: int,
         train_split: float,
         batch_size: int,
         num_epochs: int
         ) -> None:
    model = get_model(image_height, image_width)
    train, val = load_dataset(image_height,
                              image_width,
                              train_split,
                              batch_size)

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    logging.log(logging.INFO, f'Training model for {num_epochs} epochs...')
    model.fit(train, epochs=num_epochs, validation_data=val)

    # Pruning
    prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
    pruned_model = prune_low_magnitude(model, **PRUNING_PARAMS)

    pruned_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    logging.log(logging.INFO, f'Fine-tuning pruned model for 3 epochs...')
    callbacks = [tfmot.sparsity.keras.UpdatePruningStep()]
    pruned_model.fit(train, epochs=3, validation_data=val, callbacks=callbacks)

    pruned_model = tfmot.sparsity.keras.strip_pruning(pruned_model)

    # Clustering
    cluster_weights = cluster.cluster_weights
    clustered_model = cluster_weights(pruned_model, **CLUSTERING_PARAMS)

    clustered_model.compile(optimizer='adam',
                            loss='categorical_crossentropy',
                            metrics=['accuracy'])

    logging.log(logging.INFO, 'Fine-tuning clustered model for 3 epochs...')
    clustered_model.fit(train, epochs=3, validation_data=val)

    clustered_model = tfmot.clustering.keras.strip_clustering(clustered_model)

    # Pruning and clustering-preserving quantization aware training
    pcqat_model = tfmot.quantization.keras.quantize_annotate_model(
        clustered_model)
    quant_scheme = tfmot.experimental.combine. \
        Default8BitClusterPreserveQuantizeScheme()

    pcqat_model = tfmot.quantization.keras.quantize_apply(pcqat_model,
                                                          quant_scheme)

    pcqat_model.compile(optimizer='adam',
                        loss='categorical_crossentropy',
                        metrics=['accuracy'])

    logging.log(logging.INFO, 'Fine-tuning model with PCQAT for 1 epoch...')
    pcqat_model.fit(train, epochs=1, validation_data=val)

    # TFLite conversion
    converter = tf.lite.TFLiteConverter.from_keras_model(pcqat_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    pcqat_tflite_model = converter.convert()

    with zipfile.ZipFile(output_file, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        zip_file.writestr(output_file, pcqat_tflite_model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_file', type=str)
    parser.add_argument('--image_width', type=int, default=224)
    parser.add_argument('--image_height', type=int, default=224)
    parser.add_argument('--train_split', type=float, default=0.8)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=50)

    args = parser.parse_args()
    main(args.output_file,
         args.image_width,
         args.image_height,
         args.train_split,
         args.batch_size,
         args.num_epochs)
