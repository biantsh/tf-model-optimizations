import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_model_optimization as tfmot
from tensorflow import keras
from keras import layers
from tensorflow_model_optimization.python.core.clustering.keras.experimental import \
    cluster

CLASS_NAMES = ['cat', 'dog']
TRAIN_SPLIT = 0.8
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224

preprocessing = keras.Sequential([
    layers.Resizing(IMAGE_HEIGHT, IMAGE_WIDTH),
    layers.Rescaling(1. / 127.5, offset=-1)
])


def one_hot(label):
    return [1 - label, label]


data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
    layers.RandomContrast(0.2),
    layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
])

model = keras.Sequential([
    layers.Conv2D(15, (3, 3), activation='relu',
                  input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 3)),
    layers.MaxPooling2D(pool_size=(4, 4)),
    layers.Conv2D(15, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(4, 4)),
    layers.Dropout(0.2),
    layers.Conv2D(2, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(4, 4)),

    layers.GlobalMaxPool2D(),
    layers.Activation('softmax')
])

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

train, val = tfds.load('cats_vs_dogs',
                       split=[f'train[:{TRAIN_SPLIT * 100:.0f}%]',
                              f'train[{TRAIN_SPLIT * 100:.0f}%:]'],
                       as_supervised=True)

train = train.map(lambda x, y: (preprocessing(x), one_hot(y)))
val = val.map(lambda x, y: (preprocessing(x), one_hot(y)))

train = train.batch(32).map(lambda x, y: (data_augmentation(x), y))
val = val.batch(32)

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train, epochs=50, validation_data=val)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()
model_file = 'model_2.tflite'
# Save the model.
with open(model_file, 'wb') as f:
    f.write(tflite_model)

## Pruning
prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

pruning_params = {
    'pruning_schedule': tfmot.sparsity.keras.ConstantSparsity(0.5,
                                                              begin_step=0,
                                                              frequency=100)
}

callbacks = [
    tfmot.sparsity.keras.UpdatePruningStep()
]

pruned_model = prune_low_magnitude(model, **pruning_params)

# Use smaller learning rate for fine-tuning
opt = keras.optimizers.Adam(learning_rate=1e-5)

pruned_model.compile(
    loss='categorical_crossentropy',
    optimizer=opt,
    metrics=['accuracy'])

# Fine-tune model
pruned_model.fit(
    train,
    epochs=3,
    validation_data=val,
    callbacks=callbacks)

stripped_pruned_model = tfmot.sparsity.keras.strip_pruning(pruned_model)
##

## Clustering
cluster_weights = tfmot.clustering.keras.cluster_weights
CentroidInitialization = tfmot.clustering.keras.CentroidInitialization

cluster_weights = cluster.cluster_weights

clustering_params = {
    'number_of_clusters': 8,
    'cluster_centroids_init': CentroidInitialization.KMEANS_PLUS_PLUS,
    'preserve_sparsity': True
}

sparsity_clustered_model = cluster_weights(stripped_pruned_model,
                                           **clustering_params)

sparsity_clustered_model.compile(optimizer='adam',
                                 loss='categorical_crossentropy',
                                 metrics=['accuracy'])

print('Train sparsity preserving clustering model:')
sparsity_clustered_model.fit(train, epochs=3, validation_data=val)

stripped_clustered_model = tfmot.clustering.keras.strip_clustering(
    sparsity_clustered_model)
##

## PCQAT
quant_aware_annotate_model = tfmot.quantization.keras.quantize_annotate_model(
    stripped_clustered_model)
pcqat_model = tfmot.quantization.keras.quantize_apply(
    quant_aware_annotate_model,
    tfmot.experimental.combine.Default8BitClusterPreserveQuantizeScheme(
        preserve_sparsity=True))

pcqat_model.compile(optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
print('Train pcqat model:')
pcqat_model.fit(train, batch_size=32, epochs=1, validation_data=val)
##


## TFLITE
converter = tf.lite.TFLiteConverter.from_keras_model(pcqat_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
pcqat_tflite_model = converter.convert()
pcqat_model_file = 'pcqat_model_2.tflite'
# Save the model.
with open(pcqat_model_file, 'wb') as f:
    f.write(pcqat_tflite_model)
##
