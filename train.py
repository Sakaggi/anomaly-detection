import numpy as np
import tensorflow as tf
import sklearn.metrics
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import matplotlib.pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def train_01_LEGO_model():
    train_datagen = ImageDataGenerator(rescale=1./ 255)

    train_generator = train_datagen.flow_from_directory('/train_data',
                                                        target_size=(160, 160),
                                                        class_mode='binary',
                                                        batch_size=50)

    valid_datagen = ImageDataGenerator(rescale=1./ 255)

    valid_generator = valid_datagen.flow_from_directory('/eval_data',
                                                        target_size=(160, 160), class_mode='binary', batch_size=50)

    # test_steps_per_epoch = np.math.ceil(valid_generator.samples / valid_generator.batch_size)

    DESIRED_ACCURACY = 1.00

    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if (logs.get('val_acc') > DESIRED_ACCURACY):
                if(logs.get('acc') > DESIRED_ACCURACY):
                    print("\nReached 100% accuracy so cancelling training!")
                    self.model.stop_training = True

    callbacks = myCallback()

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(160, 160, 3)),
        tf.keras.layers.MaxPooling2D(2, 2),
        # The second convolution
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        # The third convolution
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        # The fourth convolution
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        # The fifth convolution
        tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        # Flatten the results to feed into a DNN
        tf.keras.layers.Flatten(),
        # 128 neuron hidden layer
        tf.keras.layers.Dense(128, activation='relu'),
        # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('NG') and 1 for the other ('OK')
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy',
                  optimizer=RMSprop(lr=0.001),
                  metrics=['acc'])

    # predictions = model.predict_generator(valid_generator, steps=test_steps_per_epoch)
    # # Get most likely class
    # predicted_classes = np.argmax(predictions, axis=1)
    # true_classes = valid_generator.classes
    # class_labels = list(valid_generator.class_indices.keys())

    history = model.fit_generator(
        train_generator,
        epochs=10,
        validation_data=valid_generator,
        verbose=1,
        callbacks=[callbacks]
    )
    model.save("01_LEGO.hdf5")


    # report = metrics.classification_report(true_classes, predicted_classes, target_names=class_labels)
    # print(report)

    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))  # Get number of epochs

    # ------------------------------------------------
    # Plot training and validation accuracy per epoch
    # ------------------------------------------------
    plt.plot(epochs, acc, 'r', "Training Accuracy")
    plt.plot(epochs, val_acc, 'b', "Validation Accuracy")
    plt.title('Training and validation accuracy')
    plt.figure()

    # ------------------------------------------------
    # Plot training and validation loss per epoch
    # ------------------------------------------------
    plt.plot(epochs, loss, 'r', "Training Loss")
    plt.plot(epochs, val_loss, 'b', "Validation Loss")
    plt.figure()
    plt.show()

    return history.history['val_acc'][-1]


train_01_LEGO_model()
