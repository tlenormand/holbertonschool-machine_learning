#!/usr/bin/env python3

import tensorflow.keras as K
import matplotlib.pyplot as plt
from tensorflow.python.client import device_lib
import time


def preprocess_data(X, Y):
    """ pre-processes the data for the model

    Arguments:
        X: numpy.ndarray of shape (m, 32, 32, 3) containing the CIFAR 10 data,
           where m is the number of data points
        Y: numpy.ndarray of shape (m,) containing the CIFAR 10 labels for X

    Returns:
        X_p: numpy.ndarray containing the preprocessed X
        Y_p: numpy.ndarray containing the preprocessed Y
    """
    X_p = K.applications.resnet.preprocess_input(X)
    Y_p = K.utils.to_categorical(Y, 10)

    return X_p, Y_p


if __name__ == '__main__':
    # available devices (CPU, GPU)
    print(f'Available devices: {device_lib.list_local_devices()}')

    (x_train, y_train), (x_test, y_test) = K.datasets.cifar10.load_data()
    x_train, y_train = preprocess_data(x_train, y_train)
    x_test, y_test = preprocess_data(x_test, y_test)
    start = time.time()

    # ========================================================================
    # INPUT MODEL
    # ========================================================================

    input_model = K.Sequential()

    input_model.add(K.layers.Input(shape=(32, 32, 3)))

    input_model.add(K.layers.Lambda(
        lambda image: K.backend.resize_images(
            image,
            height_factor=224/32,
            width_factor=224/32,
            data_format='channels_last',
            interpolation='nearest',
        )
    ))

    # ========================================================================
    # BASE MODEL
    # ========================================================================

    # model build ResNet, DenseNet, Inception, Xception, VGG16, VGG19
    base_model = K.applications.ResNet101(
        include_top=False,
        weights='imagenet',
        input_tensor=None,
        input_shape=(224, 224, 3),
        pooling='avg',
        classes=10,
        classifier_activation='softmax',
    )

    # model non trainable until layer conv5_block1_1_relu
    for layer in base_model.layers:
        if layer.name == 'conv5_block1_1_relu':
            break
        layer.trainable = False

    # ========================================================================
    # OUTPUT MODEL
    # ========================================================================

    output_model = K.Sequential()

    # fully connected
    output_model.add(K.layers.Dense(512))
    output_model.add(K.layers.BatchNormalization())
    output_model.add(K.layers.Activation('relu'))
    output_model.add(K.layers.Dropout(0.4))

    output_model.add(K.layers.Dense(512))
    output_model.add(K.layers.Activation('relu'))
    output_model.add(K.layers.BatchNormalization())
    output_model.add(K.layers.Dropout(0.4))

    output_model.add(K.layers.Dense(10, activation='softmax'))

    # ========================================================================
    # COMPILE
    # ========================================================================

    model = K.Sequential()

    model.add(input_model)
    model.add(base_model)
    model.add(output_model)

    model.compile(
        optimizer=K.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )

    model.build(input_shape=(None, 32, 32, 3))

    # ========================================================================
    # CALLBACKS
    # ========================================================================

    callbacks = []

    # early stopping
    callbacks.append(K.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=3,
        verbose=1,
    ))

    # model checkpoint
    callbacks.append(K.callbacks.ModelCheckpoint(
        filepath='cifar10.h5',
        monitor='val_accuracy',
        save_best_only=True,
    ))

    # ========================================================================
    # TRAINING
    # ========================================================================

    model.fit(
        x=x_train,
        y=y_train,
        batch_size=32,
        epochs=50,
        verbose=1,
        shuffle=True,
        validation_data=(x_test, y_test),
        callbacks=callbacks,
    )

    print('====================')
    print('Training completed !')
    print('   Total time: {}'.format(time.time() - start))
    print('   val_accuracy:', model.history.history['val_accuracy'][-1])
    print('   val_loss:', model.history.history['val_loss'][-1])
    print('   accuracy:', model.history.history['accuracy'][-1])
    print('   loss:', model.history.history['loss'][-1])
    print('====================')

    # ========================================================================
    # PLOT
    # ========================================================================

    plt.plot(
        model.history.history['accuracy'],
        label='accuracy',
        color='red'
    )
    plt.plot(
        model.history.history['val_accuracy'],
        label='val_accuracy',
        color='red',
        marker='o'
    )
    plt.plot(
        model.history.history['loss'],
        label='loss',
        color='blue'
    )
    plt.plot(
        model.history.history['val_loss'],
        label='val_loss',
        color='blue',
        marker='o'
    )
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.show()

    # ========================================================================
    # SAVE CSV
    # ========================================================================

    with open('history.csv', 'w') as f:
        f.write('epoch,accuracy,loss,val_accuracy,val_loss\n')
        for i in range(len(model.history.history['accuracy'])):
            f.write('{},{},{},{},{}\n'.format(
                i,
                model.history.history['accuracy'][i],
                model.history.history['loss'][i],
                model.history.history['val_accuracy'][i],
                model.history.history['val_loss'][i],
            ))

    # ========================================================================
    # SAVE MODEL
    # ========================================================================

    old_model = K.models.load_model('cifar10.h5')
    old_val_accuracy = old_model.history.history['val_accuracy'][-1]
    new_val_accuracy = model.history.history['val_accuracy'][-1]

    if new_val_accuracy > old_val_accuracy:
        model.save('cifar10.h5')
        save_str = 'model saved !'
    else:
        save_str = 'model not saved !'

    print(
        f'{save_str}\n\
        Old val_accuracy: {old_model.history.history["val_accuracy"][-1]}\n\
        New val_accuracy: {model.history.history["val_accuracy"][-1]}\n\
        Improvement: {old_val_accuracy - new_val_accuracy}'
    )
