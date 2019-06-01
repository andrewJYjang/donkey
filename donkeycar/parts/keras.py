""""

keras.py

functions to run and train autopilots using keras

"""

from keras import Input
from keras.models import Model, load_model
from keras.layers import Convolution2D, BatchNormalization, Dropout, Flatten, Dense
from keras.callbacks import ModelCheckpoint, EarlyStopping


class KerasPilot:

    def load(self, model_path, compile=True):
        self.model = load_model(model_path, compile=compile)

    def shutdown(self):
        pass

    def train(self, train_gen, val_gen,
              saved_model_path, epochs=100, steps=100, train_split=0.8,
              verbose=1, min_delta=.0005, patience=5, use_early_stop=True):
        """
        train_gen: generator that yields an array of images an array of

        """

        # checkpoint to save model after each epoch
        save_best = ModelCheckpoint(saved_model_path,
                                    monitor='val_loss',
                                    verbose=verbose,
                                    save_best_only=True,
                                    mode='min')

        # stop training if the validation error stops improving.
        early_stop = EarlyStopping(monitor='val_loss',
                                   min_delta=min_delta,
                                   patience=patience,
                                   verbose=verbose,
                                   mode='auto')

        callbacks_list = [save_best]

        if use_early_stop:
            callbacks_list.append(early_stop)

        hist = self.model.fit_generator(
            train_gen,
            steps_per_epoch=steps,
            epochs=epochs,
            verbose=1,
            validation_data=val_gen,
            callbacks=callbacks_list,
            validation_steps=int(steps * (1.0 - train_split) / train_split))
        return hist


class KerasLinear(KerasPilot):
    def __init__(self, shape, model=None, num_outputs=None, *args, **kwargs):
        super(KerasLinear, self).__init__(*args, **kwargs)
        if model:
            self.model = model
        self.model = default_linear(shape)

    def run(self, img_arr):
        img_arr = img_arr.reshape((1,) + img_arr.shape)
        outputs = self.model.predict(img_arr)
        # print(len(outputs), outputs)
        steering = outputs[0]
        throttle = outputs[1]
        return steering[0][0], throttle[0][0]


class KerasTransfer(KerasPilot):
    from keras.applications.mobilenet_v2 import preprocess_input
    def __init__(self, shape, model=None, num_outputs=None, *args, **kwargs):
        super(KerasTransfer, self).__init__(*args, **kwargs)
        if model:
            self.model = model
        self.model = default_transfer(shape)

    def run(self, img_arr):
        img_arr = preprocess_input(img_arr)
        img_arr = img_arr.reshape((1,) + img_arr.shape)
        outputs = self.model.predict(img_arr)
        steering = outputs[0]
        throttle = outputs[1]
        return steering[0][0], throttle[0][0]


def default_linear(shape):
    img_in = Input(shape=shape, name='img_in')
    x = img_in

    x = BatchNormalization()(x)
    # Convolution2D class name is an alias for Conv2D
    x = Convolution2D(filters=24, kernel_size=(5, 5), strides=(2, 2), activation='relu')(x)
    x = Convolution2D(filters=32, kernel_size=(5, 5), strides=(2, 2), activation='relu')(x)
    x = Convolution2D(filters=64, kernel_size=(5, 5), strides=(2, 2), activation='relu')(x)
    x = Convolution2D(filters=64, kernel_size=(3, 3), strides=(2, 2), activation='relu')(x)
    x = Convolution2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu')(x)

    x = Flatten(name='flattened')(x)
    x = Dense(units=100, activation='relu')(x)
    x = Dropout(rate=.1)(x)
    x = Dense(units=50, activation='relu')(x)
    x = Dropout(rate=.1)(x)

    # categorical output of the angle
    angle_out = Dense(units=1, activation='linear', name='angle_out')(x)
    # continous output of throttle
    throttle_out = Dense(units=1, activation='linear', name='throttle_out')(x)

    model = Model(inputs=[img_in], outputs=[angle_out, throttle_out])

    model.compile(optimizer='adam',
                  loss={'angle_out': 'mean_squared_error',
                        'throttle_out': 'mean_squared_error'},
                  loss_weights={'angle_out': 0.5, 'throttle_out': .5})

    return model

def default_transfer(shape):
    from keras.applications.mobilenet_v2 import MobileNetV2
    base = MobileNetV2(input_shape=shape, include_top=False, weights='imagenet')
    # Mark base layers as not trainable
    for layer in base.layers:
      layer.trainable = False

    x = Flatten(name='flattened')(base.output)
    x = Dense(units=100, activation='relu')(x)
    x = Dropout(rate=.1)(x)
    x = Dense(units=50, activation='relu')(x)
    x = Dropout(rate=.1)(x)

    # Outputs
    angle_out = Dense(units=1, activation='relu', name='angle_out')(x)
    throttle_out = Dense(units=1, activation='relu', name='throttle_out')(x)

    model = Model(inputs=base.input, outputs=[angle_out, throttle_out])
    model.compile(optimizer='adam',
                  loss={'angle_out': 'mean_squared_error',
                        'throttle_out': 'mean_squared_error'},
                  loss_weights={'angle_out': 0.5, 'throttle_out': .5})

    return model
