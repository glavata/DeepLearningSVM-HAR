import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv1D, Conv2D, MaxPooling2D, MaxPooling1D, Flatten, Input, Concatenate
from keras.utils import to_categorical
from keras import regularizers
from keras.models import Model
from keras import backend as K

class CNN():

    def __init__(self, params, num_classes, shape_first, shape_sec):
        self.__data_loaded = False
        self.__model_trained = False
        self.__params = params
        self.__num_classes = num_classes
        self.__shape_fir = shape_first
        self.__shape_sec = shape_sec
        self.__define_model()

    def __define_model(self):
        params = self.__params
        shape_fir = (self.__shape_fir[1], self.__shape_fir[2], self.__shape_fir[3])
        shape_sec = (self.__shape_sec[1])

        input_top = Input(shape_fir)
        model_top = Conv2D(128, kernel_size=(3, 3), activation='relu')(input_top)
        model_top = MaxPooling2D(pool_size=(2, 2))(model_top)
        model_top = Conv2D(64, kernel_size=(3, 3), activation='relu')(model_top)
        model_top = MaxPooling2D(pool_size=(2, 2))(model_top)
        model_top = Conv2D(64, kernel_size=(3, 3), activation='relu')(model_top)
        model_top = MaxPooling2D(pool_size=(2, 2))(model_top)
        model_top = Flatten()(model_top)

        input_bottom = Input(shape_sec)
        model_bottom = Dense(1024, activation='sigmoid')(input_bottom)
        model_bottom = Dense(128, activation='sigmoid')(model_bottom)

        merged = Concatenate()([model_top, model_bottom])
        output = Dense(self.__num_classes, activation='softmax')(merged)

        model_final = Model(inputs=[input_top, input_bottom], outputs=[output])
        model_final.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

        self.__model = model_final

    def reinit(self):
        K.clear_session()
        self.__define_model()
        
    def train_model(self, X, y):
        y_categ = to_categorical(y, self.__num_classes)
        res = self.__model.train_on_batch(X, y_categ)
        print(res)
        return res

    def predict(self, X):
        return np.argmax(self.__model.predict(X), axis=-1)
        
    def get_out_data(self, new_data):
        old_model = self.__model

        layer_count = len(old_model.layers)
        new_model = Model(inputs=old_model.input, outputs=old_model.layers[layer_count-2].output)
        new_features = new_model.predict(new_data)
        return new_features

