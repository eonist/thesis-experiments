from keras import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier


class NeuralNetwork(KerasClassifier):
    def __call__(self, input_dim=4, n_layers=3, start_units=10, loss="mean_squared_error", optimizer="adam"):
        model = Sequential()

        decay = start_units ** (-1 / (n_layers - 1))

        for i in range(n_layers):
            units = int(round(start_units * (decay ** i)))

            if i == 0:
                model.add(Dense(units, input_dim=input_dim, activation='relu'))
            elif i == n_layers - 1:
                model.add(Dense(units, activation='sigmoid'))
            else:
                model.add(Dense(units, activation='relu'))

        model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])

        return model
