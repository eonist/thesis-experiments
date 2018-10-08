from keras_tqdm import TQDMCallback
from sklearn.pipeline import Pipeline

from models.neural_network import NeuralNetwork


class CustomPipeline(Pipeline):
    def fit(self, X, y=None, **fit_params):
        Xt, fit_params = self._fit(X, y, **fit_params)
        if self._final_estimator is not None:

            if isinstance(self._final_estimator, NeuralNetwork):
                return self._final_estimator.fit(
                    Xt, y, **fit_params,
                    validation_split=0.25,
                    callbacks=[TQDMCallback(leave_inner=False, show_inner=False)] if False else [],
                    verbose=0
                )
            else:
                self._final_estimator.fit(Xt, y, **fit_params)
        return self
