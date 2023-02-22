from abc import ABC, abstractmethod


class Model(ABC):
    def __init__(self):
        self._trained = False

    @abstractmethod
    def train(self, data, labels):
        self._trained = True

    @abstractmethod
    def predict(self, data):
        if not self._trained:
            raise ValueError("Model must be trained before predictions")
        # make predictions

        pass
