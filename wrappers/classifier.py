"""Wrapper sklearn para classificação binária"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

from src.neural_network import NeuralNetwork
from src.layer import Layer


class NeuralNetworkBinaryClassifier(BaseEstimator, ClassifierMixin):
    """
    Adaptador da NeuralNetwork para classificação binária compatível com
    a API do scikit-learn.
    """

    def __init__(
        self,
        hidden_layers=(16,),
        hidden_activation="relu",
        output_activation="sigmoid",
        learning_rate=0.01,
        epochs=500,
        batch_size=32,
        seed=42,
    ):
        """
        Inicializa o classificador.

        Parâmetros:
            hidden_layers: Quantidade de neurônios em cada camada oculta.
            hidden_activation: Função de ativação das camadas ocultas.
            output_activation: Função de ativação da camada de saída.
            learning_rate: Taxa de aprendizado.
            epochs: Número máximo de épocas.
            batch_size: Tamanho do mini-batch.
            seed: Seed para reprodutibilidade.
        """
        self.hidden_layers = hidden_layers
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.seed = seed

        self.model_ = None
        self.classes_ = None

    def fit(self, X, y):
        """
        Treina a rede neural.

        Parâmetros:
            X: Matriz de atributos.
            y: Vetor de rótulos binários.

        Retorna:
            NeuralNetworkClassifier: Instância treinada.
        """
        X = np.asarray(X)
        y = np.asarray(y).reshape(-1, 1)

        self.model_ = NeuralNetwork(
            cost_function="binary_crossentropy", learning_rate=self.learning_rate
        )

        prev = X.shape[1]

        for neurons in self.hidden_layers:
            self.model_.add_layer(
                Layer(
                    input_size=prev,
                    output_size=neurons,
                    activation=self.hidden_activation,
                )
            )
            prev = neurons

        self.model_.add_layer(
            Layer(input_size=prev, output_size=1, activation=self.output_activation)
        )

        self.model_.fit(
            X,
            y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            seed=self.seed,
            show_summary=False,
        )

        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        """
        Estima as probabilidades das classes.

        Parâmetros:
            X: Matriz de atributos.

        Retorna:
            np.ndarray: Probabilidades das classes negativa e positiva.
        """
        p = self.model_.predict(X).ravel()
        return np.column_stack([1 - p, p])

    def predict(self, X):
        """
        Prediz as classes das amostras.

        Parâmetros:
            X: Matriz de atributos.

        Retorna:
            np.ndarray: Classes preditas.
        """
        p = self.model_.predict(X).ravel()
        return (p >= 0.5).astype(int)
