"""Wrapper sklearn para classificação multiclasse"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder

from src.neural_network import NeuralNetwork
from src.layer import Layer


class NeuralNetworkMulticlassClassifier(BaseEstimator, ClassifierMixin):
    """
    Adaptador da NeuralNetwork para classificação multiclasse compatível
    com a API do scikit-learn.
    """

    def __init__(
        self,
        hidden_layers=(16, 8),
        hidden_activation="relu",
        output_activation="softmax",
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
        self.encoder_ = None
        self.classes_ = None

    def fit(self, X, y):
        """
        Treina a rede neural.

        Parâmetros:
            X: Matriz de atributos.
            y: Vetor de rótulos.

        Retorna:
            NeuralNetworkMulticlass: Instância treinada.
        """
        X = np.asarray(X)
        self.encoder_ = LabelEncoder()
        y_int = self.encoder_.fit_transform(y)

        self.classes_ = self.encoder_.classes_
        n_classes = len(self.classes_)

        y_onehot = np.eye(n_classes)[y_int]

        self.model_ = NeuralNetwork(
            cost_function="categorical_crossentropy", learning_rate=self.learning_rate
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
            Layer(
                input_size=prev,
                output_size=n_classes,
                activation=self.output_activation,
            )
        )

        self.model_.fit(
            X,
            y_onehot,
            epochs=self.epochs,
            batch_size=self.batch_size,
            seed=self.seed,
            show_summary=False,
        )

        return self

    def predict_proba(self, X):
        """
        Estima as probabilidades de cada classe.

        Parâmetros:
            X: Matriz de atributos.

        Retorna:
            np.ndarray: Probabilidades preditas para cada classe.
        """
        return self.model_.predict(X)

    def predict(self, X):
        """
        Prediz as classes das amostras.

        Parâmetros:
            X: Matriz de atributos.

        Retorna:
            np.ndarray: Classes preditas.
        """
        probs = self.predict_proba(X)

        idx = np.argmax(probs, axis=1)
        return self.encoder_.inverse_transform(idx)
