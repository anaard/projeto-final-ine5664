"""Wrapper sklearn para regressão"""

from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np

from src.neural_network import NeuralNetwork
from src.layer import Layer


class NeuralNetworkRegressor(BaseEstimator, RegressorMixin):
    """
    Adaptador da NeuralNetwork para regressão compatível com
    a API do scikit-learn.
    """

    def __init__(
        self,
        hidden_layers=(16,),
        hidden_activation="relu",
        output_activation="linear",
        learning_rate=0.01,
        epochs=500,
        batch_size=32,
        seed=42,
    ):
        """
        Inicializa o regressor.

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

    def fit(self, X: np.ndarray, y):
        """
        Treina a rede neural.

        Parâmetros:
            X: Matriz de atributos.
            y: Vetor de valores alvo.

        Retorna:
            NeuralNetworkRegressor: Instância treinada.
        """

        X = np.asarray(X)
        y = np.asarray(y).reshape(-1, 1)

        self.model_ = NeuralNetwork(
            cost_function="mse", learning_rate=self.learning_rate
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

        return self

    def predict(self, X):
        """
        Prediz valores contínuos.

        Parâmetros:
            X: Matriz de atributos.

        Retorna:
            np.ndarray: Valores preditos.
        """
        return self.model_.predict(X).ravel()
