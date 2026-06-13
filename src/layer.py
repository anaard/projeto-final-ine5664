"""Módulo de camada densa para redes neurais feedforward."""

from dataclasses import dataclass
import numpy as np
from .activations import Activation, get_activation
from .initializers import (
    Initializer,
    get_initializer,
    RandomNormalInitializer,
    ZeroInitializer,
)


@dataclass
class LayerGradients:
    """Gradientes calculados durante o backpropagation.

    Atributos:
        grad_x: Gradiente em relação à entrada, formato ``(batch, input_size)``.
        grad_w: Gradiente em relação aos pesos, formato ``(input_size, output_size)``.
        grad_b: Gradiente em relação aos vieses, formato ``(1, output_size)``.
    """

    grad_x: np.ndarray
    grad_w: np.ndarray
    grad_b: np.ndarray


class Layer:
    """Camada totalmente conectada: ``activation(x @ W + b)``."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        activation: Activation | str,
        weight_initializer: Initializer | str = RandomNormalInitializer(),
        bias_initializer: Initializer | str = ZeroInitializer(),
    ) -> None:
        """Inicializa a camada e seus parâmetros treináveis.

        Parâmetros:
            input_size: Número de features de entrada.
            output_size: Número de neurônios de saída.
            activation: Função de ativação (instância ou nome em string).
            weight_initializer: Inicializador dos pesos. Padrão: normal aleatório.
            bias_initializer: Inicializador dos vieses. Padrão: zeros.
        """

        self.__input_size = input_size
        self.__output_size = output_size
        self.__activation = get_activation(activation)

        weight_init = get_initializer(weight_initializer)
        bias_init = get_initializer(bias_initializer)
        self.__weights = weight_init.initialize((input_size, output_size))
        self.__biases = bias_init.initialize((1, output_size))

        self.__z = self.__x = None

    @property
    def input_size(self) -> int:
        """Número de features de entrada."""
        return self.__input_size

    @property
    def output_size(self) -> int:
        """Número de neurônios de saída."""
        return self.__output_size

    @property
    def activation(self) -> Activation:
        """Função de ativação da camada."""
        return self.__activation

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Executa o passo forward e armazena valores para o backward.

        Parâmetros:
            x: Entrada de formato ``(batch_size, input_size)``.

        Retorna:
            Saída ativada de formato ``(batch_size, output_size)``.
        """
        self.__x = x
        self.__z = x @ self.__weights + self.__biases
        return self.__activation.forward(self.__z)

    def backward(self, grad_y: np.ndarray) -> LayerGradients:
        """Executa o passo backward e retorna os gradientes.

        Parâmetros:
            grad_y: Gradiente da perda em relação à saída de formato ``(batch_size, output_size)``.

        Retorna:
            :class:`LayerGradients` com os gradientes em relação a x, W e b.
        """
        m = self.__x.shape[0]

        grad_z = grad_y * self.__activation.backward(self.__z)
        grad_w = self.__x.T @ grad_z / m
        grad_b = np.sum(grad_z, axis=0, keepdims=True) / m
        grad_x = grad_z @ self.__weights.T

        return LayerGradients(grad_x, grad_w, grad_b)

    def update(self, grad_w: np.ndarray, grad_b: np.ndarray) -> None:
        """Atualiza pesos e vieses.

        Args:
            grad_w: Gradiente dos pesos já escalonado pelo learning rate.
            grad_b: Gradiente dos vieses já escalonado pelo learning rate.
        """
        self.__weights -= grad_w
        self.__biases -= grad_b
