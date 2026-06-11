"""
Módulo de inicializadores para redes neurais.

Implementa quatro inicializadores distintos. Cada inicializador herda
da classe abstrata Initializer e deve implementar initialize.

Inicializadores disponíveis:
- Xavier
- He (Kaiming)
- Random Normal
- Zero
"""

from abc import ABC, abstractmethod
import numpy as np


class Initializer(ABC):
    """
    Classe base para inicializadores de pesos.

    Um inicializador é responsável por gerar os valores iniciais dos
    pesos e vieses da rede neural.

    Todos os inicializadores devem implementar o método initialize(),
    que recebe a forma desejada da matriz e retorna um ndarray contendo
    os valores inicializados.
    """

    @abstractmethod
    def initialize(self, shape: tuple[int, int]) -> np.ndarray:
        """
        Inicializa uma matriz de pesos.

        Parâmetros:
            shape: Tupla (fan_in, fan_out) representando as dimensões
                da matriz de pesos.

        Retorna:
            Um ndarray contendo os valores inicializados.
        """


class XavierInitializer(Initializer):
    """
    Inicializador de Xavier ou Glorot Initialization.
    Normalmente usado com funções de ativação sigmoid e tanh.

    Distribuição: Uniforme(-a, a)
    Onde:
        a = sqrt(6 / (fan_in + fan_out))

    Objetivo:
        Manter aproximadamente constante a variância das ativações
        entre as camadas, reduzindo os problemas de desaparecimento
        e explosão dos gradientes.
    """

    def initialize(self, shape: tuple[int, int]) -> np.ndarray:
        fan_in, fan_out = shape

        limit = np.sqrt(6 / (fan_in + fan_out))

        return np.random.uniform(-limit, limit, shape)


class HeInitializer(Initializer):
    """
    Inicializador de He ou Kaiming Initialization.
    Normalmente usado com funções de ativação ReLU e suas variantes.

    Distribuição: Normal(0, sqrt(2 / fan_in))

    Objetivo:
        Preservar a variância das ativações em redes profundas,
        melhorando a propagação dos gradientes em funções do tipo ReLU.
    """

    def initialize(self, shape: tuple[int, int]) -> np.ndarray:
        fan_in, _ = shape

        return np.random.normal(0, np.sqrt(2 / fan_in), shape)


class RandomNormalInitializer(Initializer):
    """
    Inicializador aleatório com distribuição normal.

    Distribuição: Normal(mean, std)

    Objetivo:
        Gerar pesos aleatórios a partir de uma distribuição gaussiana
        parametrizada pelo usuário.
    """

    def __init__(self, mean: float = 0, std: float = 0.01):
        self.__mean = mean
        self.__std = std

    def initialize(self, shape: tuple[int, int]) -> np.ndarray:
        return np.random.normal(self.__mean, self.__std, shape)


class ZeroInitializer(Initializer):
    """
    Inicializador nulo.

    Distribuição: W = 0

    Objetivo:
        Inicializar todos os elementos da matriz com zero.

    Observação:
        É adequado para vieses, mas não é recomendado para pesos de
        redes neurais, pois todos os neurônios de uma mesma camada
        aprenderão exatamente os mesmos parâmetros.
    """

    def initialize(self, shape: tuple[int, int]) -> np.ndarray:
        return np.zeros(shape)
