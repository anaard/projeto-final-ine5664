"""
Módulo de funções de ativação para redes neurais.

Implementa oito funções de ativação distintas e suas derivadas,
necessárias para o algoritmo de backpropagation. Cada função herda
da classe abstrata Activation e deve implementar forward e backward.

Funções disponíveis:
- ReLU
- Sigmoid
- Tanh
- LeakyReLU
- ELU
- SELU
- Softmax
- Linear
"""

from abc import ABC, abstractmethod
import numpy as np


class Activation(ABC):
    """Classe base abstrata para funções de ativação."""

    @abstractmethod
    def forward(self, pre_activation: np.ndarray) -> np.ndarray:
        """Calcula a saída da função de ativação."""

    @abstractmethod
    def backward(self, pre_activation: np.ndarray) -> np.ndarray:
        """Calcula a derivada da função de ativação em relação à entrada."""


class ReLU(Activation):
    """
    Rectified Linear Unit (ReLU).

    Função: max(0, z)

    Derivada: 1 se v >= 0; 0 caso contrário.
    """

    def forward(self, pre_activation: np.ndarray) -> np.ndarray:
        """Aplica ReLU elemento a elemento."""
        return np.maximum(0.0, pre_activation)

    def backward(self, pre_activation: np.ndarray) -> np.ndarray:
        """Retorna a derivada da ReLU."""
        return (pre_activation >= 0.0).astype(float)


class Sigmoid(Activation):
    """
    Função Sigmóide (logística).

    Função: 1 / (1 + e^-z)

    Derivada: f(z) * (1 - f(z))
    """

    def forward(self, pre_activation: np.ndarray) -> np.ndarray:
        """Aplica a função sigmóide elemento a elemento de forma estável."""
        result = np.zeros_like(pre_activation, dtype=float)

        pos_mask = pre_activation >= 0.0
        neg_mask = ~pos_mask

        result[pos_mask] = 1.0 / (1.0 + np.exp(-pre_activation[pos_mask]))
        exp_neg = np.exp(pre_activation[neg_mask])
        result[neg_mask] = exp_neg / (1.0 + exp_neg)

        return result

    def backward(self, pre_activation: np.ndarray) -> np.ndarray:
        """Retorna a derivada da sigmóide."""
        sigma = self.forward(pre_activation)
        return sigma * (1.0 - sigma)


class Tanh(Activation):
    """
    Tangente Hiperbólica.

    Função: tanh(z)

    Derivada: 1 - tanh^2(z)
    """

    def forward(self, pre_activation: np.ndarray) -> np.ndarray:
        """Aplica a tangente hiperbólica elemento a elemento."""
        return np.tanh(pre_activation)

    def backward(self, pre_activation: np.ndarray) -> np.ndarray:
        """Calcula a derivada da tangente hiperbólica."""
        return 1.0 - np.tanh(pre_activation) ** 2


class LeakyReLU(Activation):
    """
    Leaky Rectified Linear Unit (Leaky ReLU).

    Função: z se z > 0; alpha * z caso contrário.

    Derivada: 1 se z > 0; alpha caso contrário.

    Parâmetros:
        alpha (float): Inclinação para entradas negativas (padrão: 0.01).
    """

    def __init__(self, alpha: float = 0.01):
        """
        Inicializa a LeakyReLU.

        Parâmetros:
            alpha (float): Coeficiente de vazamento (deve ser positivo).

        Raises:
            ValueError: Se alpha não for positivo.
        """
        if alpha <= 0:
            raise ValueError(f"alpha deve ser positivo, recebido: {alpha}")
        self.alpha = alpha

    def forward(self, pre_activation: np.ndarray) -> np.ndarray:
        """Aplica a LeakyReLU elemento a elemento."""
        return np.where(
            pre_activation > 0.0, pre_activation, self.alpha * pre_activation
        )

    def backward(self, pre_activation: np.ndarray) -> np.ndarray:
        """Calcula a derivada da LeakyReLU."""
        return np.where(pre_activation > 0.0, 1.0, self.alpha)


class ELU(Activation):
    """
    Exponential Linear Unit (ELU).

    Função: z se z >= 0; alpha * (e^z - 1) caso contrário.

    Derivada: 1 se z >= 0; alpha * e^z caso contrário.

    Parâmetros:
        alpha (float): Controla a saturação para entradas negativas (padrão: 1.0).
    """

    def __init__(self, alpha: float = 1.0):
        """
        Inicializa a ELU.

        Parâmetros:
            alpha (float): Valor de saturação negativa (deve ser positivo).

        Raises:
            ValueError: Se alpha não for positivo.
        """
        if alpha <= 0:
            raise ValueError(f"alpha deve ser positivo, recebido: {alpha}")
        self.alpha = alpha

    def forward(self, pre_activation: np.ndarray) -> np.ndarray:
        """Aplica a ELU elemento a elemento de forma otimizada."""
        result = pre_activation.copy()

        neg_mask = pre_activation < 0.0
        result[neg_mask] = self.alpha * (np.exp(pre_activation[neg_mask]) - 1.0)

        return result

    def backward(self, pre_activation: np.ndarray) -> np.ndarray:
        """Calcula a derivada da ELU."""
        result = np.ones_like(pre_activation, dtype=float)

        neg_mask = pre_activation < 0.0
        result[neg_mask] = self.alpha * np.exp(pre_activation[neg_mask])

        return result


class SELU(Activation):
    """
    Scaled Exponential Linear Unit (SELU).

    Função: SCALE * (z se z > 0; ALPHA * (e^z - 1) caso contrário).

    Derivada: SCALE * (1 se z > 0; ALPHA * e^z caso contrário).

    Referência: KLAMBAUER, Günter et al. Self-normalizing neural networks.
    Advances in neural information processing systems, v. 30, 2017.
    """

    # Constantes retiradas do artigo original.
    SCALE: float = 1.0507
    ALPHA: float = 1.67326

    def forward(self, pre_activation: np.ndarray) -> np.ndarray:
        """Aplica a SELU elemento a elemento de forma otimizada."""
        result = pre_activation.copy()

        neg_mask = pre_activation < 0.0
        result[neg_mask] = self.ALPHA * (np.exp(pre_activation[neg_mask]) - 1.0)

        return self.SCALE * result

    def backward(self, pre_activation: np.ndarray) -> np.ndarray:
        """Calcula a derivada da SELU."""
        result = np.ones_like(pre_activation, dtype=float)

        neg_mask = pre_activation < 0.0
        result[neg_mask] = self.ALPHA * np.exp(pre_activation[neg_mask])

        return self.SCALE * result


class Softmax(Activation):
    """
    Função Softmax.

    Função: e^z_i / sum_j e^z_j

    Observações:
        * Para evitar overflow em exp(z), a implementação subtrai o maior valor
        de cada amostra antes de calcular a exponencial, sem alterar o resultado
        matemático da Softmax.
        * backward() retorna 1 (identidade). Esta classe assume uso
        conjunto com CategoricalCrossEntropy, que já incorpora o gradiente
        combinado (ŷ - y)/n, evitando o cálculo do Jacobiano completo.
    """

    def forward(self, pre_activation: np.ndarray) -> np.ndarray:
        """Aplica a Softmax."""
        shifted = pre_activation - np.max(pre_activation, axis=1, keepdims=True)
        exp_vals = np.exp(shifted)

        return exp_vals / np.sum(exp_vals, axis=1, keepdims=True)

    def backward(self, pre_activation: np.ndarray) -> np.ndarray:
        """O gradiente real é calculado diretamente em CategoricalCrossEntropy."""
        return np.ones_like(pre_activation)


class Linear(Activation):
    """
    Função Linear (Identidade).

    Função: z

    Derivada: 1
    """

    def forward(self, pre_activation: np.ndarray) -> np.ndarray:
        """Retorna a própria entrada."""
        return pre_activation

    def backward(self, pre_activation: np.ndarray) -> np.ndarray:
        """Retorna a derivada da função linear (1)."""
        return np.ones_like(pre_activation)


_ACTIVATIONS: dict = {
    "relu": ReLU,
    "sigmoid": Sigmoid,
    "tanh": Tanh,
    "leaky_relu": LeakyReLU,
    "elu": ELU,
    "selu": SELU,
    "softmax": Softmax,
    "linear": Linear,
}


def get_activation(name, **kwargs) -> Activation:
    """
    Instancia uma função de ativação a partir de seu nome ou instância.

    Parâmetros:
        name (str ou Activation): Nome da ativação ('relu', 'sigmoid', 'tanh',
        'leaky_relu', 'elu', 'selu', 'softmax', 'linear') ou uma instância de Activation já criada.
        **kwargs: Parâmetros extras passados ao construtor (ex.: alpha=0.1 para LeakyReLU ou ELU).

    Retorna:
        Activation: Instância da função de ativação.

    Raises:
        TypeError: Se name não for str nem Activation.
        ValueError: Se o nome não corresponder a nenhuma ativação registrada.
    """
    if isinstance(name, Activation):
        return name
    if not isinstance(name, str):
        raise TypeError(
            f"'name' deve ser str ou Activation, recebido: {type(name).__name__}."
        )

    key = name.lower().strip()
    if key not in _ACTIVATIONS:
        valid = list(_ACTIVATIONS.keys())
        raise ValueError(f"Ativação '{name}' não encontrada. Opções válidas: {valid}")
    return _ACTIVATIONS[key](**kwargs)
