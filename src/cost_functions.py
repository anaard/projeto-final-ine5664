"""
Módulo de funções de custo para redes neurais.

Cada função de custo implementa compute (valor escalar da perda) e
gradient (gradiente em relação às predições), usado no backpropagation.

Funções disponíveis:
- MeanSquaredError (MSE)
- BinaryCrossEntropy (BCE)
- CategoricalCrossEntropy (CCE)
"""

from abc import ABC, abstractmethod
import numpy as np


class Loss(ABC):
    """Classe base abstrata para funções de custo."""

    @abstractmethod
    def compute(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """
        Calcula o valor escalar da perda.

        Parâmetros:
            predictions (np.ndarray): Saídas da rede neural.
            targets (np.ndarray): Valores alvo reais.

        Retorna
            float: Valor da perda para o batch atual.
        """

    @abstractmethod
    def gradient(self, predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """
        Calcula o gradiente da perda em relação às predições.

        Parâmetros
            predictions (np.ndarray): Saídas da rede neural.
            targets (np.ndarray): Valores alvo reais.

        Retorna
            np.ndarray: Gradiente.
        """


class MeanSquaredError(Loss):
    """
    Erro Quadrático Médio, ou Mean Squared Error (MSE).
    Normalmente usada para tarefas de regressão.

    Função: 1/m * sum[i=1..m] (y_i - ŷ_i)^2
    Derivada: 2 * (ŷ_i - y_i)
    """

    def compute(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Calcula o MSE médio sobre o batch."""
        return float(np.mean((predictions - targets) ** 2))

    def gradient(self, predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """Retorna o gradiente do MSE em relação às predições."""
        return 2.0 * (predictions - targets)


# Constante para evitar log(0) e divisão por zero
_EPSILON: float = 1e-15


class BinaryCrossEntropy(Loss):
    """
    Entropia Cruzada Binária, ou Binary Cross-Entropy (BCE).
    Normalmente usada para classificação binária.

    Função: - 1/m * sum[i=1..m] [y_i * ln(ŷ_i) + (1 - y_i) * ln(1 - ŷ_i)]

    Derivada: (ŷ - y) / ŷ * (1 - ŷ)
    """

    def compute(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """Calcula a BCE média sobre o batch."""
        prob = np.clip(predictions, _EPSILON, 1.0 - _EPSILON)
        return float(
            -np.mean(targets * np.log(prob) + (1.0 - targets) * np.log(1.0 - prob))
        )

    def gradient(self, predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """Retorna o gradiente da BCE em relação às predições."""
        prob = np.clip(predictions, _EPSILON, 1.0 - _EPSILON)
        return (prob - targets) / (prob * (1.0 - prob))


class CategoricalCrossEntropy(Loss):
    """
    Entropia Cruzada Categórica, ou Categorical Cross-Entropy (CCE).
    Normalmente usada para classificação multiclasse com Softmax na camada de saída.

    Função: - 1/m * sum[i=1..m] sum[c=1..C] y_ic * ln(ŷ_ic)

    Derivada combinada (CCE + Softmax):
        dL/dz = (ŷ - y)

    Observação:
        O gradiente implementado não é o da CCE isolada (-(y/ŷ) / m), mas sim
        o gradiente combinado CCE + Softmax, obtido ao se derivar a perda
        diretamente em relação à entrada pré-ativação da Softmax. Isso é
        matematicamente equivalente quando a camada de saída usa Softmax com
        alvos em one-hot encoding, e é numericamente mais estável por evitar
        a divisão por ŷ.

        Por isso, esta classe deve ser usada exclusivamente com Softmax como
        função de ativação da última camada. Para outras ativações, use
        MeanSquaredError ou BinaryCrossEntropy.
    """

    def compute(self, predictions: np.ndarray, targets: np.ndarray) -> float:
        """
        Calcula a CCE média sobre o batch.

        Parâmetros:
            predictions (np.ndarray): Probabilidades preditas, forma (n, n_classes).
            targets (np.ndarray): Rótulos em one-hot encoding, forma (n, n_classes).

        Retorna:
            float: Valor da CCE.
        """
        prob = np.clip(predictions, _EPSILON, 1.0)
        return float(-np.mean(np.sum(targets * np.log(prob), axis=1)))

    def gradient(self, predictions: np.ndarray, targets: np.ndarray) -> np.ndarray:
        """Retorna o gradiente da CCE em relação às predições."""
        return predictions - targets
