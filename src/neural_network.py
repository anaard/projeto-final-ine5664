"""Módulo da rede neural feedforward com suporte a mini-batches."""

from dataclasses import dataclass, field
import numpy as np
from .cost_functions import Loss, get_loss, CategoricalCrossEntropy
from .activations import Softmax
from .layer import Layer


@dataclass
class History:
    """Histórico de treinamento da rede.

    Atributos:
        epochs: Lista com o número de cada época executada.
        losses: Lista com a perda média registrada em cada época.
    """

    epochs: list[int] = field(default_factory=list)
    losses: list[float] = field(default_factory=list)
    eval_losses: list[float] = field(default_factory=list)


class NeuralNetwork:
    """Rede neural totalmente conectada com treinamento por gradiente descendente."""

    def __init__(self, cost_function: Loss | str, learning_rate: float = 0.01) -> None:
        """Inicializa a rede neural.

        Parâmetros:
            cost_function: Função de custo (instância ou nome em string).
            learning_rate: Taxa de aprendizado. Padrão: 0.01.
        """
        self.__cost_function = get_loss(cost_function)
        self.__learning_rate = learning_rate
        self.__layers: list[Layer] = []

        self.__is_fitted = False

        self.__history = History()

    @property
    def history(self) -> History:
        """Histórico de épocas e perdas do treinamento."""
        return self.__history

    def add_layer(self, layer: Layer) -> None:
        """Adiciona uma camada à rede.

        Parâmetros:
            layer: Instância de :class:`~layer.Layer` a ser adicionada.

        Raises:
            TypeError: Se ``layer`` não for do tipo :class:`~layer.Layer`.
        """
        if isinstance(layer, Layer):
            self.__layers.append(layer)
            return

        raise TypeError(
            f"'layer' deve ser do tipo Layer, recebido: {type(layer).__name__}."
        )

    def fit(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        epochs: int = 100,
        batch_size: int | None = None,
        seed: int | None = None,
        validation_data: tuple[np.ndarray, np.ndarray] | None = None,
        validation_split: float = 0.0,
        patience: int | None = None,
        min_delta: float = 1e-4,
        show_summary: bool = True,
    ) -> None:
        """Treina a rede neural.

        Parâmetros:
            inputs: Dados de entrada, formato (n_amostras, n_features).
            targets: Rótulos esperados, formato (n_amostras, n_saídas).
            epochs: Número máximo de épocas. Padrão: 100.
            batch_size: Tamanho do mini-batch. None usa o dataset completo.
            seed: Seed para embaralhamento reprodutível.
            validation_data: Tupla (X_val, y_val) para avaliação a cada época.
            validation_split: Fração dos dados de treino usada como validação.
                Ignorado se validation_data for fornecido.
            patience: Épocas sem melhora antes de encerrar o treino (early
                stopping). None desativa. Requer validation_data ou
                validation_split.
            min_delta: Melhora mínima na perda monitorada para ser considerada
                uma evolução real. Padrão: 1e-4.
            show_summary: Se True, exibe resumo ao final.
        """
        self.__validate_output_activation()

        # Separar dados de validação
        if validation_data is not None:
            val_inputs, val_targets = validation_data
        elif 0.0 < validation_split < 1.0:
            split = int(len(inputs) * (1 - validation_split))
            inputs, val_inputs = inputs[:split], inputs[split:]
            targets, val_targets = targets[:split], targets[split:]
            validation_data = (val_inputs, val_targets)

        has_validation = validation_data is not None

        num_samples = inputs.shape[0]
        effective_batch = num_samples if batch_size is None else batch_size

        best_eval_loss = float("inf")
        patience_counter = 0

        rng = np.random.default_rng(seed)
        for epoch in range(epochs):
            epoch_loss = self.__epoch(inputs, targets, effective_batch, rng)

            self.__history.epochs.append(epoch + 1)
            self.__history.losses.append(epoch_loss)

            if has_validation:
                eval_loss = self.__evaluate(val_inputs, val_targets)
                self.__history.eval_losses.append(eval_loss)

            # Early stopping
            if patience is not None and has_validation:
                if eval_loss < best_eval_loss - min_delta:
                    best_eval_loss = eval_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping na época {epoch + 1}.")
                        break

        self.__is_fitted = True

        if show_summary:
            self.__summary()

    def __epoch(
        self,
        inputs: np.ndarray,
        targets: np.ndarray,
        batch_size: int,
        rng: np.random.Generator,
    ) -> float:
        """Executa uma época completa com mini-batches embaralhados.

        Parâmetros:
            inputs: Dataset de entrada completo.
            targets: Rótulos completos.
            batch_size: Tamanho do mini-batch.
            rng: Gerador de números aleatórios para o embaralhamento.

        Retorna:
            Perda (loss) média da época.
        """
        num_samples = inputs.shape[0]

        idx = rng.permutation(num_samples)
        data_s = inputs[idx]
        targets_s = targets[idx]

        losses = []
        for start in range(0, num_samples, batch_size):
            batch_inputs = data_s[start : start + batch_size]
            batch_targets = targets_s[start : start + batch_size]
            predictions = self.__feed_forward(batch_inputs)
            losses.append(self.__cost_function.compute(predictions, batch_targets))
            gradients = self.__back_propagation(predictions, batch_targets)
            self.__gradient_descent(gradients)

        return float(np.mean(losses))

    def __feed_forward(self, inputs: np.ndarray) -> np.ndarray:
        """Propaga a entrada por todas as camadas.

        Parâmetros:
            inputs: Batch de entrada.

        Retorna:
            Predições da rede.
        """
        activations = inputs
        for layer in self.__layers:
            activations = layer.forward(activations)
        return activations

    def __back_propagation(self, predictions: np.ndarray, targets: np.ndarray) -> list:
        """Calcula os gradientes de todas as camadas via backpropagation.

        Parâmetros:
            predictions: Saídas da rede para o batch atual.
            targets: Rótulos esperados para o batch atual.

        Retorna:
            Lista de tuplas ``(grad_w, grad_b)`` ordenada por camada.
        """
        gradients = []
        delta = self.__cost_function.gradient(predictions, targets)

        for layer in reversed(self.__layers):
            grads = layer.backward(delta)
            delta = grads.grad_x
            gradients.append((grads.grad_w, grads.grad_b))

        gradients.reverse()
        return gradients

    def __evaluate(self, inputs: np.ndarray, targets: np.ndarray) -> float:
        """Calcula a perda sobre um conjunto sem atualizar os pesos.

        Parâmetros:
            inputs: Dados de entrada.
            targets: Rótulos esperados.

        Retorna:
            Perda escalar sobre o conjunto completo.
        """
        predictions = self.__feed_forward(inputs)
        return float(self.__cost_function.compute(predictions, targets))

    def __gradient_descent(self, gradients: list[tuple]) -> None:
        """Atualiza os parâmetros de todas as camadas.

        Parâmetros:
            gradients: Lista de tuplas ``(grad_w, grad_b)`` já ordenada por camada.
        """
        for layer, (grad_w, grad_b) in zip(self.__layers, gradients):
            layer.update(
                self.__learning_rate * grad_w, self.__learning_rate * grad_b
            )

    def predict(self, inputs: np.ndarray) -> np.ndarray:
        """Gera predições para novos dados.

        Parâmetros:
            inputs: Dados de entrada, formato ``(n_amostras, n_features)``.

        Retorna:
            Predições da rede, formato ``(n_amostras, n_saídas)``.

        Raises:
            RuntimeError: Se ``fit()`` ainda não tiver sido chamado.
        """
        if not self.__is_fitted:
            raise RuntimeError(
                "O modelo ainda não foi treinado (chame a função fit() antes)."
            )

        return self.__feed_forward(inputs)

    def __summary(self) -> None:
        """Exibe um resumo do treinamento no terminal."""
        print("-" * 30)
        print("Training Summary")
        print("-" * 30)
        print(f"Epochs: {self.__history.epochs[-1]}")
        print(f"Learning rate: {self.__learning_rate:.6f}")
        print(f"Loss function: {type(self.__cost_function).__name__}")
        print(f"Layers: {len(self.__layers)}")
        print(f"Parameters: {sum(
            l.input_size * l.output_size + l.output_size for l in self.__layers
        )}")
        print(f"Final loss: {self.__history.losses[-1]:.6f}")
        print(f"Best loss: {min(self.__history.losses):.6f}")
        if self.__history.eval_losses:
            print(f"Final eval loss: {self.__history.eval_losses[-1]:.6f}")
            print(f"Best eval loss: {min(self.__history.eval_losses):.6f}")
        print(f"Loss reduction: {100 * (
            self.__history.losses[0] - self.__history.losses[-1]
        ) / self.__history.losses[0]:.2f}%")
        print("-" * 30)

    def __validate_output_activation(self) -> None:
        """Verifica compatibilidade entre a ativação de saída e a função de custo."""
        if not self.__layers:
            return

        last_activation = self.__layers[-1].activation

        softmax_sem_cce = isinstance(last_activation, Softmax) and not isinstance(
            self.__cost_function, CategoricalCrossEntropy
        )
        cce_sem_softmax = isinstance(
            self.__cost_function, CategoricalCrossEntropy
        ) and not isinstance(last_activation, Softmax)

        if softmax_sem_cce:
            raise ValueError(
                f"Softmax requer CategoricalCrossEntropy como função de custo, "
                f"mas foi recebida: {type(self.__cost_function).__name__}."
            )
        if cce_sem_softmax:
            raise ValueError(
                f"CategoricalCrossEntropy requer Softmax na camada de saída, "
                f"mas foi recebida: {type(last_activation).__name__}."
            )
