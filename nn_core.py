"""
Neural network model factory and training utilities.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models


@dataclass
class NetworkConfig:
    """Defines the architecture and training hyperparameters for the neural network."""
    hidden_layers: list[int]      # e.g. [64, 32] -> two hidden layers
    activation: str = "relu"      # relu | tanh | sigmoid
    learning_rate: float = 0.001
    optimizer: str = "adam"       # adam | sgd | rmsprop

    def describe(self) -> str:
        """Return a human-readable description of the network architecture."""
        sizes = [784, *self.hidden_layers, 10]
        return " → ".join(str(s) for s in sizes)


def load_mnist():
    """Load MNIST and normalize to [0, 1] flat vectors."""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 784).astype("float32") / 255.0
    x_test = x_test.reshape(-1, 784).astype("float32") / 255.0
    return (x_train, y_train), (x_test, y_test)


def build_model(cfg: NetworkConfig) -> tf.keras.Model:
    """Construct a Keras model based on the provided configuration."""
    model = models.Sequential(name="DigitNet")
    model.add(layers.Input(shape=(784,), name="pixels"))
    for i, units in enumerate(cfg.hidden_layers, start=1):
        model.add(layers.Dense(units, activation=cfg.activation, name=f"hidden_{i}"))
    model.add(layers.Dense(10, activation="softmax", name="output"))

    opt = {
        "adam": tf.keras.optimizers.Adam,
        "sgd": tf.keras.optimizers.SGD,
        "rmsprop": tf.keras.optimizers.RMSprop,
    }[cfg.optimizer](learning_rate=cfg.learning_rate)

    model.compile(
        optimizer=opt,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


class StreamlitProgress(tf.keras.callbacks.Callback):
    """Push per-epoch metrics back into the Streamlit UI."""

    def __init__(self, on_epoch_end: Callable[[int, dict], None]):
        super().__init__()
        self._on_epoch_end = on_epoch_end

    def on_epoch_end(self, epoch, logs=None):  # noqa: D401
        """Called by Keras after each epoch. Delegates to the provided callback."""
        self._on_epoch_end(epoch, logs or {})


def train_model(
    model: tf.keras.Model,
    data,
    epochs: int = 5,
    batch_size: int = 128,
    train_size: int = 10000,
    on_epoch_end: Callable[[int, dict], None] | None = None,
):
    """Train the model on the provided data, optionally reporting progress after each epoch."""
    (x_train, y_train), (x_test, y_test) = data
    # Use a subset for snappy interactive training
    x_train = x_train[:train_size]
    y_train = y_train[:train_size]

    callbacks = []
    if on_epoch_end is not None:
        callbacks.append(StreamlitProgress(on_epoch_end))

    history = model.fit(
        x_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test[:2000], y_test[:2000]),
        verbose=0,
        callbacks=callbacks,
    )
    return history


def get_layer_activations(model: tf.keras.Model, x: np.ndarray) -> list[np.ndarray]:
    """Return activations from every Dense layer for input x (shape (1, 784))."""
    outputs = [layer.output for layer in model.layers if isinstance(layer, layers.Dense)]
    activation_model = tf.keras.Model(inputs=model.inputs, outputs=outputs)
    acts = activation_model.predict(x, verbose=0)
    if not isinstance(acts, list):
        acts = [acts]
    return acts
