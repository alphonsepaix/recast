"""Ce fichier fournit une implémentation des lois de Weibull, Gamma
et des lois de mélange basées sur celles-ci.
"""


from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd


class Distribution:
    """Cette classe représente une distribution de probabilité générale."""

    def prob(self, x: Any) -> tf.Tensor:
        raise NotImplementedError

    def log_prob(self, x: Any) -> tf.Tensor:
        raise NotImplementedError

    def survival(self, x: Any) -> tf.Tensor:
        raise NotImplementedError

    def log_survival_function(self, x: Any) -> tf.Tensor:
        raise NotImplementedError

    @property
    def tfd(self) -> tfd.Distribution:
        """Renvoie la distribution TensorFlow équivalente pour le tirage."""
        raise NotImplementedError

    @property
    def params(self) -> tf.Tensor:
        """Renvoie les paramètres de la loi."""
        raise NotImplementedError

    def sample(self, shape: tuple[int] = ()) -> np.ndarray:
        """Effectue un tirage à partir de la distribution TensorFlow
        sous-jacente.
        """
        return self.tfd.sample(shape).numpy()


class Mixture(Distribution):
    """Cette classe représente une loi de mélange basée sur une unique loi."""

    def __init__(self, weights: Any, dist: Distribution) -> None:
        self.weights = tf.cast(weights, 'float32')
        self.dist = dist
        self.eps = dist.eps
        self.n_params = self.weights.shape[-1] * (dist.n_params + 1)

    def prob(self, x: Any) -> tf.Tensor:
        if not isinstance(x, tf.Tensor):
            x = tf.constant(x, dtype='float32')
        # tf.newaxis est nécessaire pour le broadcasting.
        return tf.reduce_sum(
            self.weights * self.dist.prob(x[..., tf.newaxis]), -1)

    def log_prob(self, x: Any) -> tf.Tensor:
        y = self.prob(x)
        return tf.math.log(tf.math.maximum(y, self.eps))

    def survival(self, x: Any) -> tf.Tensor:
        if not isinstance(x, tf.Tensor):
            x = tf.constant(x, dtype='float32')
        # tf.newaxis est nécessaire ici également.
        return tf.reduce_sum(
            self.weights * self.dist.survival(x[..., tf.newaxis]), -1)

    def log_survival_function(self, x: Any) -> tf.Tensor:
        y = self.survival(x)
        return tf.math.log(tf.math.maximum(y, self.eps))

    @property
    def tfd(self) -> tfd.Distribution:
        return tfd.MixtureSameFamily(
            mixture_distribution=tfd.Categorical(probs=self.weights),
            components_distribution=self.dist.tfd
        )

    @property
    def params(self) -> tuple[tf.Tensor, tf.Tensor]:
        return self.weights, self.dist.params


class Weibull(Distribution):
    """Cette classe fournit les fonctions usuelles pour la
    loi de Weibull.
    """

    name = 'Weibull'
    n_params = 2

    def __init__(self, b: Any, k: Any, eps: float = 1e-8) -> None:
        self.eps = eps
        self.b = tf.math.maximum(tf.cast(b, 'float32'), self.eps)
        self.k = tf.math.maximum(tf.cast(k, 'float32'), self.eps)

    def prob(self, x: Any) -> tf.Tensor:
        return tf.math.exp(self.log_prob(x))

    def log_prob(self, x: Any) -> tf.Tensor:
        x = tf.maximum(tf.cast(x, 'float32'), self.eps)
        return tf.math.log(self.b) + tf.math.log(self.k) \
            + (self.k - 1) * tf.math.log(x) - self.b * tf.math.pow(x, self.k)

    def survival(self, x: Any) -> tf.Tensor:
        return tf.math.exp(self.log_survival_function(x))

    def log_survival_function(self, x: Any) -> tf.Tensor:
        x = tf.maximum(tf.cast(x, 'float32'), self.eps)
        return -self.b * tf.math.pow(x, self.k)

    @property
    def tfd(self) -> tfd.Distribution:
        return tfd.Weibull(self.k, tf.math.pow(self.b, -1 / self.k))

    @property
    def params(self) -> tf.Tensor:
        return tf.stack([self.b, self.k], -1)


class Gamma(Distribution):
    """Cette classe fournit les fonctions usuelles pour la
    loi Gamma.
    """

    name = 'Gamma'
    n_params = 2

    def __init__(self, alpha: Any, beta: Any, eps: float = 1e-8) -> None:
        self.eps = eps
        self.alpha = tf.math.maximum(tf.cast(alpha, 'float32'), self.eps)
        self.beta = tf.math.maximum(tf.cast(beta, 'float32'), self.eps)

    def prob(self, x: Any) -> tf.Tensor:
        return tf.math.exp(self.log_prob(x))

    def log_prob(self, x: Any) -> tf.Tensor:
        x = tf.maximum(tf.cast(x, 'float32'), self.eps)
        return (self.alpha - 1) * tf.math.log(x) \
            + self.alpha * tf.math.log(self.beta) \
            - self.beta * x \
            - tf.math.lgamma(self.alpha)

    def survival(self, x: Any) -> tf.Tensor:
        x = tf.maximum(tf.cast(x, 'float32'), self.eps)
        return 1 - tf.math.igamma(self.alpha, self.beta * x) \
            / tf.math.exp(tf.math.lgamma(self.alpha))

    def log_survival_function(self, x: Any) -> tf.Tensor:
        y = tf.maximum(self.survival(x), self.eps)
        return tf.math.log(x)

    @property
    def tfd(self) -> tfd.Distribution:
        return tfd.Gamma(self.alpha, self.beta)

    @property
    def params(self) -> tf.Tensor:
        return tf.stack([self.alpha, self.beta], -1)


class ETAS(Distribution):
    """Cette classe fournit la densité du prochain temps d'attente
    pour une séquence et des paramètres donnés.
    """

    def __init__(self, arrival_times: np.ndarray, magnitudes: np.ndarray,
                 alpha: float = 2, bar_n: float = 0.9, p: float = 1.1,
                 c: float = 1e-9, beta: float = np.log(10),
                 eps: float = 1e-10) -> None:
        self.eps = eps
        A = bar_n * (p - 1) * (beta - alpha) / (beta * c ** (1 - p))

        def d(x: float) -> float:
            t = arrival_times
            m = magnitudes
            lhs = A * np.sum(np.exp(alpha * m) * (x + t[-1] + c - t) ** (-p))
            rhs_1 = (x + t[-1] + c - t) ** (1 - p)
            rhs_2 = (t[-1] + c - t) ** (1 - p)
            rhs = np.exp(-A / (1 - p) *
                         np.sum(np.exp(alpha * m) * (rhs_1 - rhs_2)))
            return lhs * rhs

        self.density = np.vectorize(d)

    def prob(self, x: Any) -> np.ndarray:
        return self.density(x)

    def log_prob(self, x: Any) -> np.ndarray:
        y = np.maximum(self.prob(x), self.eps)
        return np.log(y)


class Exponential(Distribution):
    """Cette classe fournit les fonctions usuelles pour la loi
    exponentielle.
    """

    def __init__(self, beta: float = np.log(10)):
        self.beta = beta
        self.mean = 1 / beta

        def d(x: float) -> float:
            return beta * np.exp(-beta * x)

        self.density = np.vectorize(d)

    def prob(self, x: Any) -> np.ndarray:
        return self.density(x)

    def sample(self, size: Optional[tuple[int]] = None):
        return -1 / self.beta * np.log(np.random.uniform(size=size))
