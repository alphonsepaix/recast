import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from typing import Optional
from distributions import *


class Model:
    def __init__(self,
                 C: int = 32,
                 dist: Distribution = Weibull,
                 R: int = 1,
                 learning_rate: float = 5e-3,
                 dropout_rate: float = 0.,
                 beta: float = np.log(10),
                 strategy: str = "tau_log_tau"
    ) -> None:
        """Initialize the model parameters."""

        if strategy not in ["tau_log_tau", "log_tau"]:
            raise ValueError(f"Unknown strategy for time encoding: {strategy}")

        self.C = C          # Number of units for the GRU layer.
        self.R = R          # Number of components for the mixture outputted by the dense layer.
        self.dist = dist

        self.rnn = keras.layers.GRU(
            units=C, return_sequences=True, return_state=True)
        self.dense = keras.layers.Dense(R * (dist.n_params + 1))

        self.learning_rate = learning_rate
        self.optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

        self.dropout_rate = dropout_rate
        self.beta = beta
        self.strategy = strategy

        self.magnitudes = False
        self.eps = 1e-8
        self.tau_mean = 1.
        self.mag_dist = GutenbergRichter(beta=beta)
        self.mag_mean = self.mag_dist.mean
          

    @property
    def layers(self) -> list[keras.layers.Layer]:
        return [self.rnn, self.dense]


    @property
    def weights(self) -> list[tf.Tensor]:
        return [layer.get_weights() for layer in self.layers]
    

    @property
    def trainable_weights(self) -> list[tf.Tensor]:
        """Return the trainable weights for gradient descent."""

        return self.rnn.trainable_weights + self.dense.trainable_weights
    

    def encode_times(self, inter_times: np.ndarray) -> tf.Tensor:
        """Encode the inter-arrival times."""
        inter_times = tf.expand_dims(tf.cast(inter_times, "float32"), -1)
        log_tau = tf.math.log(tf.maximum(inter_times, self.eps))
        if self.strategy == "log_tau":
            return log_tau - tf.math.log(self.tau_mean)
        elif self.strategy == "tau_log_tau":
            return tf.concat([inter_times - self.tau_mean,
                             log_tau - tf.math.log(self.tau_mean)], -1)
    

    def encode_magnitudes(self, magnitudes: np.ndarray) -> tf.Tensor:
        """Encode the magnitudes."""
        magnitudes = tf.cast(magnitudes, "float32")
        return tf.expand_dims(magnitudes - self.mag_mean, -1)
    

    @tf.function
    def get_context(self,
                    inter_times: np.ndarray,
                    magnitudes: Optional[np.ndarray] = None
    ) -> tf.Tensor:
        """
        Return the context vector produced by the GRU layer and shifted to the
        right so that each inter-arrival time parametrizes the distribution
        of the next one.
        """

        rnn_input = self.encode_times(inter_times)
        if self.magnitudes:
            rnn_input = tf.concat([rnn_input, self.encode_magnitudes(magnitudes)], -1)

        rnn_output = self.rnn(rnn_input)[0]
        # Shift the context vector to the right.
        context = tf.pad(rnn_output[:, :-1, :], [[0, 0], [1, 0], [0, 0]])
        
        return keras.layers.Dropout(self.dropout_rate)(context, training=True)


    def get_distributions(self, context: tf.Tensor) -> Mixture:
        """Return the distributions given the output of the GRU layer."""

        dense_output = self.dense(context)
        # Split the tensors according to the number of components and
        # the number of parameters of the underlying distribution.
        weights, *params = tf.split(dense_output, 1 + self.dist.n_params, -1)

        # Enforce constraints on the parameters.
        weights = tf.nn.softmax(weights)
        params = [tf.nn.softplus(param) for param in params]

        return Mixture(weights, self.dist(*params, eps=self.eps))
    

    @tf.function
    def nll_loss(self,
                 inter_times: np.ndarray,
                 seq_lengths: np.ndarray,
                 magnitudes: Optional[np.ndarray] = None
    ) -> tf.Tensor:
        """
        Return the negative log-likelihood on each sequence of the dataset.
        """

        context = self.get_context(inter_times, magnitudes)
        distributions = self.get_distributions(context)

        log_prob = distributions.log_prob(inter_times)
        log_surv = distributions.log_survival_function(inter_times)

        # The padded elements are set to 0.
        mask = tf.cumsum(tf.ones_like(inter_times), -1) \
            <= tf.cast(seq_lengths[..., np.newaxis], "float32")
        log_prob = tf.where(mask, log_prob, 0)

        # We only select the last inter-arrival time for each sequence
        # to compute the log-survival term.
        last_idx = tf.stack([tf.range(len(seq_lengths)), seq_lengths], -1)
        log_surv = tf.gather_nd(log_surv, last_idx)
        
        log_like = tf.reduce_sum(log_prob, -1) + log_surv
        return -log_like


    def fit(self,
            inter_times: np.ndarray,
            seq_lengths: list[int],
            t_end: float | list[float],
            epochs: int,
            magnitudes: Optional[np.ndarray] = None,
            return_distributions: bool = False,
            verbose: int = 1
    ) -> list[float]:
        """Train the model by maximum likelihood estimation."""

        history = {}
        if magnitudes is not None:
            self.magnitudes = True

        inter_times = tf.cast(inter_times, "float32")
        seq_lengths = tf.cast(seq_lengths, "int32")
        t_end = tf.cast(t_end, "float32")

        for epoch in tf.range(1, epochs + 1):
            with tf.GradientTape() as tape:
                context = self.get_context(inter_times, magnitudes)
                distributions = self.get_distributions(context)
                nll_loss = self.nll_loss(inter_times, seq_lengths, magnitudes)
                loss = tf.reduce_mean(nll_loss / t_end)

                history.setdefault("loss", []).append(loss.numpy())
                if return_distributions:
                    history.setdefault("distributions", []).append(distributions)

            # Perform gradient descent with regard to the trainable
            # weights of the model.
            gradients = tape.gradient(loss, self.trainable_weights)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))

            if verbose == 2 \
                or (verbose == 1 and (epoch == 1 or epoch % 10 == 0)):
                print(f"Loss at epoch {epoch:>4}: {loss:>8.3f}")
        
        return history
    

    def generate(self,
                 batch_size: int,
                 t_end: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate a batch of sequences from scratch."""

        state = tf.zeros([batch_size, self.C])
        inter_times = tf.zeros([batch_size, 0])
        magnitudes = tf.zeros([batch_size, 0])
        generated = False

        while not generated:
            next_distributions = self.get_distributions(state)
            
            next_inter_times = tf.expand_dims(
                next_distributions.sample(), -1)
            inter_times = tf.concat([inter_times, next_inter_times], -1)
            rnn_input = self.encode_times(next_inter_times)

            if self.magnitudes:
                next_magnitudes = self.mag_dist.sample(size=(batch_size, 1))
                magnitudes = tf.concat([magnitudes, next_magnitudes], -1)
                rnn_input = tf.concat([rnn_input, self.encode_magnitudes(next_magnitudes)], -1)
            
            state = self.rnn(rnn_input, initial_state=state)[1]
            
            generated = tf.reduce_min(tf.reduce_sum(inter_times, -1)) >= t_end

        seq_lengths = np.sum(np.cumsum(inter_times, -1) < t_end, -1)
        generated_data = np.cumsum(inter_times, -1)
        if self.magnitudes:
            generated_data = np.stack([generated_data, magnitudes], -1)
        return generated_data, seq_lengths
            

    def predict(self,
                arrival_times: np.ndarray,
                magnitudes: Optional[np.ndarray] = None,
                n: int = 1,
                return_distributions: bool = False
    ) -> tuple[np.ndarray, list[Distribution]] | np.ndarray:
        """Predict the next arrival times given a past sequence."""

        inter_times = np.diff(arrival_times, prepend=0)
        rnn_input = self.encode_times(inter_times)
        if self.magnitudes:
            rnn_input = tf.concat([rnn_input, self.encode_magnitudes(magnitudes)], -1)
    
        state = self.rnn(rnn_input[tf.newaxis, ...])[1]

        inter_times = []
        magnitudes = []
        distributions = []

        for _ in range(n):
            next_distribution = self.get_distributions(state)
            distributions.append(next_distribution)
            next_inter_time = next_distribution.sample()
            inter_times.append(next_inter_time[0])
    
            rnn_input = self.encode_times(next_inter_time)
            if self.magnitudes:
                next_magnitude = self.mag_dist.sample(size=1) 
                magnitudes.append(next_magnitude[0])
                rnn_input = tf.concat([rnn_input[:, tf.newaxis], self.encode_magnitudes(next_magnitude)[..., tf.newaxis]], -1)

            state = self.rnn(rnn_input, initial_state=state)[1]

        predictions = arrival_times[-1] + np.cumsum(inter_times)
        if self.magnitudes:
            predictions = np.stack([predictions, magnitudes], -1)

        if return_distributions:
            return predictions, distributions
        return predictions
    

    @property
    def build_params(self) -> dict:
        """Return all the build parameters of the model."""

        return {
            "C": self.C,
            "dist": self.dist,
            "R": self.R,
            "learning_rate": self.learning_rate,
            "dropout_rate": self.dropout_rate,
            "beta": self.beta,
            "strategy": self.strategy
        }
    

    def save(self, filename: str) -> None:
        """Save the model build parameters and weights to a file."""

        build_params = self.build_params
        rnn_input_dim = self.rnn.weights[0].shape[0]
        with open(filename, "wb") as f:
            pickle.dump([build_params,
                         self.weights,
                         self.magnitudes,
                         rnn_input_dim], f)


    @staticmethod
    def load(filename: str) -> "Model":
        """Load a model that was saved with the static method from a file."""

        with open(filename, "rb") as f:
            build_params, weights, magnitudes, input_dim = pickle.load(f)

        model = Model(*build_params.values())

        # Necessary to set the layers weights.
        model.rnn(np.random.randn(1, 1, input_dim))
        model.dense(np.random.randn(1, 1, model.C))
        model.magnitudes = magnitudes

        for w, layer in zip(weights, model.layers):
            layer.set_weights(w)

        return model
