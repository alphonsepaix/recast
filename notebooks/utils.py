# Fichier utilisé pour mettre rapidemment en place les notebooks


import os
import pickle
import sys


sys.path.insert(0, os.path.abspath('..'))


import jax
import jax.numpy as jnp
import matplotlib as mpl
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
layers = keras.layers
mnist = keras.datasets.mnist
import tensorflow_probability as tfp
tfd = tfp.distributions

from recast.distributions import *
from recast.etas import *
from recast.model import *


sns.set_style('whitegrid')


def poisson(T: float, mu: float) -> list[float]:
    """Génère un processus de Poisson homogène."""
    N = np.random.poisson(mu * T)
    return np.sort(np.random.uniform(0, T, size=N))


def poisson_inh(T: float, mu: float = 1) -> list[float] | None:
    """Génère un processus de Poisson inhomogène."""
    intensity = lambda t: 1 + np.sin(0.1 * t)
    x = np.linspace(0, T, 1000)
    sup = np.max(intensity(x))

    times = np.array([-1 / mu * np.log(np.random.uniform())])

    while times[-1] < T:
        tau = -1 / mu * np.log(np.random.uniform())
        next_time = times[-1] + tau
        D = np.random.uniform()
        if D <= intensity(next_time) / sup:
            times = np.append(times, next_time)

    return times[times <= T]


def intensity(t: float, seq: np.ndarray, mu: float = 1,
              alpha: float = 2, beta: float = 2.3) -> float:
        return mu + np.sum(alpha * np.exp(-beta * (t - seq[seq < t])))


def hawkes(mu: float = 1, alpha: float = 1, beta: float = 2,
           T: float = 100) -> list[float] | None:
    """Génère un processus de Hawkes."""
    times = np.array([-1 / mu * np.log(np.random.uniform())])

    if times.size == 0:
        return None

    while times[-1] < T:
        lambd_bar = intensity(times[-1], times, mu, alpha, beta)
        tau = -1 / lambd_bar * np.log(np.random.uniform())
        next_time = times[-1] + tau
        D = np.random.uniform()
        if D * lambd_bar <= intensity(next_time, times, mu, alpha, beta):
            times = np.append(times, next_time)

    return times[times <= T]


def cmp_dist(model: Model, seq: np.ndarray, start: int,
             ylim: tuple[float, float] = (0, 1)) -> None:
    """Compare les distributions ETAS et RECAST."""
    sub_seq = seq[:start]
    t = sub_seq[:, 0]
    m = sub_seq[:, 1]
    recast_dist = model.predict(t, m, 1, True)[1][0]
    etas_dist = ETAS(t, m)
    fig, axes = plt.subplots(2, 1, sharex='col', sharey='col', figsize=(6, 3))
    x = np.linspace(0, 3, 1000)
    axes[0].plot(x, recast_dist.prob(x), label='RECAST')
    axes[1].plot(x, etas_dist.prob(x), label='ETAS')
    axes[1].set_ylim(ylim)
    plt.show()


def plot_etas_seq(seq: pd.DataFrame, n: int = 0,
                  dpi: Optional[int] = None) -> None:
    """Affiche la séquence et les n plus grandes magnitudes."""
    fig, ax = plt.subplots(figsize=(6, 4), dpi=dpi)
    ax.plot(seq.time, seq.index + 1)
    high_mags_idx = seq.magnitude.sort_values(ascending=False).index[:n]
    for idx in high_mags_idx:
        t = seq.time.iloc[idx]
        m = seq.magnitude.iloc[idx]
        ax.scatter(t, idx + 1, label=f'{m:.2f}', marker='.', s=100)
        ax.vlines(t, 0, idx + 1, linestyle='--', alpha=0.5, colors='black')
    if n:
        ax.legend(loc='upper left', ncol=2)
    plt.show()


def get_seq(generated_data: np.ndarray, seq_lengths: np.ndarray,
            i: int) -> pd.DataFrame:
    """Retourne un tableau pandas qui contient les données générées."""
    arrival_times = generated_data[i, :seq_lengths[i], 0]
    magnitudes = generated_data[i, :seq_lengths[i], 1]
    seq = pd.DataFrame({'time': arrival_times, 'magnitude': magnitudes})
    return seq


def plot_seqs(seqs: list[np.ndarray]):
    for seq in seqs:
        plt.plot(seq[:, 0], np.cumsum(np.ones(seq.shape[0])))
    plt.show()


def cumsum(arr: np.ndarray) -> np.ndarray:
    return np.cumsum(np.ones_like(arr))


def predict_and_plot(model: Model, seq: np.ndarray, n_before: int,
                     n_after: Optional[int] = None,
                     dpi: Optional[int] = None) -> np.ndarray:
    """Affiche la nouvelle séquence avec les données prédites."""
    n_preds = n_after if n_after else n_before
    past_seq = seq[:-n_before]
    t = past_seq[:, 0]
    m = past_seq[:, 1]
    preds = model.predict(t, m, n_preds)
    t_preds = preds[:, 0]
    m_preds = preds[:, 1]
    seq_size = seq.shape[0]
    last_idx_target = min(seq_size - 1, seq_size - n_before + n_preds)
    t_targets = seq[-n_before:last_idx_target, 0]

    fig, ax = plt.subplots(dpi=dpi)
    ax.plot(t, cumsum(t), label='Séquence passée')
    ax.plot(t_preds, t.size + cumsum(t_preds), label='Prédictions', alpha=0.6)
    ax.plot(t_targets, t.size + cumsum(t_targets), label='Cibles', alpha=0.6)
    ax.axhline(y=t.size, linestyle='--', c='black', alpha=0.7)
    ax.legend()
    plt.show()

    new_seq = np.concatenate([past_seq[:, :2], preds], 0)
    return new_seq


def get_dist_at_epoch(i: int, dists: list[Mixture],
                      dist: Distribution = Weibull) -> Mixture:
    weights, params = dists[i].params
    weights = weights[0, 250]
    b = params[..., 0][0, 250]
    k = params[..., 1][0, 250]
    return Mixture(weights, dist(b, k))


def sgd(F, x_train, y_train, theta_0, lr, n_step):
    grad_F = jax.grad(F, argnums=1)

    thetas = [theta_0]

    for t in range(n_step):
        idx = np.random.randint(len(x_train))
        y_pred = F(x_train[idx], theta_0)
        loss = (y_train[idx] - y_pred) ** 2
        grads = grad_F(loss, theta_0)
        theta_0 -= lr * grads
        thetas.append(theta_0)

    return thetas


def data_generation():
    a, b = 3.3, -3.1

    def true_f(x):
        return x * jnp.exp(-(x - b) ** 2 /2 ) + jnp.exp(-(x - a) ** 2 / 2)

    n = 500
    x_train = np.random.uniform(low=-10, high=10, size=n)
    y_train = true_f(x_train) + 0.1 * np.random.randn(n)
    
    return x_train, y_train


def plot_sgd(F, x_train, y_train, thetas):
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    fig.tight_layout()

    thetas = np.array(thetas)

    ax = axes[0]

    ax.plot(thetas[:, 0], thetas[:, 1], zorder=1)
    ax.scatter(thetas[0, 0], thetas[0, 1], c='blue', s=40,
               label='Départ', zorder=2, marker='v')
    ax.scatter(thetas[-1][0], thetas[-1][1], c='red', s=40,
               label='Arrivée', zorder=2, marker='^')
    ax.scatter([3.3], [-3.1], label='Objectif', c='green', s=40, marker='>')
    ax.legend()
    ax.set_xlabel(r'$\theta_0$')
    ax.set_ylabel(r'$\theta_1$')

    xs = np.linspace(-10, 10, 500)

    ax = axes[1]
    ax.scatter(x_train, y_train, s=5, c='C0', alpha=0.5)
    ax.plot(xs, F(xs, thetas[0]), c='C1', linewidth=3,
            linestyle='--', label='Modèle')
    ax.set_title(r'Modèle initial (avec $\theta^0$)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    ax = axes[2]
    ax.scatter(x_train, y_train, s=5, c='C0', alpha=0.5)
    ax.plot(xs, F(xs, thetas[-1]), c='C1', linewidth=3,
            linestyle='--')
    ax.set_title(r'Modèle final (avec $\theta^T$)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    left_patch = mlines.Line2D([], [], color='C0', label='Cibles',
                               linestyle='', marker='o')
    right_patch = mpatches.Patch(color='C1', label='Modèle')
    fig.legend(handles=[left_patch, right_patch], bbox_to_anchor=[0.78, 0],
               ncol=2)


def F(x, theta):
    a = theta[0]
    b = theta[1]
    return x * jnp.exp(-(x - b) ** 2 / 2) + jnp.exp(-(x - a) ** 2 / 2)
