"""Ce fichier fournit une implémentation de l'algorithme ETAS."""


from collections.abc import Iterable
import os
from typing import Optional
import shutil
import subprocess

import numpy as np
import pandas as pd

from .distributions import ETAS


def logrand():
    return np.log(np.random.uniform())


def etas_rust(**kwargs) -> np.ndarray | None:
    cmd = 'etas'
    if shutil.which(cmd) == None:
        raise FileNotFoundError('could not find the Rust binary')
    for arg, value in kwargs.items():
        arg = arg.replace('_', '-')
        cmd += f' --{arg} {value}'
    subprocess.run(cmd, shell=True)
    filename = 'data.csv'
    try:
        data = pd.read_csv(filename, index_col=0)
        os.remove(filename)
        return data.to_numpy()
    except FileNotFoundError:
        return None


def etas_py(mu: float = 1, alpha: float = 2, bar_n: float = 0.9,
            p: float = 1.1, c: float = 1e-9, beta: float = np.log(10),
            t_end: float = 1e3, max_len: Optional[int] = None
            ) -> np.ndarray | None:
    """Génère et renvoie une séquence ETAS dans un tableau NumPy
    (le premier axe contient les temps d'arrivée, le deuxième les
    magnitudes et le troisième les parents correspondants).
    """
    if max_len is not None and max_len < 0:
        raise ValueError('max_len must be positive')

    A = bar_n / (beta * c ** (1 - p) / ((p - 1) * (beta - alpha)))
    tc = 0
    t = np.array([])
    m = np.array([])
    parent = np.array([])

    # Génère les séismes de fond.
    while tc < t_end:
        dt = -1 / mu * logrand()
        tc += dt
        if tc < t_end:
            t = np.append(t, tc)
            m = np.append(m, -1 / beta * logrand())
            parent = np.append(parent, -1)

    if len(t) == 0:
        return None

    if A <= 0:
        return np.stack([t, m, parent], -1)[:max_len]

    n = 0

    # Génère les répliques.
    while True:
        tc = 0
        while True:
            tmp = (tc + c) ** (1 - p) \
                + (p - 1) / (A * np.exp(alpha * m[n])) * logrand()
            if tmp > 0:
                dt = tmp ** (1 / (1 - p)) - tc - c
                tc += dt
                if tc + t[n] < t_end:
                    t = np.append(t, tc + t[n])
                    m = np.append(m, -1 / beta * logrand())
                    parent = np.append(parent, n)
                else:
                    break
            else:
                break

        # Trie la séquence par temps d'arrivée.
        idx = np.argsort(t)
        t = t[idx]
        m = m[idx]
        parent = parent[idx]
        n += 1

        if max_len is not None:
            if max_len == n - 1:
                break

        if n == len(t):
            break

    return np.stack([t, m, parent], -1)[:max_len]


def etas(engine='python', **kwargs) -> np.ndarray | None:
    kwargs.pop('filename', None)
    kwargs.pop('verbose', None)
    if engine == 'python':
        data = etas_py(**kwargs)
    elif engine == 'rust':
        try:
            data = etas_rust(**kwargs)
        except FileNotFoundError:
            print('Falling back to the Python engine because the Rust binary \
was not found.')
            data = etas_py(**kwargs)
    else:
        raise ValueError('unknown engine')
    if data is None:
        print('The generated sequence was empty.')
    return data


def create_training_dataset(seqs: list[np.ndarray],
                            t_end: float | list[float]
                            ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Génère des listes adaptées pour l'entraînement à partir
    d'une liste de séquences.
    """
    seq_lengths = []
    inter_times = []
    magnitudes = []

    if not isinstance(t_end, Iterable):
        t_end = np.repeat(t_end, len(seqs))

    # On parcourt chaque séquence.
    for seq, t_end in zip(seqs, t_end):
        seq_lengths.append(len(seq))
        inter_times.append(np.diff(seq[:, 0], prepend=0, append=t_end))
        magnitudes.append(seq[:, 1])

    # On pad les données.
    max_length = np.max(seq_lengths)
    for i, length in enumerate(seq_lengths):
        inter_times[i] = np.pad(inter_times[i], (0, max_length - length))
        magnitudes[i] = np.pad(magnitudes[i], (0, max_length - length + 1))

    inter_times = np.array(inter_times)
    magnitudes = np.array(magnitudes)
    seq_lengths = np.array(seq_lengths)
    return inter_times, magnitudes, seq_lengths, t_end


def generate_dataset(batch_size: int = 10,
                     verbose: bool = False,
                     **kwargs
                     ) -> tuple[list[np.ndarray], tuple, float | list[float]]:
    """Génère un jeu de données pour l'entraînement."""
    seqs = []
    if 'max_len' in kwargs:
        t_end = []
        # Le dernier événement fixera la fin de l'intervalle.
        kwargs['max_len'] += 1
    else:
        t_end = kwargs.get('t_end', 1e3)

    # On génère chaque séquence. Si max_len est spécifié, on ajoute
    # le dernier temps d'arrivée de la séquence à la liste t_end
    # avant de retirer le dernier élément.
    for i in range(batch_size):
        seq = etas(**kwargs)
        if 'max_len' in kwargs:
            t_end.append(seq[-1, 0])
            seq = np.delete(seq, -1, 0)
        seqs.append(seq)

        if verbose:
            print(f'Generating sequences: {i + 1}/{batch_size}\r', end='')
    print()

    return seqs, create_training_dataset(seqs, t_end)


def to_frame(seq: np.ndarray) -> pd.DataFrame:
    """Convertit la séquence en tableau pandas."""
    columns = ['time', 'magnitude', 'parent']
    if seq.shape[-1] == 2:
        columns.pop()
    return pd.DataFrame(seq, columns=columns)


def log_likelihood(seq: np.ndarray, t_end: float, **kwargs) -> float:
    """Calcule la log-vraisemblance de la séquence donnée."""
    t = seq[:, 0]
    m = seq[:, 1]
    mu = kwargs.get('mu', 1)

    # La distribution du premier temps d'attente suit une loi exponentielle.
    tau_distributions = [lambda x: mu * np.exp(-mu * x)]

    for i in range(1, seq.shape[0]):
        tau_distributions.append(ETAS(t[:i], m[:i], **kwargs).density)

    inter_times = np.diff(t, prepend=0)
    log_like = 0
    for f, tau in zip(tau_distributions, inter_times):
        log_like += np.log(f(tau))

    # Il manque le terme de survie.

    return log_like / t_end
