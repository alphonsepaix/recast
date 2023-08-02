import numpy as np
import pandas as pd
from collections.abc import Iterable
from typing import Optional
from distributions import ETAS


def etas(mu: float = 1,
         alpha: float = 2,
         bar_n:float = 0.9,
         p: float = 1.1, 
         c: float = 1e-9, 
         beta: float = np.log(10), 
         t_end: float = 1e3,
         max_len: Optional[int] = None
) -> np.ndarray | None:
    """Generate an ETAS sequence for the given parameters."""

    def logrand(): return np.log(np.random.uniform())

    A = bar_n / (beta * c ** (1 - p) / ((p - 1) * (beta - alpha)))
    tc = 0

    t = np.array([])
    m = np.array([])
    parent = np.array([])

    # Generate the background events.
    while tc < t_end:
        dt = -1 / mu * logrand()
        tc += dt
        if tc < t_end:
            t = np.append(t, tc)
            m = np.append(m, -1 / beta * logrand())
            parent = np.append(parent, -1)

    if A <= 0 or len(t) == 0:
        return
    
    # Generate the aftershocks.
    n = 0
    n_aftershocks = 0
    while True:
        tc = 0
        while True:
            tmp = (tc + c) ** (1 - p) + (p - 1) / (A * np.exp(alpha * m[n])) \
                * logrand()
            if tmp > 0:
                dt = tmp ** (1 / (1 - p)) - tc - c
                tc += dt
                if tc + t[n] < t_end:
                    t = np.append(t, tc + t[n])
                    m = np.append(m, -1 / beta * logrand())
                    parent = np.append(parent, n)
                    n_aftershocks += 1

                    # Return early if max_len was specified.
                    # We need to generate at least (max_len - 1) children
                    # otherwise we will be returning a simple Poisson process.
                    if max_len and n_aftershocks == max_len - 1:
                        break
                else:
                    break
            else:
                break
                
        # Sort each sequence by arrival times.
        idx = np.argsort(t)
        t = t[idx]
        m = m[idx]
        parent = parent[idx]
        n += 1

        # We need to sort before returning.
        if max_len and n_aftershocks == max_len - 1:
            break

        if n == len(t):
            break
    
    # Return a matrix with three dimensions.
    return np.stack([t, m, parent], -1)[:max_len]


def create_training_dataset(seqs: list[np.ndarray],
                            t_end: float | list[float]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a dataset suited for model training from a list of sequences."""

    seq_lengths = []
    inter_times = []
    magnitudes = []

    if not isinstance(t_end, Iterable):
        t_end = np.repeat(t_end, len(seqs))

    # Get the arrival times, inter-arrival times and magnitudes for each
    # sequence.
    for seq, t_end in zip(seqs, t_end):
        seq_lengths.append(len(seq))
        inter_times.append(np.diff(seq[:, 0], prepend=0, append=t_end))
        magnitudes.append(seq[:, 1])

    # Pad the inter-arrival times and magnitudes.
    max_length = np.max(seq_lengths)
    for i, length in enumerate(seq_lengths):
        inter_times[i] = np.pad(inter_times[i], (0, max_length - length))
        magnitudes[i] = np.pad(magnitudes[i], (0, max_length - length + 1))

    # Cast and return the matrices.
    inter_times = np.array(inter_times)
    magnitudes = np.array(magnitudes)
    seq_lengths = np.array(seq_lengths)
    return inter_times, magnitudes, seq_lengths


def generate_dataset(batch_size: int = 10,
                     verbose: bool = False,
                     **kwargs
) -> tuple[list[np.ndarray], tuple, float | list[float]]:
    """Generate a dataset suited for model training."""

    seqs = []
    if "max_len" in kwargs:
        t_end = []
        # Last event will determine the interval length.
        kwargs["max_len"] += 1
    else:
        t_end = kwargs.get("t_end", 1e3)    # t_end default to 1e3.

    for i in range(batch_size):
        seq = etas(**kwargs)
        if "max_len" in kwargs:
            t_end.append(seq[-1, 0])
            seq = np.delete(seq, -1, 0)
        seqs.append(seq)

        if verbose:
            print(f"Generating sequences: {i + 1:4}/{batch_size}\r", end="")
    print()     # new line
            
    return seqs, create_training_dataset(seqs, t_end), t_end


def to_frame(seq: np.ndarray) -> pd.DataFrame:
    """Return the sequence as a pandas dataframe."""

    columns = ["time", "magnitude", "parent"]
    if seq.shape[-1] == 2:
        columns.pop()
    return pd.DataFrame(seq, columns=columns)


def log_likelihood(seq: np.ndarray, t_end: float, **kwargs) -> float:
    """Compute the ETAS log-likelihood on the given sequence."""

    t = seq[:, 0]
    m = seq[:, 1]
    mu = kwargs.get("mu", 1)
    
    tau_distributions = [lambda x: mu * np.exp(-mu * x)]
    for i in range(1, seq.shape[0]):
        tau_distributions.append(ETAS(t[:i], m[:i], **kwargs).density)

    inter_times = np.diff(t, prepend=0)

    log_like = 0
    for f, t in zip(tau_distributions, inter_times):
        log_like += np.log(f(t))

    return log_like / t_end
