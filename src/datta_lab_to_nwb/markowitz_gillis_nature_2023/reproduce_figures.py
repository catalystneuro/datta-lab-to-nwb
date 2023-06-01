import pandas as pd
import numpy as np
from pynwb import NWBHDF5IO
import matplotlib.pyplot as plt
import colorcet as cc


def reproduce_fig1d(file_path):
    n_frames = 361
    start = 3520
    with NWBHDF5IO(file_path, mode="r") as io:
        nwbfile = io.read()
        signal_dff = nwbfile.processing["ophys"]["signal_dff"].data[start : start + n_frames]
        reference_dff = nwbfile.processing["ophys"]["reference_dff"].data[start : start + n_frames]

    time = np.arange(n_frames) / 30
    fig, ax = plt.subplots(6, 1)
    fig.dpi = 200

    ax[5].plot(time, reference_dff, color="#afafaf")
    ax[5].plot(time, signal_dff, color="g")
    ax[5].set_ylim(-0.02, 0.07)
    ax[5].set_yticks([-0.02, 0.07])
    ax[5].set_xlim(time[0], time[-1])
    ax[5].set_xticks([time[0], time[-1]])
    ax[5].set_ylabel("dF/F0")
    ax[5].set_xlabel("Time (seconds)")
    ax[5].spines["top"].set_visible(False)
    ax[5].spines["right"].set_visible(False)

    plt.subplots_adjust(hspace=0.55)
    plt.suptitle("Figure 1d")
    plt.show()
