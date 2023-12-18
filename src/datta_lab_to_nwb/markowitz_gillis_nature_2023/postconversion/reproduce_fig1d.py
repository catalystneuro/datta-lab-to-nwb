import pandas as pd
import numpy as np
from pynwb import NWBHDF5IO
from neuroconv.utils import load_dict_from_file
import matplotlib.pyplot as plt
import colorcet as cc
from pathlib import Path


def reproduce_fig1d(file_path, metadata_path):
    metadata = load_dict_from_file(metadata_path)
    sorted_index2id = metadata["BehavioralSyllable"]["sorted_index2id"]
    n_frames = 361
    start = 3520
    with NWBHDF5IO(file_path, mode="r", load_namespaces=True) as io:
        nwbfile = io.read()
        start_time = nwbfile.processing["behavior"]["BehavioralSyllableOffline"].timestamps[start]
        stop_time = nwbfile.processing["behavior"]["BehavioralSyllableOffline"].timestamps[start + n_frames]
        photometry_timestamps = nwbfile.processing["ophys"]["ReferenceDfOverF"].timestamps[:]

        signal_dff = nwbfile.processing["ophys"]["SignalDfOverF"].data[:]
        signal_dff = signal_dff[np.logical_and(photometry_timestamps >= start_time, photometry_timestamps < stop_time)]
        reference_dff = nwbfile.processing["ophys"]["ReferenceDfOverF"].data[:]
        reference_dff = reference_dff[
            np.logical_and(photometry_timestamps >= start_time, photometry_timestamps < stop_time)
        ]

        position = pd.DataFrame(
            nwbfile.processing["behavior"]["Position"]["SpatialSeries"].data, columns=["x", "y", "height"]
        )
        angle = pd.Series(nwbfile.processing["behavior"]["CompassDirection"]["HeadOrientation"].data)
        syllables = pd.Series(nwbfile.processing["behavior"]["BehavioralSyllableOffline"].data).map(sorted_index2id)
    vel_height = (
        position["height"].interpolate(limit_direction="both").diff(2).iloc[start : start + n_frames].to_numpy()
        / 2
        * 30
    )
    velocity = np.sqrt(
        position["x"].interpolate(limit_direction="both").diff() ** 2
        + position["y"].interpolate(limit_direction="both").diff() ** 2
    )
    acc = velocity.diff(2).iloc[start : start + n_frames].to_numpy() * 30
    vel = velocity.iloc[start : start + n_frames].to_numpy() * 30
    vel_angle = angle.interpolate(limit_direction="both").diff(2).iloc[start : start + n_frames].to_numpy() * 30 * -1
    syllables = syllables.iloc[start : start + n_frames].to_numpy()
    t = np.arange(n_frames) / 30

    time = t
    fig, ax = plt.subplots(6, 1)
    fig.dpi = 200

    ax[0].plot(time, vel, color="k")
    ax[0].set_xlim(time[0], time[-1])
    ax[0].set_xticks([time[0], time[-1]])
    ax[0].set_xticklabels([])
    ax[0].set_ylim(0, 15 * 30)
    ax[0].set_yticks([0, 15 * 30])
    ax[0].set_ylabel("mm/s")
    ax[0].text(0, 1, "Velocity", transform=ax[0].transAxes)
    ax[0].spines["top"].set_visible(False)
    ax[0].spines["right"].set_visible(False)

    ax[1].plot(time, acc, color="k")
    ax[1].set_xlim(time[0], time[-1])
    ax[1].set_xticks([time[0], time[-1]])
    ax[1].set_xticklabels([])
    ax[1].set_ylim(-5 * 30, 5 * 30)
    ax[1].set_ylabel("mm/s2")
    ax[1].set_yticks([-150, 0, 150])
    ax[1].text(0, 1, "Acceleration", transform=ax[1].transAxes)
    ax[1].spines["top"].set_visible(False)
    ax[1].spines["right"].set_visible(False)

    ax[2].plot(time, vel_angle, color="k")
    ax[2].set_xlim(time[0], time[-1])
    ax[2].set_xticks([time[0], time[-1]])
    ax[2].set_xticklabels([])
    ax[2].set_ylim(-0.55 * 30, 0.55 * 30)
    ax[2].set_yticks([-16, 0, 16])
    ax[2].set_ylabel("rad/s")
    ax[2].text(0, 1, "Angular velocity", transform=ax[2].transAxes)
    ax[2].spines["top"].set_visible(False)
    ax[2].spines["right"].set_visible(False)

    ax[3].plot(time, vel_height, color="k")
    ax[3].set_xlim(time[0], time[-1])
    ax[3].set_xticks([time[0], time[-1]])
    ax[3].set_xticklabels([])
    ax[3].set_ylim(-5 * 30, 5 * 30)
    ax[3].set_yticks([-5 * 30, 0, 5 * 30])
    ax[3].set_ylabel("mm/s")
    ax[3].text(0, 1, "Height Velocity", transform=ax[3].transAxes)
    ax[3].spines["top"].set_visible(False)
    ax[3].spines["right"].set_visible(False)

    ax[4].imshow(syllables[np.newaxis, :], aspect="auto", interpolation="none", cmap=cc.cm.glasbey, vmin=0)
    ax[4].set_xticks([])
    ax[4].set_yticks([])
    ax[4].text(0, 1.1, "Syllable", transform=ax[4].transAxes)

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


if __name__ == "__main__":
    file_path = Path(
        "/Volumes/T7/CatalystNeuro/NWB/Datta/conversion_nwb/reinforcement-photometry-2891f649-4fbd-4119-a807-b8ef507edfab.nwb"
    )
    metadata_path = Path(
        "/Users/pauladkisson/Documents/CatalystNeuro/NWB/DattaConv/catalystneuro/datta-lab-to-nwb/src/datta_lab_to_nwb/markowitz_gillis_nature_2023/markowitz_gillis_nature_2023_metadata.yaml"
    )
    reproduce_fig1d(file_path, metadata_path)
