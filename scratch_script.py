# scratch script to test various implementations in datta-lab-to-nwb
from datta_lab_to_nwb.markowitz_gillis_nature_2023 import markowitz_gillis_nature_2023behaviorinterface
from pynwb import NWBFile
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import colorcet as cc


def run_conversion():
    file_path = "/Volumes/T7/CatalystNeuro/NWB/Datta/dopamine-reinforces-spontaneous-behavior/dlight_raw_data/dlight_photometry_processed_full.parquet"
    example_session = "2891f649-4fbd-4119-a807-b8ef507edfab"
    interface = markowitz_gillis_nature_2023behaviorinterface.MarkowitzGillisNature2023BehaviorInterface(
        file_path, example_session
    )
    metadata = interface.get_metadata()
    nwbfile = NWBFile(**metadata["NWBFile"])
    nwbfile = interface.run_conversion(nwbfile, metadata)
    return nwbfile


def reproduce_fig1d(nwbfile):
    n_frames = 361
    start = 3520
    position = pd.DataFrame(
        nwbfile.processing["behavior"]["Position"]["SpatialSeries"].data, columns=["x", "y", "height"]
    )
    angle = pd.DataFrame(
        nwbfile.processing["behavior"]["CompassDirection"]["OrientationEllipse"].data, columns=["angle"]
    )
    syllables = pd.DataFrame(
        nwbfile.processing["behavior"]["SyllableTimeSeries"].time_series["BehavioralSyllable"].data,
        columns=["syllable"],
    )
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

    plt.subplots_adjust(hspace=0.55)
    plt.suptitle("Figure 1d")
    plt.show()


if __name__ == "__main__":
    nwbfile = run_conversion()
    reproduce_fig1d(nwbfile)
