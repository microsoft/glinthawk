import matplotlib.pyplot as plt


def nice_plt_cfg():
    plt.rcParams.update({
        "text.usetex": True,
        'legend.fontsize': 6,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'axes.labelsize': 8,
        'axes.titlesize': 8,
        'figure.labelsize': 8,
        'figure.titlesize': 8,
        'hatch.linewidth': 0.5
    })


def nice_ax(ax: plt.Axes):
    ax.grid()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.plot(1, 0, ">k", transform=ax.transAxes, clip_on=False)
    ax.plot(0, 1, "^k", transform=ax.transAxes, clip_on=False)
    # ax.legend(ncol=3, loc='lower left', bbox_to_anchor=(-0.15, 1.03, 1.2, 0.1), mode="expand", handlelength=3)
