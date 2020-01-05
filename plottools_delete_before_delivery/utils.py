import math

import matplotlib.pyplot as plt


def z_table(confidence):
    return {
        0.99: 2.576,
        0.95: 1.96,
        0.90: 1.645
    }[confidence]


def multi_confidence_plot(names, results, confidence=0.95, xlabel="", ylabel="",
                          display=True, save=False, filename=None, title="", xticks=None):

    N = len(results[0, 0])

    current_figure, ax = plt.subplots(1)
    xticks = range(1, N + 1) if xticks is None else xticks

    for a, agent_name in enumerate(names):

        means = results[a, 0]
        stds = results[a, 1]
        errors = z_table(confidence) * (stds / math.sqrt(N))

        ax.plot(xticks, means, lw=2, label=f"{agent_name} (N=1)", marker="o")
        ax.fill_between(xticks, means + errors, means - errors, alpha=0.20)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.legend(loc='upper left')
    ax.grid()

    if display:
        plt.show()

    if save:
        filename = title if filename is None else filename
        current_figure.savefig(filename)

    plt.close()


def single_confidence_plot(means, stds, confidence=0.95, xlabel="", ylabel="",
                           display=True, save=False, filename=None, title="", xticks=None):

    assert len(means) == len(stds)

    N = len(means)
    errors = z_table(confidence) * (stds / math.sqrt(N))
    current_figure, ax = plt.subplots(1)
    xticks = range(1, N + 1) if xticks is None else xticks
    ax.plot(xticks, means, lw=2, label=f"(N={N})", marker="o")
    ax.fill_between(xticks, means + errors, means - errors, alpha=0.20)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.legend(loc='upper left')
    ax.grid()

    if display:
        plt.show()

    if save:
        filename = title if filename is None else filename
        current_figure.savefig(filename)

    plt.close()
