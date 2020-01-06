from collections import defaultdict
from multiprocessing import RLock

import matplotlib.pyplot as plt
import numpy as np

from plottools_delete_before_delivery.Stats import Stats


class Plot:

    def __init__(self, title,
                 x_label, y_label,
                 num_xticks, x_tick_step,
                 confidence=0.95,
                 ymin=None, ymax=None, colors=None):

        self._title = title

        self._stats = defaultdict(lambda: Stats(num_xticks, confidence))
        self._colors = {} if colors is None else colors
        self._legend = {}

        self._x_label = x_label
        self._y_label = y_label

        self._x_tick_step = x_tick_step
        self._x_ticks = num_xticks

        self._current_figure = None
        self._highest_y_value = 1

        self._lock = RLock()

        self._ymin = ymin
        self._ymax = ymax

    def num_runs(self, agent_name):
        stats = self._stats[agent_name]
        return stats.num_runs

    def add_run(self, agent_name, run, color=None, add_to_legend=True):

        if color is not None:
            self._colors[agent_name] = color

        self._legend[agent_name] = add_to_legend

        with self._lock:
            # First measurement added for agent, create color
            if agent_name not in self._colors:
                color = self._random_color(list(self._colors.values()))
                self._colors[agent_name] = color

            if len(run) < self._x_ticks:
                padded_run = np.ones((self._x_ticks,)) * run[-1]
                padded_run[:run.shape[0]] = run
                run = padded_run

            stats = self._stats[agent_name]
            stats.update(run)

            self._highest_y_value = max(int(run.max()) + 1, self._highest_y_value)

    def show(self, error_fill=True, error_fill_transparency=0.25):
        if len(self._stats) > 0:
            if self._current_figure is not None: plt.close(self._current_figure)
            self._make_fig(error_fill, error_fill_transparency)
            self._current_figure.show()
            plt.close(self._current_figure)
            del self._current_figure
            self._current_figure = None

    def savefig(self, filename=None, error_fill=True, error_fill_transparency=0.25):
        if len(self._stats) > 0:
            if filename is None: filename = self._title
            if self._current_figure is not None: plt.close(self._current_figure)
            self._make_fig(error_fill, error_fill_transparency)
            self._current_figure.savefig(filename)
            plt.close(self._current_figure)
            del self._current_figure
            self._current_figure = None

    def save(self, directory):
        # 1 - Save Metadata
        # 2 - Save env models evaluation per agent (numpy matrix)
        pass

    def load(self, directory):
        # 1 - Load Metadata
        # 2 - Load env models evaluation per agent (numpy matrix)
        pass

    def _make_fig(self, error_fill=True, error_fill_transparency=0.25, show_legend=True):

        x_ticks = (np.arange(self._x_ticks) + 1) * self._x_tick_step
        self._current_figure, ax = plt.subplots(1)

        for agent_name, stats in self._stats.items():

            num_runs = stats.num_runs
            means = stats.means
            errors = stats.errors

            color = self._colors[agent_name]

            if self._legend[agent_name]:
                ax.plot(x_ticks, means, lw=2, label=f"{agent_name} (N={num_runs})", color=color, marker="o")
            else:
                ax.plot(x_ticks, means, lw=2, color=color, marker="o")

            if error_fill:
                ax.fill_between(x_ticks, means + errors, means - errors, facecolor=color, alpha=error_fill_transparency)

        ax.set_title(self._title)
        ax.set_xlabel(self._x_label)
        ax.set_ylabel(self._y_label)

        if self._ymin is not None:
            ax.set_ylim(bottom=self._ymin)
        else:
            ax.set_ylim(top=self._highest_y_value)

        if self._ymin is not None:
            ax.set_ylim(top=self._ymax)

        if show_legend:
            ax.legend()

        ax.grid()

    def showbar (self, error_fill=True, error_fill_transparency=0.25):
        if len(self._stats) > 0:
            if self._current_figure is not None: plt.close(self._current_figure)
            self._make_fig_bar(error_fill, error_fill_transparency)
            self._current_figure.show()
            plt.close(self._current_figure)
            del self._current_figure
            self._current_figure = None

    def savefigbar (self, filename=None, error_fill=True, error_fill_transparency=0.25):
        if len(self._stats) > 0:
            if filename is None: filename = self._title
            if self._current_figure is not None: plt.close(self._current_figure)
            self._make_fig_bar(error_fill, error_fill_transparency)
            self._current_figure.savefig(filename)
            plt.close(self._current_figure)
            del self._current_figure
            self._current_figure = None

    def _make_fig_bar (self, error_fill=True, error_fill_transparency=0.25, show_legend=True):

        agents_names = []
        agents_mean = []
        agents_error = []
        agents_color = []
        agents_labels = []

        for agent_name, stats in self._stats.items():
            print (stats.means, stats._std_devs)
            num_runs = stats.num_runs
            agents_names.append(agent_name)
            agents_mean.append (stats.means[0])
            agents_error.append (stats.errors[0])
            agents_color.append (self._colors[agent_name])
            agents_labels.append(f"{agent_name} (N={num_runs})")
            """
            if self._legend[agent_name]:
                ax.bar(x_ticks, means, lw=2, label=f"{agent_name} (N={num_runs})", color=color, marker="o")
            else:
                ax.bar(x_ticks, means, lw=2, color=color, marker="o")
            """
        # Build the plot
        x_ticks = np.arange(len(agents_names))
        self._current_figure, ax = plt.subplots()
        ecolor = 'black'

        ax.bar(x_ticks,agents_mean, yerr=agents_error, align='center', alpha=0.5, color=agents_color, capsize=10)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(agents_labels)

        ax.yaxis.grid(True)
        # ax.set_title(self._title)

        ax.set_ylabel(self._y_label)

        if self._ymin is not None:
            ax.set_ylim(bottom=self._ymin)
        else:
            ax.set_ylim(top=self._highest_y_value)

        if self._ymin is not None:
            ax.set_ylim(top=self._ymax)

        ax.grid()

    @staticmethod
    def _random_color(excluded_colors):
        excluded_colors = excluded_colors or []
        color_map = plt.get_cmap('gist_rainbow')
        if len(excluded_colors) == 0:
            color = color_map(np.random.uniform())
        else:
            color = excluded_colors[0]
            while color in excluded_colors:
                color = color_map(np.random.uniform())
        return color

