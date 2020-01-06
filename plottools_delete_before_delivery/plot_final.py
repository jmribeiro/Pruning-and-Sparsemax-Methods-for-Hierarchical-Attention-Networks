from os.path import isfile, join
import argparse
import pathlib

import numpy as np

from plottools_delete_before_delivery.Plot import Plot
import pandas as pd
import os


# from utils import

def plot_help(final_accuracy, model, plotfinalacc, acurracy):
    final_accuracy.append(acurracy)
    run = np.array(final_accuracy)
    plotfinalacc.add_run(model, run)


def plot_csv(file, colors):
    os.chdir("../results")
    df = pd.read_csv(file)

    plotfinalacc = Plot(f"{dataset}-Final Test Accurancy ", "Epoch", "Test Accuracy", 1, 1, colors=colors,
                        confidence=0.99, ymin=0)

    for model in colors:
        for index, row in df.iterrows():
            final_accuracy = []
            if model in row[0]:
                plot_help(final_accuracy, model, plotfinalacc, row[1])
            if model == 'hn' and 'hierarchical network' in row[0]:
                plot_help(final_accuracy, model, plotfinalacc, row[1])
            if model == 'hsan' and 'sparsemax' in row[0]:
                plot_help(final_accuracy, model, plotfinalacc, row[1])
            if model == 'hpan' and 'pruned' in row[0]:
                plot_help(final_accuracy, model, plotfinalacc, row[1])
            if model == 'han' and 'attention network' in row[0]:
                plot_help(final_accuracy, model, plotfinalacc, row[1])

    plotfinalacc.savefigbar("final_test_accuracies.png")


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Main arguments
    parser.add_argument('dataset', choices=['imdb', 'yelp', 'yahoo', 'amazon'],
                        help="Which dataset to train the model on?")
    parser.add_argument('-epochs', type=int, default=3)
    parser.add_argument('-nruns', type=int, default=2)

    opt = parser.parse_args()

    dataset = opt.dataset
    epochs = opt.epochs
    N = opt.nruns

    colors = {
        "han": "blue",
        "hpan": "green",
        "hsan": "red",
        "hn": "grey",
        "lstm": "orange"
    }

    plot_csv('final_test_accuracies.csv', colors)

    plotvacc = Plot(f"{dataset}-Validation Accurancy per model", "Epoch", "Val Accuracy", epochs, 1, colors=colors,
                    confidence=0.99, ymin=0)
    plotloss = Plot(f"{dataset}-Loss per model", "Epoch", "Loss", epochs, 1, colors=colors, confidence=0.99,
                    ymin=0)
    plotfinalacc = Plot(f"{dataset}-Final Test Accurancy ", "Epoch", "Test Accuracy", 1, 1, colors=colors,
                        confidence=0.99, ymin=0)
    for model in colors:
        directory = f"results/{dataset}/{model}"
        pathlib.Path(directory).mkdir(parents=True, exist_ok=True)

        for file in os.listdir(directory):

            if isfile(join(directory, file)):
                if (".txt" in file) and plotfinalacc.num_runs(model) < N:
                    with open(directory + "/" + file, 'r') as data:
                        final_accuracy = []
                        data = float(data.read())

                        final_accuracy.append(data)
                        run = np.array(final_accuracy)
                        plotfinalacc.add_run(model, run)
                else:
                    if "losses" in file and plotloss.num_runs(model) < N:
                        run = np.load(directory + "/" + file)
                        plotloss.add_run(model, run)
                    else:
                        if plotvacc.num_runs(model) < N:
                            run = np.load(directory + "/" + file)
                            plotvacc.add_run(model, run)

    plot_directory = f"plots/{dataset}/"
    pathlib.Path(plot_directory).mkdir(parents=True, exist_ok=True)
    plotloss.show()
    plotloss.savefig(plot_directory + f"{dataset}results_loss.pdf")
    plotvacc.show()
    plotvacc.savefig(plot_directory + f"{dataset}results_validationacc.pdf")
    plotfinalacc.showbar()
    plotfinalacc.savefigbar(plot_directory + f"{dataset}results_finalacc.pdf")

    print("..")
    plot_csv('final_test_accuracies.csv', colors)