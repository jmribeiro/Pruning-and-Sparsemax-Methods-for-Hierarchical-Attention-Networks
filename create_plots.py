import argparse
import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np

from random import getrandbits

# ############ #
# Presentation #
# ############ #


def load_npy_files(path):

    global losses, valid_accs, final_accuracy
    sub_path = os.path.dirname(os.path.realpath(__file__)) + path
    files_path = pathlib.Path(sub_path)
    files = [f for f in os.listdir(files_path) if (f.endswith(".npy") or f.endswith(".txt"))]
    losses = []
    valid_accs = []
    final_acc = []

    for file in files:
        file_path= pathlib.Path(sub_path + "/"+file)

        if ".txt" in file:
            with open(file_path, 'r') as data:
                final_accuracy = data.read()
                final_acc.append(final_accuracy)
        else:
            data = np.load(file_path)
        if "losses" in file:
            losses.append(data)
        if "accs" in file:
            valid_accs.append(data)
    return losses, valid_accs,final_acc


def plotfile(run, plottable, ylabel, title, name):
    plt.clf()
    plt.xlabel('Run')
    plt.ylabel(ylabel)
    plt.xticks(run)
    plt.plot(run, plottable)
    plt.title(title)
    plt.savefig('%s.pdf' % name, bbox_inches='tight')
    plt.close()


def plotfiles(plottable, ylabel, title, name):
    plt.clf()
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)

    for plotitem  in plottable:
        nitem = len(plotitem)
        nitems = np.arange(1, nitem + 1)
        plt.xticks(nitems)
        # plt.plot(torch.arange(1, nitem + 1), plotitem)
        plt.plot(nitems, plotitem)
    plt.ylabel (ylabel)
    plt.title(title)
    plt.savefig('%s.pdf' % name, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Main arguments
    parser.add_argument('model', choices=['han', 'phan', 'hsan', 'lstm', 'hn'],
                        help="Which model should the script run?")
    parser.add_argument('dataset', choices=['yelp', 'yahoo', 'imdb', 'amazon'],
                        help="Which dataset to train the model on?")

     # Model Parameters

    parser.add_argument('-epochs', type=int, default=20)
    parser.add_argument('-optimizer', choices=['sgd', 'adam'], default='adam')
    parser.add_argument('-learning_rate', type=float, default=0.001)
    parser.add_argument('-l2_decay', type=float, default=0.0)
    parser.add_argument('-batch_size', type=int, default=64)
    parser.add_argument('-cuda', action='store_true', help='Use cuda for parallelization if devices available')

    # Miscellaneous

    parser.add_argument('-debug', action='store_true', help="Datasets pruned into smaller sizes for faster loading.")
    parser.add_argument('-quiet', action='store_true', help='No execution output.')
    parser.add_argument('-no_plot', action='store_true', help='Whether or not to plot training losses and validation accuracies.')

    opt = parser.parse_args()

    path = f"/results/{opt.dataset}/{opt.model}"
    model = opt.model

    if not opt.quiet: print(f"*** loading results ***", end="", flush=True)

    train_mean_losses, valid_accs, final_test_accuracy = load_npy_files(path)

    if not opt.quiet: print(f"*** Plotting validation accuracies and training losses ***", end="", flush=True)


    try:
        os.mkdir("plots")
    except FileExistsError:
        pass

    fileid = getrandbits(64)
    plotfiles(train_mean_losses, ylabel='Loss', title = f"{opt.dataset}-{model}-training-loss",
            name=f"plots/{fileid}-{opt.dataset}-{model}-training-loss")

    plotfiles(valid_accs, ylabel='Accuracy', title = f"{opt.dataset}-{model}-validation-accuracy",
             name=f"plots/{fileid}-{opt.dataset}-{model}-validation-accuracy")

    nfiles = len(final_test_accuracy )
    plotfile(np.arange(1, nfiles + 1), final_test_accuracy , ylabel='Final Accuracy',
             title=f"{opt.dataset}-{model}-final-accuracy", name=f"plots/{fileid}-{opt.dataset}-{model}-final-accuracy")

    if not opt.quiet: print(" (Done)\n", flush=True)
