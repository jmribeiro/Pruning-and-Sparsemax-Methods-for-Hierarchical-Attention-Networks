from os import listdir
from os.path import isfile, join

import numpy as np

from Plot import Plot

# from utils import

if __name__ == '__main__':

    epochs = 3

    N = 2

    colors = {
        "han": "blue",
        "phan": "green",
        "hsan": "red"
        # "hn": "grey",
        # "lstm": "orange"
    }
    plotvacc = Plot("Validation Accurancy per model", "Epoch", "Val Accuracy", epochs, 1, colors=colors, confidence=0.99, ymin=0)
    plotloss = Plot("Loss per model", "Epoch", "Loss", epochs, 1, colors=colors, confidence=0.99,
                ymin=0)
    plotfinalacc = Plot ("Final Test Accurancy ", "Epoch", "Test Accuracy", 1, 1, colors=colors, confidence=0.99, ymin=0)
    for model in colors:
        directory = f"results/imdb/{model}"

        for file in listdir(directory):
            
            if isfile(join(directory, file)):
                if (".txt" in file ) and plotfinalacc.num_runs(model) < N:
                    with open(directory + "/" + file, 'r') as data:
                        final_accuracy = []
                        data = float(data.read())

                        final_accuracy.append(data)
                        run = np.array(final_accuracy)
                        plotfinalacc.add_run (model, run)
                else:
                    if "losses" in file and plotloss.num_runs(model) < N:
                        run = np.load(directory + "/" + file)
                        plotloss.add_run(model, run)
                    else:
                        if plotvacc.num_runs(model) < N:
                            run = np.load(directory + "/" + file)
                            plotvacc.add_run(model, run)
    plotloss.show()
    plotloss.savefig("results_loss_1.pdf")
    plotvacc.show()
    plotvacc.savefig("results_validationacc_1.pdf")
    plotfinalacc.showbar()
    plotfinalacc.savefigbar("results_finalacc_1.pdf")