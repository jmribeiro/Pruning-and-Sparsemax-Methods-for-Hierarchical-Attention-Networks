import argparse
import os

import matplotlib.pyplot as plt

import torch
from torch import nn
from torchtext.data import BucketIterator
from tqdm import tqdm

from datasets import YelpDataset, YahooDataset, IMDBDataset, AmazonDataset
from models import HierarchicalAttentionNetwork, PrunedHierarchicalAttentionNetwork, LSTMClassifier


# #################### #
# Classification Utils #
# #################### #

def train_batch(batch, model, optimizer, criterion):

    X = batch.text.to(model.device)
    y = batch.label.to(model.device)

    optimizer.zero_grad()
    model.train()

    y_hat = model(X)
    loss = criterion(y_hat, y)

    loss.backward()
    optimizer.step()

    return loss.detach()


def predict(model, X):
    scores = model(X)
    predicted_labels = scores.argmax(dim=-1)
    return predicted_labels


def evaluate(model, dataloader):

    n_correct = 0
    n_possible = 0

    for batch in dataloader:
        X = batch.text.to(model.device)
        y = batch.label.to(model.device)
        model.eval()
        y_hat = predict(model, X)
        n_correct += (y == y_hat).sum().item()
        n_possible += float(y.shape[0])

    return n_correct / n_possible


# ############ #
# Presentation #
# ############ #

def plot(epochs, plottable, ylabel, name):
    plt.clf()
    plt.xlabel('Epoch')
    plt.ylabel(ylabel)
    plt.plot(epochs, plottable)
    plt.title(name)
    plt.savefig('%s.pdf' % name, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Main arguments
    parser.add_argument('model', choices=['han', 'phan', 'hsan', 'lstm'], help="Which model should the script run?")
    parser.add_argument('dataset', choices=['yelp', 'yahoo', 'imdb', 'amazon'], help="Which dataset to train the model on?")

    # Model Parameters
    parser.add_argument('-embeddings_size', type=int, help="Length of the word embeddings.", default=200)
    parser.add_argument('-layers', type=int, help="Number of layers", default=1)
    parser.add_argument('-hidden_sizes', type=int, help="Number of units per hidden layer", default=50)
    parser.add_argument('-bidirectional', action="store_true")
    parser.add_argument('-activation', help="Activation function", default="relu")
    parser.add_argument('-dropout', type=float, help="Dropout probability", default=0.1)

    # Optimization Parameters
    parser.add_argument('-epochs', type=int, default=20)
    parser.add_argument('-optimizer', choices=['sgd', 'adam'], default='adam')
    parser.add_argument('-learning_rate', type=float, default=0.001)
    parser.add_argument('-l2_decay', type=float, default=0.0)
    parser.add_argument('-batch_size', type=int, default=64)
    parser.add_argument('-cuda', action='store_true', help='Use cuda for parallelization if devices available')

    # Miscellaneous
    parser.add_argument('-polarity', action='store_true', help="Positive/Negative labels for datasets.")
    parser.add_argument('-debug', action='store_true', help="Datasets pruned into smaller sizes for faster loading.")
    parser.add_argument('-quiet', action='store_true', help='No execution output.')
    parser.add_argument('-tqdm', action='store_true', help='Whether or not to use TQDM progress bar in training.')
    parser.add_argument('-no_plot', action='store_true', help='Whether or not to plot training losses and validation accuracies.')

    opt = parser.parse_args()

    if not opt.quiet:
        print("# ################################### #", flush=True)
        print("#    Pruning and Sparsemax Methods    #", flush=True)
        print("# for Hierarchical Attention Networks #", flush=True)
        print("# ################################### #\n", flush=True)

    # ############# #
    # 1 - Load Data #
    # ############# #

    if not opt.quiet:
        print(f"*** Loading {opt.dataset} dataset{f' [small size / debug mode]' if opt.debug else ''} ***", end="", flush=True)

    if opt.dataset == "yelp": dataset = YelpDataset(full=not opt.polarity, debug=opt.debug)
    elif opt.dataset == "yahoo": dataset = YahooDataset(debug=opt.debug)
    elif opt.dataset == "imdb": dataset = IMDBDataset()
    elif opt.dataset == "amazon": dataset = AmazonDataset(full=not opt.polarity, debug=opt.debug)
    else: dataset = None  # Unreachable code

    trainloader, valloader, testloader = BucketIterator.splits((dataset.training, dataset.validation, dataset.test), batch_size=opt.batch_size, sort_key=dataset.sort_key)

    if not opt.quiet: print(f" (Done) [{len(dataset)} training samples]", flush=True)

    # ################ #
    # 2 - Create Model #
    # ################ #

    device = torch.device("cuda:0" if torch.cuda.is_available() and opt.cuda else "cpu")

    if not opt.quiet: print(f"*** Setting up {opt.model} model on device {device} ***", end="", flush=True)

    if opt.model == "han": model = HierarchicalAttentionNetwork(dataset.n_classes, dataset.n_words, opt.embeddings_size, opt.layers, opt.hidden_sizes, opt.dropout, dataset.padding_value, dataset.end_of_sentence_value, device) # FIXME - Proper arguments when done
    elif opt.model == "phan": model = PrunedHierarchicalAttentionNetwork(device) # FIXME - Proper arguments when done
    elif opt.model == "hsan": model = PrunedHierarchicalAttentionNetwork(device) # FIXME - Proper arguments when done
    elif opt.model == "lstm": model = LSTMClassifier(dataset.n_classes, dataset.n_words, opt.embeddings_size, opt.layers, opt.hidden_sizes, opt.bidirectional, opt.dropout, dataset.padding_value, device)
    else: dataset = None  # Unreachable code

    if not opt.quiet: print(" (Done)", flush=True)

    # ############# #
    # 3 - Optimizer #
    # ############# #

    if not opt.quiet: print(f"*** Setting up {opt.optimizer} optimizer ***", end="", flush=True)

    optimizer = {
        "adam": torch.optim.Adam,
        "sgd": torch.optim.SGD
    }[opt.optimizer](
        model.parameters(),
        lr=opt.learning_rate,
        weight_decay=opt.l2_decay
    )
    criterion = nn.CrossEntropyLoss()

    if not opt.quiet: print(" (Done)\n", flush=True)

    # ###################### #
    # 4 - Train and Evaluate #
    # ###################### #

    if not opt.quiet:
        print(f"# ######## #", flush=True)
        print(f"# Training #", flush=True)
        print(f"# ######## #", flush=True)

    epochs = torch.arange(1, opt.epochs + 1)
    train_mean_losses = []
    valid_accs = []
    train_losses = []

    for epoch in epochs:

        if not opt.quiet: print('\nTraining epoch {}'.format(epoch), flush=True)

        for batch in tqdm(trainloader) if opt.tqdm else trainloader:
            loss = train_batch(batch, model, optimizer, criterion)
            train_losses.append(loss)

        mean_loss = torch.tensor(train_losses).mean().item()
        if not opt.quiet: print('Training loss: %.4f' % mean_loss, flush=True)

        train_mean_losses.append(mean_loss)
        valid_accs.append(evaluate(model, valloader))
        if not opt.quiet: print('Valid acc: %.4f' % (valid_accs[-1]), flush=True)

    final_test_accuracy = evaluate(model, testloader)
    if not opt.quiet: print('\nFinal Test acc: %.4f' % final_test_accuracy, flush=True)

    # ######## #
    # 4 - Plot #
    # ######## #

    if not opt.no_plot:

        if not opt.quiet: print(f"*** Plotting validation accuracies and training losses ***", end="", flush=True)
        try: os.mkdir(opt.plot_dir)
        except FileExistsError: pass

        plot(epochs, train_mean_losses, ylabel='Loss', name=f"{opt.plot_dir}/{opt.model}-training-loss")
        plot(epochs, valid_accs, ylabel='Accuracy', name=f"{opt.plot_dir}/{opt.model}-validation-accuracy")

        if not opt.quiet: print(" (Done)\n", flush=True)
