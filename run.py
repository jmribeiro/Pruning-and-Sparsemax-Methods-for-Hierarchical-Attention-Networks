import argparse
import os

import matplotlib.pyplot as plt

import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from datasets import OCRDataset, Yelp13Dataset, Yelp14Dataset, Yelp15Dataset, YahooDataset, IMDBDataset, AmazonDataset
from models import HierarchicalAttentionNetwork, PrunedHierarchicalAttentionNetwork, FeedforwardNetwork


# #################### #
# Classification Utils #
# #################### #

def train_batch(X, y, model, optimizer, criterion):

    X = X.to(model.device)
    y = y.to(model.device)

    optimizer.zero_grad()
    model.train()

    y_hat = model(X)
    loss = criterion(y_hat, y)

    loss.backward()
    optimizer.step()

    return loss


def predict(model, X):
    scores = model(X)
    predicted_labels = scores.argmax(dim=-1)
    return predicted_labels


def evaluate(model, X, y):

    X = X.to(model.device)
    y = y.to(model.device)

    model.eval()

    y_hat = predict(model, X)
    n_correct = (y == y_hat).sum().item()
    n_possible = float(y.shape[0])

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


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('model', choices=[
        'hierarchical_attention_network',
        'pruned_hierarchical_attention_network',
        'hierarchical_sparsemax_attention_network',
        'mlp'
    ], help="Which model should the script run?")

    parser.add_argument('-dataset', help="Name of the dataset.", default='ocr')
    parser.add_argument('-resources', help="Path to datasets folder.", default='resources')

    # Model Parameters
    parser.add_argument('-layers', type=int, help="Number of layers", default=1)
    parser.add_argument('-hidden_sizes', type=int, help="Number of units per hidden layer", default=100)
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
    parser.add_argument('-quiet', action='store_true', help='No execution output.')
    parser.add_argument('-tqdm', action='store_true', help='Whether or not to use TQDM progress bar in training.')
    parser.add_argument('-save_plot', action='store_true', help='Whether or not to plot validation losses and accuracies.')

    opt = parser.parse_args()

    # ############# #
    # 1 - Load Data #
    # ############# #

    if opt.dataset == "ocr":       dataset = OCRDataset(opt.resources+"/letter.data")
    elif opt.dataset == "yelp13":  dataset = Yelp13Dataset(opt.resources)
    elif opt.dataset == "yelp14":  dataset = Yelp14Dataset(opt.resources)
    elif opt.dataset == "yelp15":  dataset = Yelp15Dataset(opt.resources)
    elif opt.dataset == "yahoo":   dataset = YahooDataset(opt.resources)
    elif opt.dataset == "imdb":    dataset = IMDBDataset(opt.resources)
    elif opt.dataset == "amazon":  dataset = AmazonDataset(opt.resources)
    else: raise ValueError(f"Unknown dataset {opt.dataset}")

    X_dev, y_dev = dataset.X_dev, dataset.y_dev
    X_test, y_test = dataset.X_test, dataset.y_test

    # ################ #
    # 2 - Create Model #
    # ################ #

    device = torch.device("cuda:0" if torch.cuda.is_available() and opt.cuda else "cpu")
    if not opt.quiet: print(f"Using device '{device}'", flush=True)

    if opt.model == "hierarchical_attention_network": model = HierarchicalAttentionNetwork(device)
    elif opt.model == "pruned_hierarchical_attention_network": model = PrunedHierarchicalAttentionNetwork(device)
    elif opt.model == "hierarchical_sparsemax_attention_network": model = PrunedHierarchicalAttentionNetwork(device)
    elif opt.model == "mlp": model = FeedforwardNetwork(dataset.n_classes, dataset.n_features, opt.hidden_sizes, opt.layers, opt.activation, opt.dropout, device)

    # ############# #
    # 3 - Optimizer #
    # ############# #

    optimizer = {
        "adam": torch.optim.Adam,
        "sgd": torch.optim.SGD
    }[opt.optimizer](
        model.parameters(),
        lr=opt.learning_rate,
        weight_decay=opt.l2_decay
    )
    criterion = nn.CrossEntropyLoss()

    # ###################### #
    # 4 - Train and Evaluate #
    # ###################### #

    epochs = torch.arange(1, opt.epochs + 1)
    train_mean_losses = []
    valid_accs = []
    train_losses = []
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True)

    for epoch in epochs:
        if not opt.quiet: print('\nTraining epoch {}'.format(epoch), flush=True)
        progress_bar = tqdm(dataloader) if opt.tqdm else dataloader
        for X_batch, y_batch in progress_bar:
            loss = train_batch(X_batch, y_batch, model, optimizer, criterion)
            train_losses.append(loss)

        mean_loss = torch.tensor(train_losses).mean().item()
        if not opt.quiet: print('Training loss: %.4f' % mean_loss, flush=True)

        train_mean_losses.append(mean_loss)
        valid_accs.append(evaluate(model, X_dev, y_dev))
        if not opt.quiet: print('Valid acc: %.4f' % (valid_accs[-1]), flush=True)

    final_test_accuracy = evaluate(model, X_test, y_test)
    if not opt.quiet: print('\nFinal Test acc: %.4f' % final_test_accuracy, flush=True)

    # ######## #
    # 4 - Plot #
    # ######## #

    if opt.save_plot:
        try: os.mkdir(opt.plot_dir)
        except FileExistsError: pass
        plot(epochs, train_mean_losses, ylabel='Loss', name=f"{opt.plot_dir}/{opt.model}-training-loss")
        plot(epochs, valid_accs, ylabel='Accuracy', name=f"{opt.plot_dir}/{opt.model}-validation-accuracy")

    return final_test_accuracy


if __name__ == '__main__':
    main()
