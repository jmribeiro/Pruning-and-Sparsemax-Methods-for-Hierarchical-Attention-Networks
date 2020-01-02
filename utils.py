import torch
import os
import pathlib
import numpy as np
from datasets import YelpDataset, YahooDataset, IMDBDataset, AmazonDataset
from models import HierarchicalAttentionNetwork, PrunedHierarchicalAttentionNetwork, LSTMClassifier, HierarchicalNetwork, HierarchicalSparsemaxAttentionNetwork


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

    loss_dtach = loss.detach()

    # There's a memory leak somewhere
    if "cuda" in model.device.type:
        torch.cuda.empty_cache()

    return loss_dtach


def predict(model, X):
    scores = model(X)
    predicted_labels = scores.argmax(dim=-1)
    return predicted_labels


def evaluate_batch(model, batch):
    X = batch.text.to(model.device)
    y = batch.label.to(model.device)
    model.eval()
    y_hat = predict(model, X)
    n_correct = (y == y_hat).sum().item()
    # There's a memory leak somewhere
    if "cuda" in model.device.type:
        torch.cuda.empty_cache()
    return n_correct


def evaluate(model, dataloader):
    n_correct = 0
    n_possible = 0

    for batch in dataloader:
        n_correct += evaluate_batch(model, batch)
        n_possible += float(batch.batch_size)

    return n_correct / n_possible

def load_dataset(opt):

    if not opt.quiet: print(f"*** Loading {opt.dataset} dataset{f' [small size / debug mode]' if opt.debug else ''} ***", end="", flush=True)

    if opt.dataset == "yelp": dataset = YelpDataset(embeddings_size=opt.embeddings_size, full=not opt.polarity, ngrams=opt.ngrams, debug=opt.debug, sample=opt.sample, dataset_size=opt.dataset_size)
    elif opt.dataset == "yahoo": dataset = YahooDataset(embeddings_size=opt.embeddings_size, ngrams=opt.ngrams, debug=opt.debug, sample=opt.sample, dataset_size=opt.dataset_size)
    elif opt.dataset == "imdb": dataset = IMDBDataset(embeddings_size=opt.embeddings_size, sample=opt.sample, dataset_size=opt.dataset_size)
    elif opt.dataset == "amazon": dataset = AmazonDataset(embeddings_size=opt.embeddings_size, full=not opt.polarity, ngrams=opt.ngrams, debug=opt.debug, sample=opt.sample, dataset_size=opt.dataset_size)
    else: dataset = None  # Unreachable code

    if not opt.quiet: print(f" (Done) [{len(dataset)} training samples]", flush=True)
    return dataset


def select_model(model_name, dataset, opt, device):

    if model_name == "han": model = HierarchicalAttentionNetwork(dataset.n_classes, dataset.n_words, dataset.word2vec, opt.layers, opt.hidden_sizes, opt.dropout, dataset.padding_value, dataset.end_of_sentence_value, device)
    elif model_name == "phan": model = PrunedHierarchicalAttentionNetwork(dataset.n_classes, dataset.n_words, opt.attention_threshold, dataset.word2vec, opt.layers, opt.hidden_sizes, opt.dropout, dataset.padding_value, dataset.end_of_sentence_value, device)
    elif model_name == "hsan": model = HierarchicalSparsemaxAttentionNetwork(dataset.n_classes, dataset.n_words, dataset.word2vec, opt.layers, opt.hidden_sizes, opt.dropout, dataset.padding_value, dataset.end_of_sentence_value, device)
    elif model_name == "lstm": model = LSTMClassifier(dataset.n_classes, dataset.n_words, dataset.word2vec, opt.layers, opt.hidden_sizes, opt.bidirectional, opt.dropout, dataset.padding_value, device)
    elif model_name == "hn": model = HierarchicalNetwork(dataset.n_classes, dataset.n_words, dataset.word2vec, opt.layers, opt.hidden_sizes, opt.dropout, dataset.padding_value, dataset.end_of_sentence_value, device)
    else: model = None  # Unreachable code

    return model


def load_npy_files(path):

    global losses, valid_accs, final_accuracy
    sub_path = os.path.dirname(os.path.realpath(__file__)) + path
    files_path = pathlib.Path(sub_path)
    files = [f for f in os.listdir(files_path) if (f.endswith(".npy") or f.endswith(".txt"))]
    for file in files:
        file_path= pathlib.Path(sub_path + "/"+file)

        if ".txt" in file:
            with open(file_path, 'r') as data:
                final_accuracy = data.read()
        else:
            data = np.load(file_path)
        if "losses" in file:
            losses = data
        if "accs" in file:
            valid_accs = data
    return losses, valid_accs,final_accuracy
