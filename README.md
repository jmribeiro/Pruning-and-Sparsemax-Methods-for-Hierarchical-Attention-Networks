# Pruning and Sparsemax Methods for Hierarchical Attention Networks
##### Deep Structured Learning Course, Computer Science PhD Program, Instituto Superior Técnico

#### To install dependencies:
    
1) Install Pytorch (pick config at will):

    https://pytorch.org/get-started/locally/
    
2) Install latest Pytorch NLP version:

    `$ pip install git+https://github.com/PetrochukM/PyTorch-NLP.git`

3) Install remaining requirements:

    `$ pip install -r requirements.txt`
    
    `$ python -m spacy download en`
    
#### To run:

    $ python run.py model dataset
    
e.g. - Hierarchical Attention Network on IMDB dataset

    $ python run.py han imdb

##### Available models - han, hpan, hsan, lstm, hn
##### Available datasets - imdb, yelp, yahoo, amazon

### FAQ

1) _I'm getting the following error_:

        RuntimeError: "view size is not compatible with input tensor's size and stride 
        (at least one dimension spans across two contiguous subspaces). 
        Use .reshape(...) instead."
        
    Solution: Update pytorch-nlp to its latest version:
        
        $ pip uninstall pytorch-nlp
        $ pip install git+https://github.com/PetrochukM/PyTorch-NLP.git
