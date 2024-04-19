# GPT implementation

Implementation of the original Generative Pretrained Transformer for text generation proposed in _Improving Language Understanding
by Generative Pre-Training_ by Radford et al in 2018.

## Code

The model code is 100% written by me however training related code, see train.py, main.py, prepare.py and inference.py comes from Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) repo.

## Results

Kind of doesn't work. However is probably only a matter of scaling up the training with more iterations and investing more time into making the training process a bit better. It is trained on the tiny Shakespeare dataset.

## Features for the future

Batching, batching, batching...
Compute, compute, compute ...

## Try it out yourself

1. Install dependencies with `pip install -r requirements.txt`
2. Setup virtualenv
3. Get the training data by running `cd data && python3 prepare.py`
4. When that is done, set the `device` variable to the desired device in `train.py`, `model.py` and `inference.py`
5. Run `python3 train.py` to train the model, should go relatively fast
6. Run inference on the validation set with `python3 inference.py`
