import torch
import model.gpt
import os
import pickle
import numpy as np

data_dir = os.path.join('data')
meta_path = os.path.join(os.path.join("data"), 'meta.pkl')
block_size = 256
batch_size = 1
device = "mps"

# Specify the path to the model file
model_file = './model.pt'
model = model.gpt.GPT(n_heads=6, d_ff=1536, d_model=384, query_key_dim=64, values_dim=64, embedding_dim=384, drop_out=0.2, N=6, vocab_size=65)

# Load the model
model_state = torch.load(model_file)

# Use the loaded model for inference
# ...
meta_vocab_size = None
stoi = None 
itos = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    stoi = meta["stoi"]
    itos = meta["itos"]
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

def get_batch(split):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    
    x, y = x.to(device), y.to(device)
    return x, y

model.load_state_dict(model_state)
loss_fn = torch.nn.CrossEntropyLoss()
for param in model.parameters():
    print(param)
@torch.no_grad()
def evaluate(iterations=1000):
   total_loss = 0
   for i in range(iterations):
       X, Y = get_batch('test')
       predicted = model(X.flatten())
       loss = loss_fn(predicted.view(-1, meta_vocab_size), Y.view(-1))
       print(f'Iteration: {i}, Loss: {loss}')
       total_loss += loss
   print(f'Average loss: {total_loss / iterations}')

evaluate()