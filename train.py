import os
import pickle
import torch
import model.gpt
import numpy as np
import torch.nn as nn
import torch.optim as optim


device = 'mps'
batch_size = 1 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 256 
data_dir = os.path.join('data')
meta_path = os.path.join(os.path.join("data"), 'meta.pkl')
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

def encode(s):
    return [stoi[c] for c in s] # encoder: take a string, output a list of integers
def decode(l):
    return ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

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


loss_fn = nn.CrossEntropyLoss() 

model = model.gpt.GPT(n_heads=6, d_ff=1536, d_model=384, query_key_dim=64, values_dim=64, embedding_dim=384, drop_out=0.2, N=6, vocab_size=meta_vocab_size)
model.to(device=device)

# training
optimizer = optim.AdamW(params=model.parameters(), lr=1e-4)  
for i in range(8000):
    X, Y = get_batch("train")
    print(X.device, Y.device) 
    y_predict = model(X.flatten())
    loss = loss_fn(y_predict.view(-1, meta_vocab_size), Y.view(-1)) 
    print(f'Iteration: {i}, Loss {loss}')
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("Training done")
# Save the model

torch.save(model.state_dict(), '/Users/alexandervivasjernstrom/Desktop/machinelearning/transformer-implementation/model.pt')

val_loss = 0
# validation
for i in range(500):
    X, Y = get_batch('test')
    y_predict = model(X.flatten())
    val_loss += loss_fn(y_predict.view(-1, meta_vocab_size), Y.view(-1)) 
    print(f'Iteration: {i}, Loss {val_loss}')
print(f'Val loss: {val_loss / 500}')
""" test_predict = "First Citizen:\n "
test_tok = torch.tensor(encode(test_predict))
pred = model(test_tok)
pred_classes = pred.argmax(dim=1)
print(pred_classes)
predicted_tokens = []
for token_index in pred_classes:
    token = itos[token_index.item()]
    predicted_tokens.append(token)
print("".join(predicted_tokens))

 """#print(y_predict)
