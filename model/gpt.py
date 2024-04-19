import torch.nn as nn
import torch
import math
import os 
import pickle
device ="mps" 

def encode(s, stoi):
    return [stoi[c] for c in s] # encoder: take a string, output a list of integers
def decode(l, itos):
    return ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string


def posenc(d_model: int, dimension: int, position: int, T: int):
   if dimension % 2 == 0:
       return math.sin(position / (T ** (dimension / d_model)))
   else:
       return math.cos(position / (T ** ((dimension-1) / d_model)))
class FFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, embedding_dim: int, drop_out: float):
        super().__init__()
        self.feed_forward_stack = nn.Sequential(
            nn.LayerNorm(embedding_dim, device=device),
            nn.Linear(in_features=d_model, out_features=d_ff, device=device),
            nn.GELU(),
            nn.Linear(in_features=d_ff, out_features=d_model, device=device),
            nn.Dropout(p=drop_out)
        )

    def forward(self, x): 
       ffn = self.feed_forward_stack(x) 
       return ffn + x


class AttentionHead(nn.Module):
    def __init__(self, d_model: int, query_key_dim: int, values_dim: int):
        super(AttentionHead, self).__init__()
        self.query_weights = nn.Parameter(torch.randn(d_model, query_key_dim, device=device))
        self.key_weights = nn.Parameter(torch.randn(d_model, query_key_dim, device=device))
        self.value_weights = nn.Parameter(torch.randn(d_model, values_dim, device=device))
    
    def attention(self, keys: torch.Tensor, query: torch.Tensor, values: torch.Tensor):
        # attention(Q, V, K) = softargmax(QK^T / sqrt(D^QK)) * V
        query_key_product = query.matmul(keys.transpose(0, 1)) 
        scale = math.sqrt(keys.size(dim=1))
        return torch.matmul(torch.softmax(query_key_product / scale, 0), values)
    
    def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor):
        # calculate products between weights and corresponding query, key and value 
        query_product = torch.matmul(queries, self.query_weights) 
        keys_product = torch.matmul(keys, self.key_weights) 
        values_product = torch.matmul(values, self.value_weights) 
        # apply attention
        return self.attention(keys_product, query_product, values_product)


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads: int, d_model: int, query_key_dim: int, values_dim: int):
       super(MultiHeadAttention, self).__init__()
       self.n_heads = n_heads
       self.attention_head = AttentionHead(d_model, query_key_dim, values_dim)
       self.weights = nn.Parameter(torch.randn(n_heads * values_dim, d_model, device=device))

    def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor):
        heads  = [self.attention_head(queries, keys, values) for x in range(self.n_heads)]
        concatenated_heads = torch.cat(heads, dim=1)
        return torch.matmul(concatenated_heads, self.weights)

class SelfAttention(nn.Module):
    def __init__(self, n_heads: int, d_model: int, query_key_dim: int, values_dim: int, embedding_dim: int):
        super().__init__()
        self.mha = MultiHeadAttention(n_heads=n_heads, d_model=d_model, query_key_dim=query_key_dim, values_dim=values_dim)
        self.norm = nn.LayerNorm(embedding_dim, device=device)
    
    def forward(self, x):
        normed = self.norm(x)
        multi_attention = self.mha(normed, normed, normed)
        return multi_attention + x

class Decoder(nn.Module):
    def __init__(self, n_heads: int, d_ff: int, d_model: int, query_key_dim: int, values_dim: int, embedding_dim: int, drop_out: float):
        super().__init__()
        self.self_attention = SelfAttention(n_heads=n_heads, d_model=d_model, query_key_dim=query_key_dim, values_dim=values_dim, embedding_dim=embedding_dim)
        self.ffn = FFN(embedding_dim=embedding_dim, d_ff=d_ff, d_model=d_model, drop_out=drop_out)

    def forward(self, x):
        x = self.self_attention(x)
        x = self.ffn(x)
        return x
class GPT(nn.Module):
    def __init__(self, n_heads: int, d_ff: int, d_model: int, query_key_dim: int, values_dim: int, embedding_dim: int, drop_out: float, N: int, vocab_size: int):
        super().__init__()
        self.decoders = [Decoder(n_heads=n_heads, d_ff=d_ff, d_model=d_model, drop_out=drop_out, embedding_dim=embedding_dim, query_key_dim=query_key_dim, values_dim=values_dim) for i in range(N)]
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, device=device)
        self.fc = nn.Linear(embedding_dim, vocab_size, device=device)
        self.d_model = d_model
        self.embedding_dim = embedding_dim
    
    def forward(self, x):
        embedded = self.embedding(x)
        positional_encoding = torch.tensor([[posenc(self.d_model, self.embedding_dim, i, 10000) for i in range(self.embedding_dim)]], device=device)
        embedded = positional_encoding + embedded 
        decoded = embedded
        for decoder in self.decoders:
            decoded = decoder(decoded)
        return self.fc(decoded)
    
    def generate(self, prompt: str, stoi, itos):
        test_tok = torch.tensor(encode(prompt, stoi), device=device)
        pred = self.forward(test_tok)
        pred_classes = pred.argmax(dim=1)
        predicted_tokens = []
        for token_index in pred_classes:
            token = itos[token_index.item()]
            predicted_tokens.append(token)
        return predicted_tokens 