import torch

def posenc(d_model: int, dimension: int, position: int, T: int):
   if dimension % 2 == 0:
       return torch.sin(position / (T ** (dimension / d_model)))
   else:
       return torch.cos(position / (T ** ((dimension-1) / d_model)))


class AttentionHead(torch.nn.Module):
    def __init__(self, d_model: int, query_key_dim: int, values_dim: int):
        super(AttentionHead, self).__init__()
        self.query_weights = torch.nn.Parameter(torch.randn(d_model, query_key_dim))
        self.key_weights = torch.nn.Parameter(torch.randn(d_model, query_key_dim))
        self.value_weights = torch.nn.Parameter(torch.randn(d_model, values_dim))
    
    def attention(keys: torch.Tensor, query: torch.Tensor, values: torch.Tensor):
        # attention(Q, V, K) = softargmax(QK^T / sqrt(D^QK)) * V
        query_key_product = query.dot(keys.transpose()) 
        scale = torch.sqrt(keys.size(dim=1))
        return torch.matmul(torch.softmax(query_key_product / scale), values)
    
    def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor):
        # calculate products between weights and corresponding query, key and value 
        query_product = torch.matmul(queries, self.query_weights) 
        keys_product = torch.matmul(keys, self.key_weights) 
        values_product = torch.matmul(values, self.value_weights) 
        # apply attention
        return self.attention(keys_product, query_product, values_product)


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, n_heads: int, d_model: int, query_key_dim: int, values_dim: int):
       super(MultiHeadAttention, self).__init__()
       self.n_heads = n_heads
       self.attention_head = AttentionHead(d_model, query_key_dim, values_dim)
       self.weights = torch.nn.Parameter(torch.randn(n_heads * values_dim, d_model))

    def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor):
        heads  = [self.attention_head(queries, keys, values) for x in range(self.n_heads)]
        concatenated_heads = torch.cat(heads)
        return torch.matmul(concatenated_heads, self.weights)



# d_ff refers to the dimensionality of the inner layer
class FFN(torch.nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.relu = torch.nn.ReLU()
        self.fc1 = torch.nn.Linear(in_features=d_model, out_features=d_ff)
        self.fc2 = torch.nn.Linear(in_features=d_ff, out_features=d_model)

    def forward(self, x): 
        x = self.fc1(x)
        x = self.relu(x) 
        return self.fc2(x)

class Encoder(torch.nn.Module):
    def __init__(self, d_model: int, n_heads: int, query_key_dim: int, values_dim: int, d_ff: int, embedding_dim: int):
        super().__init__()
        self.mha = MultiHeadAttention(d_model=d_model, n_heads=n_heads, query_key_dim=query_key_dim, values_dim=values_dim)
        self.ffn = FFN(d_model=d_model, d_ff=d_ff)
        self.layer_norm = torch.nn.LayerNorm(embedding_dim)

    def forward(self, x):
        multi_attention = self.mha(x, x, x)
        norm1 = self.layer_norm(multi_attention + x)
        # Feedforward
        ff = self.ffn(norm1) 
        return self.layer_norm(ff + norm1) 

class Decoder(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    def forward(self, x):
        pass

class Transformer(torch.nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, d_model: int, d_ff: int, n_heads: int, N: int, query_key_dim, values_dim: int):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.encoder = Encoder(d_model=d_model, d_ff=d_ff, embedding_dim=embedding_dim, n_heads=n_heads, query_key_dim=query_key_dim, values_dim=values_dim)

    def forward(self, x):
        pass

