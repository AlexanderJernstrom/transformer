import torch

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
    
    def forward(self, x: torch.Tensor):
        # x = [X_query, X_keys, X_values]
        # forward linear layer
        queries = x[0] * self.query_weights
        keys = x[1] * self.key_weights
        values = x[2] * self.value_weights
        # apply attention
        attention = self.attention(keys, queries, values)
        return attention 


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, n_heads: int):
       super(MultiHeadAttention, self).__init__()
       self.n_heads = n_heads

    def forward(self, x):
        for i in self.n_heads: 
            pass
        pass

