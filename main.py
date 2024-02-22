import torch

class SingleHeadAttention(torch.nn.Module):
    def __init__(self, queries: torch.Tensor, values: torch.Tensor, keys: torch.Tensor, d_model: int):
        super(SingleHeadAttention, self).__init__()
        self.linear_value = torch.nn.Linear(values.size(dim=1), d_model)
        self.linear_key = torch.nn.Linear(keys.size(dim=1), d_model)
        self.linear_query = torch.nn.Linear(queries.size(dim=1), d_model)
    
    def attention(keys: torch.Tensor, query: torch.Tensor, values: torch.Tensor):
        # attention(Q, V, K) = softargmax(QK^T / sqrt(D^QK)) * V
        query_key_product = query.dot(keys.transpose()) 
        scale = torch.sqrt(keys.size(dim=1))
        return torch.matmul(torch.softmax(query_key_product / scale), values)

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, n_heads, queries, values, keys):
       super(MultiHeadAttention, self).__init__()
       self.n_heads = n_heads

        
    
    def forward(self, x):
        for i in self.n_heads: 
            pass
        pass
