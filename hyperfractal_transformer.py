import torch
import torch.nn as nn
import torch.nn.functional as F

class FractalEmbedding(nn.Module):
    """Multi-step residual embedding representing fractal depth."""
    def __init__(self, dim: int, depth: int):
        super().__init__()
        self.layers = nn.ModuleList(nn.Linear(dim, dim) for _ in range(depth))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for layer in self.layers:
            out = F.relu(layer(out)) + out
        return out

class HyperFractalTransformer(nn.Module):
    """Transformer encoder enhanced with fractal embeddings."""
    def __init__(self, vocab_size: int, dim: int = 512, depth: int = 4, heads: int = 8):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.fractal_emb = FractalEmbedding(dim, depth)
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.fc_out = nn.Linear(dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.token_emb(x)
        fractal = self.fractal_emb(embedded)
        encoded = self.encoder(fractal)
        logits = self.fc_out(encoded)
        return logits
