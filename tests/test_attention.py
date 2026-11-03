import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from types import SimpleNamespace
from attentions import attention, MultiHeadedAttention

torch.manual_seed(42)

print("=== MLA SCALED DOT-PRODUCT ATTENTION ===")

query = torch.randn(2, 3, 4)
key = torch.randn(2, 5, 4)
value = torch.randn(2, 5, 6)

output, weights = attention(query, key, value)

assert output.shape == (2, 3, 6)
assert weights.shape == (2, 3, 5)
assert torch.isfinite(output).all()
assert torch.allclose(
    weights.sum(dim=-1),
    torch.ones(2, 3),
    atol=1e-6
)

print("Output shape:", tuple(output.shape))
print("Weights shape:", tuple(weights.shape))
print("Attention weights sum to 1: PASS")

print("\n=== MLA MULTI-HEAD ATTENTION ===")

config = SimpleNamespace(
    hidden_size=8,
    num_attention_heads=2,
    attention_probs_dropout_prob=0.1
)

model = MultiHeadedAttention(config)
model.eval()

query = torch.randn(2, 3, 8)
key = torch.randn(2, 5, 8)
value = torch.randn(2, 5, 8)

with torch.no_grad():
    output = model(query, key, value)

assert output.shape == (2, 3, 8)
assert model.attn.shape == (2, 2, 3, 5)
assert torch.isfinite(output).all()

print("Number of heads:", model.h)
print("Output shape:", tuple(output.shape))
print("Attention weights shape:", tuple(model.attn.shape))
print("Finite output: PASS")

print("\nALL MLA ATTENTION TESTS PASSED")
