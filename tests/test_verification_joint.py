import sys
import types
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import AutoTokenizer
# Add the repository root to Python's import path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
# Supply only the logging helper if PyTorch Lightning is unavailable.
try:
    from pytorch_lightning.utilities import rank_zero_info
except ModuleNotFoundError:
    pl = types.ModuleType("pytorch_lightning")
    utilities = types.ModuleType("pytorch_lightning.utilities")
    utilities.rank_zero_info = lambda *args, **kwargs: None
    pl.utilities = utilities
    sys.modules["pytorch_lightning"] = pl
    sys.modules["pytorch_lightning.utilities"] = utilities

from modeling_verification import VerificationJointModel


MODEL_PATH = Path(
    "/home/prerona/.cache/huggingface/hub/"
    "models--bert-base-uncased/snapshots/"
    "86b5e0934494bd15c9632b12f734a8a67f723594"
)

NUM_EVIDENCE = 2
MAX_LENGTH = 32


def run_test(tokenizer, sent_attn, word_attn):
    hparams = SimpleNamespace(
        pretrained_model_name=str(MODEL_PATH),
        num_evidence=NUM_EVIDENCE,
        max_seq_length=MAX_LENGTH,
        aggregate_mode="attn",
        attn_bias_type="none",
        classifier_dropout_prob=0.1,
        lambda_joint=1.0,
        sent_attn=sent_attn,
        word_attn=word_attn,
    )

    model = VerificationJointModel(hparams, num_labels=3)
    model.eval()

    texts = [
        "New Delhi is the capital of India.",
        "New Delhi is the capital of India.",
        "India is a country in South Asia.",
    ]

    encoded = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    input_ids = encoded["input_ids"].unsqueeze(0)
    attention_mask = encoded["attention_mask"].unsqueeze(0)

    labels = torch.tensor([0])
    selection_labels = torch.tensor([[0, 1, 0]])

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            selection_labels=selection_labels,
            return_dict=True,
        )

    assert outputs.logits.shape == (1, 3)
    assert torch.isfinite(outputs.logits).all()
    assert torch.isfinite(outputs.loss)

    assert (model.sent_attn is not None) == sent_attn
    assert (model.word_attn is not None) == word_attn

    print(f"\nSentence attention: {sent_attn}")
    print(f"Word attention: {word_attn}")
    print("Input shape:", tuple(input_ids.shape))
    print("Logits shape:", tuple(outputs.logits.shape))
    print("Joint loss:", round(outputs.loss.item(), 4))
    print("Finite logits and loss: PASS")

    del model


def main():
    if not MODEL_PATH.is_dir():
        raise FileNotFoundError(
            f"Local BERT model not found: {MODEL_PATH}"
        )

    print("Loading local BERT tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_PATH),
        local_files_only=True,
    )

    print("\n=== MLA JOINT MODEL TESTS ===")

    run_test(tokenizer, sent_attn=False, word_attn=False)
    run_test(tokenizer, sent_attn=True, word_attn=True)

    print("\nALL JOINT MODEL TESTS PASSED")


if __name__ == "__main__":
    main()
