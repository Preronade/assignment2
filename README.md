# A Multi-Level Attention Model for Evidence-Based Fact Checking

This repository contains the implementation and CPU-based execution tests of the Multi-Level Attention (MLA) model.

## Environment

- Python 3.10
- PyTorch 2.0.1 (CPU)
- Transformers 4.40.2
- NumPy 1.26.4

## Execution Evidence

### 1. Attention Module Test

Run:

```bash
python tests/test_attention.py
```

Verified:
- Scaled dot-product attention
- Attention weight normalization
- Multi-head attention
- Output tensor validation

![MLA Attention Execution Results](docs/images/attention-test.png)

### 2. Joint Verification Model Test

Run:

```bash
python tests/test_verification_joint.py
```

Verified:
- Original BERT-based verification model
- Evidence-selection classifier
- Joint loss calculation
- Sentence-level attention
- Word-level attention
- Claim-evidence aggregation attention
- Finite logits and loss

![MLA Joint Model Test](docs/images/joint-model-test.png)

### 3. Preprocessing Test

The original preprocessing scripts were executed using synthetic JSONL examples.

Verified:
- Sentence-selection preprocessing
- Claim-verification preprocessing
- TSV output generation
- Evidence-selection labels and padding

![MLA Preprocessing Test](docs/images/preprocessing-test.png)

## Limitations

These tests demonstrate successful component execution and forward-pass validation using synthetic inputs. They do not demonstrate full FEVER dataset training, trained-model accuracy, or benchmark evaluation.
