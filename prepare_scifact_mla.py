
import csv
import json
import random
from collections import defaultdict, Counter
from pathlib import Path

ROOT = Path("data/scifact/extracted/data")
OUT = Path("data/scifact/mla_small")
SEED = 42
NUM_EVIDENCE = 5

LABEL_MAP = {
    "SUPPORT": "S",
    "CONTRADICT": "R",
    "NOT ENOUGH INFO": "N",
}

random.seed(SEED)


def clean_text(value):
    """Remove tabs, newlines, and repeated whitespace."""
    return " ".join(str(value).split())


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def get_label(record):
    evidence = record.get("evidence", {})
    labels = {
        item["label"].upper()
        for groups in evidence.values()
        for item in groups
    }

    if not labels:
        return "NOT ENOUGH INFO"

    if len(labels) != 1:
        raise ValueError(
            f"Multiple evidence labels for claim {record['id']}: {labels}"
        )

    label = next(iter(labels))
    if label not in LABEL_MAP:
        raise ValueError(f"Unknown label: {label}")

    return label


def stratified_groups(records):
    groups = defaultdict(list)

    for record in records:
        groups[get_label(record)].append(record)

    for label in LABEL_MAP:
        random.shuffle(groups[label])

    return groups


def load_corpus():
    corpus = {}

    for record in read_jsonl(ROOT / "corpus.jsonl"):
        corpus[str(record["doc_id"])] = record

    return corpus


def make_rows(record, corpus):
    claim_id = record["id"]
    claim = clean_text(record["claim"])
    label = LABEL_MAP[get_label(record)]

    evidence = record.get("evidence", {})
    gold = []

    for doc_id, groups in evidence.items():
        for group in groups:
            for sent_id in group["sentences"]:
                gold.append((str(doc_id), int(sent_id)))

    gold = list(dict.fromkeys(gold))[:NUM_EVIDENCE]
    gold_set = set(gold)

    doc_ids = list(dict.fromkeys(
        [str(x) for x in record.get("cited_doc_ids", [])]
        + [doc_id for doc_id, _ in gold]
    ))

    negatives = []

    for doc_id in doc_ids:
        doc = corpus.get(doc_id)

        if doc is None:
            continue

        for sent_id, sentence in enumerate(doc.get("abstract", [])):
            key = (doc_id, sent_id)

            if key not in gold_set:
                negatives.append(key)

    candidates = gold + negatives[:max(0, NUM_EVIDENCE - len(gold))]

    # Claim-only row
    rows = [
        [claim_id, claim, "[PAD]", -1, "[PAD]", label, 0]
    ]

    # Evidence rows
    for doc_id, sent_id in candidates:
        doc = corpus[doc_id]

        sentence = clean_text(doc["abstract"][sent_id])
        title = clean_text(doc.get("title", "[NO TITLE]"))

        selection_label = int((doc_id, sent_id) in gold_set)

        rows.append([
            claim_id,
            claim,
            title,
            sent_id,
            sentence,
            label,
            selection_label,
        ])

    # Pad to exactly NUM_EVIDENCE evidence rows
    while len(rows) < NUM_EVIDENCE + 1:
        rows.append([
            claim_id,
            claim,
            "[PAD]",
            -1,
            "[PAD]",
            label,
            0,
        ])

    if len(rows) != NUM_EVIDENCE + 1:
        raise AssertionError(f"Wrong row count for claim {claim_id}")

    # Validate every field before writing
    for row in rows:
        if len(row) != 7:
            raise AssertionError(f"Invalid TSV row: {row}")

        for value in row:
            if "\t" in str(value) or "\n" in str(value) or "\r" in str(value):
                raise AssertionError(f"Unclean TSV field: {value!r}")

    return rows


def write_split(name, records, corpus):
    path = OUT / f"{name}.tsv"

    selection_counts = Counter()
    class_counts = Counter()

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(
            f,
            delimiter="\t",
            lineterminator="\n",
            quoting=csv.QUOTE_MINIMAL,
        )

        for record in records:
            rows = make_rows(record, corpus)
            writer.writerows(rows)

            class_counts[get_label(record)] += 1

            for row in rows[1:]:
                selection_counts[row[6]] += 1

    print(f"\n{name.upper()}: {len(records)} claims")
    print("Class distribution:", dict(class_counts))
    print("Evidence selection labels:", dict(selection_counts))
    print("TSV:", path)


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    train = read_jsonl(ROOT / "claims_train.jsonl")
    dev = read_jsonl(ROOT / "claims_dev.jsonl")
    corpus = load_corpus()

    train_groups = stratified_groups(train)
    dev_groups = stratified_groups(dev)

    train_selected = []

    for label in LABEL_MAP:
        train_selected.extend(train_groups[label][:30])

    validation = []
    test = []

    for label in LABEL_MAP:
        selected = dev_groups[label][:20]
        validation.extend(selected[:10])
        test.extend(selected[10:])

    train_ids = {x["id"] for x in train_selected}
    validation_ids = {x["id"] for x in validation}
    test_ids = {x["id"] for x in test}

    assert not train_ids.intersection(validation_ids)
    assert not train_ids.intersection(test_ids)
    assert not validation_ids.intersection(test_ids)

    write_split("train", train_selected, corpus)
    write_split("dev", validation, corpus)
    write_split("test", test, corpus)

    print("\nDATASET PREPARATION PASSED")
    print(
        "Total claims:",
        len(train_selected) + len(validation) + len(test),
    )
    print("Evidence slots per claim:", NUM_EVIDENCE)
    print("Rows per claim:", NUM_EVIDENCE + 1)
    print("Corpus documents available:", len(corpus))
    print("Output directory:", OUT)


if __name__ == "__main__":
    main()
