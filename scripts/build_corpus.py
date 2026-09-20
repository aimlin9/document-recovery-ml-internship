import re
import urllib.request
import random
import json

GUTENBERG_BOOKS = {
    "train_val_pool": [
        (1342, "Pride and Prejudice"),
        (84, "Frankenstein"),
        (1661, "The Adventures of Sherlock Holmes"),
        (2701, "Moby-Dick"),
    ],
    "held_out_test": [
        (11, "Alice's Adventures in Wonderland"),
        (43, "The Strange Case of Dr Jekyll and Mr Hyde"),
    ],
}


def download_book(gutenberg_id):
    url = f"https://www.gutenberg.org/ebooks/{gutenberg_id}.txt.utf-8"
    with urllib.request.urlopen(url) as response:
        return response.read().decode("utf-8")


def strip_gutenberg_boilerplate(text):
    """Gutenberg texts have a standard header/footer with licensing
    text, not part of the actual book. We cut everything outside the
    START/END markers Gutenberg inserts into every file."""
    start_match = re.search(r"\*\*\* START OF THE PROJECT GUTENBERG.*?\*\*\*", text, re.DOTALL)
    end_match = re.search(r"\*\*\* END OF THE PROJECT GUTENBERG.*?\*\*\*", text, re.DOTALL)
    start = start_match.end() if start_match else 0
    end = end_match.start() if end_match else len(text)
    return text[start:end]


def split_into_sentences(text):
    """Simple regex-based sentence splitter, good enough for prose.
    Not perfect (struggles with abbreviations like 'Mr.'), but fine
    for our purposes since we filter aggressively afterward anyway."""
    text = re.sub(r"\s+", " ", text)  # collapse newlines/whitespace
    raw_sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    return [s.strip() for s in raw_sentences]


def is_usable_sentence(sentence, min_words=6, max_words=40):
    words = sentence.split()
    if not (min_words <= len(words) <= max_words):
        return False
    if re.search(r"[0-9_@#\\/<>]", sentence):
        return False
    if sentence.isupper():
        return False
    # Reject sentences containing standalone all-caps words (headings,
    # chapter markers like "CHAPTER", "VI.") that leaked past the
    # whole-sentence uppercase check.
    if any(w.isupper() and len(re.sub(r"[^A-Za-z]", "", w)) >= 3 for w in words):
        return False
    return True


def build_sentence_pool(book_list):
    all_sentences = []
    for gutenberg_id, title in book_list:
        print(f"Downloading: {title} (ID {gutenberg_id})...")
        raw = download_book(gutenberg_id)
        body = strip_gutenberg_boilerplate(raw)
        sentences = split_into_sentences(body)
        usable = [s for s in sentences if is_usable_sentence(s)]
        print(f"  {len(sentences)} raw sentences, {len(usable)} usable")
        for s in usable:
            all_sentences.append({"sentence": s, "source": title})
    return all_sentences


if __name__ == "__main__":
    random.seed(42)

    train_val_pool = build_sentence_pool(GUTENBERG_BOOKS["train_val_pool"])
    test_pool = build_sentence_pool(GUTENBERG_BOOKS["held_out_test"])

    random.shuffle(train_val_pool)

    # Cap at practical target sizes from the plan.
    n_train = min(8000, len(train_val_pool) - 1000)
    n_val = min(1000, len(train_val_pool) - n_train)
    n_test = min(1000, len(test_pool))

    train_set = train_val_pool[:n_train]
    val_set = train_val_pool[n_train:n_train + n_val]
    test_set = test_pool[:n_test]

    for record in train_set:
        record["split"] = "train"
    for record in val_set:
        record["split"] = "validation"
    for record in test_set:
        record["split"] = "test"

    all_records = train_set + val_set + test_set

    with open("sentence_corpus.jsonl", "w", encoding="utf-8") as f:
        for record in all_records:
            f.write(json.dumps(record) + "\n")

    print(f"\nFinal corpus sizes:")
    print(f"  train: {len(train_set)}")
    print(f"  validation: {len(val_set)}")
    test_sources = sorted(set(r["source"] for r in test_set))
    print(f"  test (held out, from {test_sources}): {len(test_set)}")
    print(f"\nSaved to sentence_corpus.jsonl")