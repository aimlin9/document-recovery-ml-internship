import json
import random
import re

FUNCTION_WORDS = {
    "a", "an", "the", "and", "or", "but", "if", "of", "to", "in", "on", "at",
    "by", "for", "with", "about", "against", "between", "into", "through",
    "during", "before", "after", "above", "below", "from", "up", "down",
    "is", "are", "was", "were", "be", "been", "being", "am",
    "he", "she", "it", "they", "we", "you", "i", "him", "her", "them", "us",
    "his", "hers", "its", "their", "our", "your", "my",
    "this", "that", "these", "those",
    "not", "no", "nor", "so", "as", "than", "then", "there", "here",
    "do", "does", "did", "have", "has", "had", "will", "would", "shall",
    "should", "can", "could", "may", "might", "must",
}


def is_content_word(word):
    stripped = re.sub(r"[^a-zA-Z]", "", word).lower()
    return stripped not in FUNCTION_WORDS and stripped != ""


def pick_two_distinct_indices(word_count, rng):
    """Picks two different word positions to mask, one at a time,
    so a sentence never gets the same word masked twice."""
    if word_count < 2:
        return None
    indices = rng.sample(range(word_count), k=min(2, word_count))
    return indices


def build_masked_examples(sentence_record, sentence_id, rng):
    words = sentence_record["sentence"].split()
    indices = pick_two_distinct_indices(len(words), rng)
    if indices is None:
        return []

    examples = []
    for word_index in indices:
        masked_words = [
            "<mask>" if i == word_index else w
            for i, w in enumerate(words)
        ]
        target_word = words[word_index]
        examples.append({
            "example_id": f"{sentence_id}_{word_index}",
            "source": sentence_record["source"],
            "split": sentence_record["split"],
            "original_sentence": sentence_record["sentence"],
            "masked_sentence": " ".join(masked_words),
            "target_word": target_word,
            "word_index": word_index,
            "is_content_word": is_content_word(target_word),
        })
    return examples


if __name__ == "__main__":
    rng = random.Random(42)

    all_examples = []
    with open("sentence_corpus.jsonl", "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            record = json.loads(line)
            examples = build_masked_examples(record, sentence_id=i, rng=rng)
            all_examples.extend(examples)

    with open("masked_examples.jsonl", "w", encoding="utf-8") as f:
        for ex in all_examples:
            f.write(json.dumps(ex) + "\n")

    by_split = {}
    for ex in all_examples:
        by_split.setdefault(ex["split"], []).append(ex)

    print("Masked example counts by split:")
    for split, exs in by_split.items():
        content_count = sum(1 for e in exs if e["is_content_word"])
        print(f"  {split}: {len(exs)} examples ({content_count} content-word, {len(exs) - content_count} function-word)")

    print(f"\nSaved to masked_examples.jsonl")