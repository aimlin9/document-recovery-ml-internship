import json, hashlib

with open("sentence_corpus.jsonl", "rb") as f:
    raw = f.read()
print("sentence_corpus.jsonl sha256:", hashlib.sha256(raw).hexdigest())

records = [json.loads(line) for line in raw.decode("utf-8").splitlines()]
test_records = [r for r in records if r["split"] == "test"]
print("Test sentences in this file, right now:", len(test_records))