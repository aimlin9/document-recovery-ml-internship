import json, hashlib

import paths

with open(paths.CORPUS_PATH, "rb") as f:
    raw = f.read()
print("sentence_corpus.jsonl sha256:", hashlib.sha256(raw).hexdigest())

records = [json.loads(line) for line in raw.decode("utf-8").splitlines()]
test_records = [r for r in records if r["split"] == "test"]
print("Test sentences in this file, right now:", len(test_records))
