# Recovering Structure From Damaged or Poorly Scanned Documents

ML internship project — Learn Depth Academy LLP, Track 2 (Advanced ML Internship), Problem ID ML-T2-022.

**Author:** Gyimah Ramsey Opoku (Student ID LD-1787404180111), KNUST, Kumasi, Ghana

## Project Overview

This project builds a pipeline that recovers text from documents with torn or missing regions, combining OCR, synthetic damage simulation, and a fine-tuned language model to reconstruct damaged words, with confidence scoring to flag uncertain reconstructions for human review.

## Pipeline Stages

1. **Stage 1 — Research Brief:** problem framing and approach (docs/Stage1_Research_Brief.pdf)
2. **Stage 2 — Development Brief:** technical plan for the 5 build steps (docs/Stage2_Development_Brief.pdf)
3. **Step 1 — Baseline OCR:** Tesseract OCR performance under synthetic degradation (docs/Step1_Baseline_Results.pdf)
4. **Step 2 — Damage Masking:** torn/missing-region mask generation and per-word damage classification (docs/Step2_Results_Note.pdf)
5. **Step 3 — Reconstruction Baseline:** pretrained RoBERTa masked-word reconstruction (docs/Step3_Results_Note.pdf)
6. **Step 4 — Fine-Tuning + Uncertainty Tagging:** fine-tuned RoBERTa, overfitting fix, and confidence/calibration layer for flagging unreliable reconstructions (docs/Step4_Results_Note.pdf, docs/Step4_Uncertainty_Tagging_Results_Note.pdf)
7. **Step 5 — End-to-End Integration:** full pipeline stitching OCR, damage detection, reconstruction, and confidence scoring into one schema-validated JSON output per document, evaluated on the full 739-sentence held-out test set (docs/Step5_Integration_Results_Note.pdf)

## Repository Structure

- `schema/` — JSON Schema integration contract (`document_schema.json`)
- `scripts/` — all pipeline code (OCR, masking, reconstruction, stitching, batch evaluation)
- `data/` — sentence corpus, masked examples, held-out manifest
- `results/` — held-out evaluation outputs (739 per-document JSON files + aggregate summary report)
- `docs/` — all results notes (PDF), one per stage/step

## Fine-Tuned Model

The fine-tuned RoBERTa model (`roberta-finetuned-final/`) is too large for GitHub and is hosted separately on Google Drive:

https://drive.google.com/drive/folders/1gbYFTWuWN3otPWE4aoP2oH5NIM9QhaHm?usp=sharing

To reproduce Step 5, download the model folder from that link and place it at the project root as `roberta-finetuned-final/` before running `scripts/build_document_json.py` or `scripts/run_held_out_evaluation.py`.

## Setup


pip install -r requirements.txt


Requires Tesseract OCR installed separately (path configured in `scripts/build_document_json.py`).

Note: `held_out_temp_images/` and other generated `.png` files are intermediate artifacts created automatically when the evaluation scripts run (deterministically, from the corpus and a fixed seed). They are not tracked in this repo and do not need to be downloaded separately.

## Key Results (Step 5, full held-out set, n=739)

- Corpus-level CER: OCR-only 0.0492 -> final stitched 0.0300
- Corpus-level WER: OCR-only 0.0560 -> final stitched 0.0383
- Reconstruction Top-1 accuracy: 40.32% overall, 28.02% on content words
- At the locked confidence threshold (0.5442): 65.76% error recall, 84.30% precision, unflagged error rate 38.23%

See `docs/Step5_Integration_Results_Note.pdf` for full methodology, the JSON schema contract, and discussion of the generalization gap between the Step 4 text-only calibration and the Step 5 real-image distribution.
