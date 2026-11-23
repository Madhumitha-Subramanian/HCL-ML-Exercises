Neural Machine Translation Using a Custom Transformer Architecture (English to French)
1. Project Overview
This project implements an English-to-French Neural Machine Translation (NMT) system built and trained strictly from scratch in PyTorch on Google Colab (NVIDIA Tesla T4 GPU).
Zero Pretrained Models or Weights: No pretrained checkpoints (T5, BART, MarianMT, mBART) or pretrained embeddings were used. All 19,677,824 parameters were initialized randomly using Xavier Uniform initialization.
Note on Terminology ("Training from Scratch" vs. "Fine-Tuning"): Because the project prohibits pretrained weights, this pipeline trains a randomly initialized Transformer architecture from scratch rather than fine-tuning an existing pretrained checkpoint.
2. Dataset & NLP Preprocessing Pipeline
Source Dataset: `Helsinki-NLP/opus_books` (`en-fr`, `127,085` raw pairs).
Multi-Stage Data Cleaning Audit (16,436 noisy rows removed — 12.93%):
Unicode `NFKC` normalization & whitespace cleanup while preserving French diacritics (`é, è, ê, à, ç, œ`) and punctuation.
Exact `(EN, FR)` and duplicate `EN` source deduplication (`1,242` rows removed).
Identical untranslated string filtering (`71` rows removed).
Swapped-language detection removing reversed French-to-English records (`6,570` rows removed).
Length (`3–60` words) and alignment ratio (`0.45–2.2`) filtering (`8,553` rows removed).
Final Clean Corpus (`110,649` pairs) — Leakage-Free 80/10/10 Split (`SEED=42`):
Training Set: `88,519` pairs (80.00%)
Validation Set: `11,065` pairs (10.00%)
Test Set: `11,065` pairs (10.00%)
Verified 0 overlapping sentences across all splits.
3. Custom Subword Tokenization
Trained a 16,000-token Byte-Level BPE tokenizer from scratch exclusively on the training split (`88,519` pairs).
Special tokens: `<pad>` (`0`), `<unk>` (`1`), `<bos>` (`2`), `<eos>` (`3`).
`MAX_SEQ_LEN = 96` (covers >99.5% of sequences) with dynamic per-batch padding.
4. Custom Transformer Architecture (From Scratch)
Embedding & Positional Encoding: `d_model = 256` scaled by `sqrt(d_model)` + deterministic sinusoidal positional encodings.
Encoder: 4 Pre-LayerNorm `EncoderLayer` blocks (`8` attention heads, `d_ff = 1024`, `GELU`, `dropout = 0.1`).
Decoder: 4 Pre-LayerNorm `DecoderLayer` blocks with masked causal self-attention, encoder-decoder cross-attention, and feed-forward sublayers.
Output Projection: Linear layer mapping `256 -> 16,000` vocabulary logits.
5. Training & Quantitative Evaluation Results
Optimizer & Schedule: `AdamW` (`peak_lr = 5e-4`, `betas = (0.9, 0.98)`, `weight_decay = 0.01`) with 8% linear warmup + cosine decay, gradient clipping (`1.0`), mixed precision (`fp16`), and label-smoothed cross-entropy loss (`0.1`, ignoring `<pad>`).
Best Checkpoint (Epoch 13 of 15, 32.54 min total):
Train Loss / Perplexity: `2.8585` / `17.44`
Validation Loss / Perplexity: `3.2813` / `26.61`
Full Unseen Test Set Loss / Perplexity (11,065 pairs): `3.2707` / `26.33`
Translation Quality Metrics (1,024 Unseen Test Sentences):
SacreBLEU (Greedy): `23.23` (`BP = 0.993`)
chrF (Greedy): `47.91`
Inference Improvement (200 Unseen Test Sentences):
Greedy (`k=1`): SacreBLEU `20.98` | chrF `46.68`
Beam Search (`k=4`, length penalty `0.7`): SacreBLEU `22.67` (+1.69) | chrF `48.12` (+1.44)
