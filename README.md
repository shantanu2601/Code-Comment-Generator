# Code-Comment-Generator

Generating natural-language comments for source code with a Transformer
encoder-decoder, trained and evaluated on the CodeXGLUE code-to-text benchmark.

The project combines two ideas:

1. **A summarizer with relative positions and copy attention.** The encoder uses
   clipped relative position representations (Shaw et al., 2018) instead of, or in
   addition to, absolute sinusoidal positions. An extra attention layer on top of the
   decoder learns a copy distribution over the source tokens, so identifiers and other
   rare tokens can be copied straight from the code (Ahmad et al., 2020).
2. **Adversarial alignment of code and comment spaces.** A text autoencoder first
   learns a latent space for comments. A code encoder is then trained, against a
   discriminator, to map code into that same space. The aligned code encoder and the
   autoencoder's decoder can initialise the summarizer.

## Model

```
code tokens ──► Encoder (relative / absolute positions) ──► memory
                                                              │
comment prefix ──► Decoder ──► h ──┬─► vocabulary softmax ────┤
                                   └─► copy attention ────────┘
                                             │
              p(w) = p_gen · p_vocab(w) + (1 − p_gen) · Σ_{i: x_i = w} a_i
```

Code and comments share the CodeT5 subword vocabulary, so copied tokens map directly
onto vocabulary ids.

**Alignment.** The comment encoder from the autoencoder is frozen and acts as the
source of truth. The code encoder is the generator. A BiLSTM discriminator sees
encoder states and predicts whether they came from a comment (real) or from code
(fake). The code encoder's loss is the adversarial loss plus an MSE term between the
mean-pooled code states and the mean-pooled states of the paired comment.

## Layout

```
codecg/
  config.py              Model and training configuration
  data.py                CodeXGLUE loading, tokenization, batching
  metrics.py             Smoothed BLEU-4, ROUGE-L, METEOR
  train.py               CLI: autoencode, align, summarize, evaluate
  modules/
    attention.py         Multi-head attention with relative positions
    transformer.py       Encoder and decoder stacks
  models/
    summarizer.py        Summarizer with copy attention, greedy and beam search
    alignment.py         Text autoencoder, discriminator, adversarial aligner
scripts/
  run_experiments.sh     All runs for one language, then a results table
tests/
  test_models.py
```

## Setup

Dependencies are pinned to releases from October 2022 (PyTorch 1.12, Transformers
4.23, Datasets 2.6) and tested with Python 3.10. Training runs on CUDA if available,
otherwise on CPU.

```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"   # for METEOR
pytest tests
```

## Usage

Train and test a summarizer (Java by default, any CodeXGLUE language via `--language`):

```bash
python -m codecg.train summarize --positions relative --run-dir runs/rel+copy
```

With adversarial alignment:

```bash
python -m codecg.train autoencode --run-dir runs/autoencoder
python -m codecg.train align --text-autoencoder runs/autoencoder/text_autoencoder.pt --run-dir runs/align
python -m codecg.train summarize \
    --init-encoder runs/align/code_encoder.pt \
    --init-decoder runs/autoencoder/text_autoencoder.pt \
    --run-dir runs/rel+copy+align
```

Score an existing checkpoint:

```bash
python -m codecg.train evaluate --checkpoint runs/rel+copy/summarizer.pt --split test
```

Every run writes `log.jsonl`, the best checkpoint by validation loss, test
predictions and `scores.test.json` to its run directory. `--limit N` trains on the
first N examples of each split, which is useful for a quick check on CPU.

`scripts/run_experiments.sh java` runs the full comparison:

| Run | Encoder positions | Copy | Initialisation |
|---|---|---|---|
| abs | absolute | no | random |
| rel | relative | no | random |
| abs+rel | both | no | random |
| rel+copy | relative | yes | random |
| rel+copy+align-enc | relative | yes | aligned code encoder |
| rel+copy+align | relative | yes | aligned code encoder + autoencoder decoder |

## Defaults

6 layers, 8 heads, d_model 512, d_ff 2048, dropout 0.2, relative clipping
distance k = 16, AdamW with learning rate 1e-4 and 2000 warmup steps, batch size 32,
50 epochs, beam size 4. Inputs are truncated to 256 code and 64 comment subwords.

## Evaluation

BLEU is smoothed sentence-level BLEU-4 averaged over the test set, as in the
CodeXGLUE code-to-text evaluator. ROUGE-L is the LCS F-measure with β = 1.2.
METEOR uses NLTK and is skipped if WordNet is not installed. All scores are
reported on a 0-100 scale.

## References

- Shaw, Uszkoreit, Vaswani. *Self-Attention with Relative Position Representations.* NAACL 2018.
- Ahmad, Chakraborty, Ray, Chang. *A Transformer-based Approach for Source Code Summarization.* ACL 2020.
- See, Liu, Manning. *Get To The Point: Summarization with Pointer-Generator Networks.* ACL 2017.
- Lu et al. *CodeXGLUE: A Machine Learning Benchmark Dataset for Code Understanding and Generation.* NeurIPS 2021 Datasets and Benchmarks.
- Wang et al. *CodeT5: Identifier-aware Unified Pre-trained Encoder-Decoder Models for Code Understanding and Generation.* EMNLP 2021.
