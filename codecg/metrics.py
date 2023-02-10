import math
import re
import sys
from collections import Counter
from typing import Dict, List, Optional


def tokenize(text: str) -> List[str]:
    # Lowercase and split punctuation off words, as CodeXGLUE's evaluator does
    text = re.sub(r"([^\w\s])", r" \1 ", text.lower())
    return text.split()


def _ngrams(tokens: List[str], n: int) -> Counter:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def sentence_bleu(reference: List[str], hypothesis: List[str], max_n: int = 4) -> float:
    """
        Smoothed sentence BLEU-4 (add-one smoothing for n > 1), the variant used by
        the CodeXGLUE code-to-text evaluator
    """
    log_bleu = 0.0
    for n in range(1, max_n + 1):
        ref_counts = _ngrams(reference, n)
        hyp_counts = _ngrams(hypothesis, n)
        correct = sum(min(c, ref_counts[g]) for g, c in hyp_counts.items())
        guess = max(len(hypothesis) - n + 1, 0)
        smooth = 1 if n > 1 else 0
        log_bleu += math.log(correct + smooth + sys.float_info.min) - math.log(guess + smooth + sys.float_info.min)

    log_bleu /= max_n
    brevity = min(0.0, 1 - (len(reference) + 1) / (len(hypothesis) + 1))
    return math.exp(log_bleu + brevity)


def _lcs(a: List[str], b: List[str]) -> int:
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    return prev[-1]


def rouge_l(reference: List[str], hypothesis: List[str], beta: float = 1.2) -> float:
    if not reference or not hypothesis:
        return 0.0
    lcs = _lcs(reference, hypothesis)
    if lcs == 0:
        return 0.0
    precision = lcs / len(hypothesis)
    recall = lcs / len(reference)
    return ((1 + beta**2) * precision * recall) / (recall + beta**2 * precision)


def meteor(reference: List[str], hypothesis: List[str]) -> Optional[float]:
    # NLTK's METEOR needs WordNet; skip the metric if it is not available
    try:
        from nltk.translate.meteor_score import meteor_score
        return meteor_score([reference], hypothesis)
    except (ImportError, LookupError):
        return None


def score(references: List[str], hypotheses: List[str]) -> Dict[str, float]:
    """
        Corpus scores as averages of sentence scores, scaled to 0-100
    """
    assert len(references) == len(hypotheses)
    refs = [tokenize(r) for r in references]
    hyps = [tokenize(h) for h in hypotheses]

    results = {
        "bleu": 100 * sum(sentence_bleu(r, h) for r, h in zip(refs, hyps)) / len(refs),
        "rouge_l": 100 * sum(rouge_l(r, h) for r, h in zip(refs, hyps)) / len(refs),
    }

    meteor_scores = [meteor(r, h) for r, h in zip(refs, hyps)]
    if all(m is not None for m in meteor_scores):
        results["meteor"] = 100 * sum(meteor_scores) / len(refs)

    return results
