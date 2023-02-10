import pytest
import torch

from codecg import metrics
from codecg.config import ModelConfig
from codecg.models import AdversarialAligner, Summarizer, TextAutoencoder
from codecg.modules import Encoder, MultiHeadAttention

PAD, BOS, EOS = 0, 1, 2


def tiny_config(**overrides) -> ModelConfig:
    base = dict(vocab_size=50, d_model=32, n_heads=4, d_ff=64, n_layers=2, dropout=0.0, max_relative_position=4)
    base.update(overrides)
    return ModelConfig(**base)


def random_batch(batch_size=3, src_len=10, tgt_len=6, vocab=50):
    src = torch.randint(3, vocab, (batch_size, src_len))
    tgt = torch.randint(3, vocab, (batch_size, tgt_len))
    tgt[:, 0], tgt[:, -1] = BOS, EOS
    src[0, -3:] = PAD  # Some padding
    return src, tgt


def test_relative_indices_are_clipped():
    attn = MultiHeadAttention(d_model=8, n_heads=2, max_relative_position=2)
    rel = attn.relative_indices(4, 6, torch.device("cpu"))
    assert rel.shape == (4, 6)
    assert rel.min() == 0 and rel.max() == 4
    # Diagonal is distance 0, which maps to index k
    assert torch.all(rel.diagonal() == 2)


@pytest.mark.parametrize("positions", ["absolute", "relative", "both"])
def test_encoder_shapes(positions):
    cfg = tiny_config(encoder_positions=positions)
    src, _ = random_batch()
    out = Encoder(cfg)(src)
    assert out.shape == (src.size(0), src.size(1), cfg.d_model)


def test_relative_positions_are_used():
    torch.manual_seed(0)
    attn = MultiHeadAttention(d_model=16, n_heads=2, max_relative_position=3).eval()
    x = torch.randn(1, 6, 16)
    with_rel, _ = attn(x, x, x)

    torch.nn.init.zeros_(attn.rel_k.weight)
    torch.nn.init.zeros_(attn.rel_v.weight)
    without_rel, _ = attn(x, x, x)
    assert not torch.allclose(with_rel, without_rel)


def test_relative_encoder_is_order_aware():
    # Without absolute positions, order information can only come from the
    # relative embeddings, so reversing the input must change the states
    torch.manual_seed(0)
    enc = Encoder(tiny_config(encoder_positions="relative", n_layers=1)).eval()
    seq = torch.randint(3, 50, (1, 6))
    out = enc(seq)
    out_reversed = enc(seq.flip(1)).flip(1)
    assert not torch.allclose(out, out_reversed)


@pytest.mark.parametrize("copy", [True, False])
def test_summarizer_outputs_distribution(copy):
    cfg = tiny_config(copy_attention=copy)
    model = Summarizer(cfg)
    src, tgt = random_batch()
    log_probs = model.log_probs(tgt[:, :-1], model.encode(src), src)
    assert log_probs.shape == (src.size(0), tgt.size(1) - 1, cfg.vocab_size)
    assert torch.allclose(log_probs.exp().sum(-1), torch.ones(log_probs.shape[:2]), atol=1e-4)


def test_copy_mass_lands_on_source_tokens():
    cfg = tiny_config(copy_attention=True)
    model = Summarizer(cfg)
    # Force the gate fully towards copying
    torch.nn.init.zeros_(model.copy_gate.weight)
    torch.nn.init.constant_(model.copy_gate.bias, -50.0)

    src, tgt = random_batch()
    probs = model.log_probs(tgt[:, :-1], model.encode(src), src).exp()
    for b in range(src.size(0)):
        source_ids = set(src[b][src[b].ne(PAD)].tolist())
        mass = probs[b][:, sorted(source_ids)].sum(-1)
        assert torch.allclose(mass, torch.ones_like(mass), atol=1e-4)


def test_summarizer_overfits_tiny_batch():
    torch.manual_seed(0)
    cfg = tiny_config(copy_attention=True)
    model = Summarizer(cfg)
    src, tgt = random_batch(batch_size=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

    first = model(src, tgt).item()
    for _ in range(300):
        loss = model(src, tgt)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    assert loss.item() < 0.1 * first

    model.eval()
    expected = [t[1:-1].tolist() for t in tgt]
    assert model.greedy(src, max_len=10) == expected
    assert model.beam_search(src, max_len=10, beam_size=3) == expected


def test_alignment_updates_only_code_side():
    cfg = tiny_config()
    text_ae = TextAutoencoder(cfg)
    aligner = AdversarialAligner(text_ae.encoder, Encoder(cfg), cfg).train()
    src, tgt = random_batch()

    g = aligner.generator_loss(src, tgt)
    g["loss"].backward()
    assert all(p.grad is None for p in aligner.text_encoder.parameters())
    assert any(p.grad is not None for p in aligner.code_encoder.parameters())

    aligner.zero_grad(set_to_none=True)  # torch < 2.0 zeroes grads by default instead
    d = aligner.discriminator_loss(src, tgt)
    d["loss"].backward()
    assert all(p.grad is None for p in aligner.code_encoder.parameters())
    assert any(p.grad is not None for p in aligner.discriminator.parameters())


def test_text_autoencoder_loss():
    cfg = tiny_config()
    _, tgt = random_batch()
    loss = TextAutoencoder(cfg)(tgt)
    assert loss.item() > 0


def test_metrics_perfect_and_disjoint():
    refs = ["returns the sum of two numbers", "opens a file"]
    perfect = metrics.score(refs, refs)
    assert perfect["bleu"] == pytest.approx(100.0)
    assert perfect["rouge_l"] == pytest.approx(100.0)

    disjoint = metrics.score(refs, ["zzz qqq", "xxx"])
    assert disjoint["bleu"] < 1.0
    assert disjoint["rouge_l"] == 0.0


def test_tokenize_splits_punctuation():
    assert metrics.tokenize("Returns x.y()") == ["returns", "x", ".", "y", "(", ")"]
