"""
    Training entry point. Three stages, each usable on its own:

        autoencode  Learn a comment latent space with a text autoencoder
        align       Adversarially align a code encoder to that latent space
        summarize   Train the code summarizer, optionally starting from the
                    aligned code encoder and the autoencoder's decoder

    and `evaluate` to score a trained summarizer with BLEU, METEOR and ROUGE-L
"""
import argparse
import json
import os
import random
import time
from dataclasses import replace
from typing import Callable, Dict, Optional

import torch
import torch.nn as nn

from codecg import metrics
from codecg.config import LANGUAGES, PATH_RUNS, ModelConfig, TrainConfig
from codecg.data import CodeCommentDataset, load_tokenizer, make_loader, model_config_for
from codecg.models import AdversarialAligner, Summarizer, TextAutoencoder
from codecg.modules import Encoder


def get_device() -> torch.device:
    # MPS is left out on purpose, its backend is still a prototype in torch 1.12
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


class Logger:
    def __init__(self, run_dir: str) -> None:
        os.makedirs(run_dir, exist_ok=True)
        self.path = os.path.join(run_dir, "log.jsonl")

    def __call__(self, **record) -> None:
        record["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
        print(" ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in record.items()), flush=True)
        with open(self.path, "a") as f:
            f.write(json.dumps(record) + "\n")


def save_checkpoint(path: str, model_cfg: ModelConfig, train_cfg: TrainConfig, **state_dicts) -> None:
    torch.save({"model_cfg": model_cfg.to_dict(), "train_cfg": train_cfg.to_dict(), **state_dicts}, path)


def load_checkpoint(path: str) -> Dict:
    return torch.load(path, map_location="cpu")


def warmup_lambda(warmup_steps: int) -> Callable[[int], float]:
    return lambda step: min(1.0, (step + 1) / max(1, warmup_steps))


@torch.no_grad()
def mean_loss(model: nn.Module, loader, step: Callable, device: torch.device) -> float:
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        total += step(model, batch, device).item()
        n += 1
    model.train()
    return total / max(1, n)


def fit(model, loader_train, loader_val, step, train_cfg: TrainConfig, log: Logger, on_best) -> None:
    """
        Generic loop: AdamW with linear warmup, gradient clipping, and a callback
        whenever the validation loss improves
    """
    device = next(model.parameters()).device
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_lambda(train_cfg.warmup_steps))

    best = float("inf")
    global_step = 0
    model.train()

    for epoch in range(1, train_cfg.epochs + 1):
        for batch in loader_train:
            loss = step(model, batch, device)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, train_cfg.grad_clip)
            optimizer.step()
            scheduler.step()
            global_step += 1

            if global_step % 100 == 0:
                log(stage="train", epoch=epoch, step=global_step, loss=loss.item())

        val = mean_loss(model, loader_val, step, device)
        log(stage="val", epoch=epoch, step=global_step, loss=val)
        if val < best:
            best = val
            on_best()


def summarizer_step(model: Summarizer, batch, device) -> torch.Tensor:
    return model(batch["src"].to(device), batch["tgt"].to(device))


def autoencoder_step(model: TextAutoencoder, batch, device) -> torch.Tensor:
    return model(batch["tgt"].to(device))


def load_data(train_cfg: TrainConfig, splits=("train", "validation", "test")):
    tokenizer = load_tokenizer()
    data = {split: CodeCommentDataset(split, tokenizer, train_cfg) for split in splits}
    return tokenizer, data


# Stages


def run_autoencode(args, model_cfg_overrides: Dict, train_cfg: TrainConfig) -> None:
    device = get_device()
    log = Logger(args.run_dir)
    tokenizer, data = load_data(train_cfg, ("train", "validation"))
    model_cfg = model_config_for(tokenizer, **model_cfg_overrides, copy_attention=False)

    model = TextAutoencoder(model_cfg).to(device)
    path = os.path.join(args.run_dir, "text_autoencoder.pt")

    fit(
        model,
        make_loader(data["train"], train_cfg, model_cfg.pad_id, shuffle=True),
        make_loader(data["validation"], train_cfg, model_cfg.pad_id, shuffle=False),
        autoencoder_step,
        train_cfg,
        log,
        on_best=lambda: save_checkpoint(path, model_cfg, train_cfg, model=model.state_dict()),
    )


def run_align(args, train_cfg: TrainConfig) -> None:
    device = get_device()
    log = Logger(args.run_dir)

    ckpt = load_checkpoint(args.text_autoencoder)
    model_cfg = ModelConfig.from_dict(ckpt["model_cfg"])

    text_ae = TextAutoencoder(model_cfg)
    text_ae.load_state_dict(ckpt["model"])

    aligner = AdversarialAligner(text_ae.encoder, Encoder(model_cfg), model_cfg, train_cfg.mse_weight).to(device)

    tokenizer, data = load_data(train_cfg, ("train", "validation"))
    loader_train = make_loader(data["train"], train_cfg, model_cfg.pad_id, shuffle=True)
    loader_val = make_loader(data["validation"], train_cfg, model_cfg.pad_id, shuffle=False)

    g_opt = torch.optim.AdamW(aligner.code_encoder.parameters(), lr=train_cfg.g_lr, weight_decay=train_cfg.weight_decay)
    d_opt = torch.optim.AdamW(aligner.discriminator.parameters(), lr=train_cfg.d_lr, weight_decay=train_cfg.weight_decay)

    path = os.path.join(args.run_dir, "code_encoder.pt")
    best = float("inf")
    global_step = 0
    aligner.train()

    for epoch in range(1, train_cfg.epochs + 1):
        for batch in loader_train:
            code, text = batch["src"].to(device), batch["tgt"].to(device)

            # Discriminator step
            d = aligner.discriminator_loss(code, text)
            d_opt.zero_grad()
            d["loss"].backward()
            d_opt.step()

            # Generator (code encoder) step
            g = aligner.generator_loss(code, text)
            g_opt.zero_grad()
            g["loss"].backward()
            nn.utils.clip_grad_norm_(aligner.code_encoder.parameters(), train_cfg.grad_clip)
            g_opt.step()

            global_step += 1
            if global_step % 100 == 0:
                log(stage="train", epoch=epoch, step=global_step, d_loss=d["loss"].item(),
                    g_adv=g["adv"].item(), g_mse=g["mse"].item())

        # Select on the MSE between pooled code and comment representations
        aligner.eval()
        with torch.no_grad():
            mses = [aligner.generator_loss(b["src"].to(device), b["tgt"].to(device))["mse"].item() for b in loader_val]
        aligner.train()

        val_mse = sum(mses) / max(1, len(mses))
        log(stage="val", epoch=epoch, step=global_step, mse=val_mse)
        if val_mse < best:
            best = val_mse
            save_checkpoint(path, model_cfg, train_cfg, code_encoder=aligner.code_encoder.state_dict())


def build_summarizer(args, model_cfg_overrides: Dict, tokenizer) -> Summarizer:
    copy_attention = not args.no_copy

    # Take the architecture from whichever pretrained part is given, so they fit
    source = args.init_encoder or args.init_decoder
    if source:
        model_cfg = ModelConfig.from_dict(load_checkpoint(source)["model_cfg"])
        model_cfg = replace(model_cfg, copy_attention=copy_attention)
    else:
        model_cfg = model_config_for(tokenizer, **model_cfg_overrides, copy_attention=copy_attention)

    model = Summarizer(model_cfg)

    if args.init_encoder:
        model.encoder.load_state_dict(load_checkpoint(args.init_encoder)["code_encoder"])
    if args.init_decoder:
        state = load_checkpoint(args.init_decoder)["model"]
        model.decoder.load_state_dict({k[len("decoder."):]: v for k, v in state.items() if k.startswith("decoder.")})
        model.generator.load_state_dict({k[len("generator."):]: v for k, v in state.items() if k.startswith("generator.")})

    return model


def run_summarize(args, model_cfg_overrides: Dict, train_cfg: TrainConfig) -> None:
    device = get_device()
    log = Logger(args.run_dir)
    tokenizer, data = load_data(train_cfg)

    model = build_summarizer(args, model_cfg_overrides, tokenizer).to(device)
    pad_id = model.cfg.pad_id
    path = os.path.join(args.run_dir, "summarizer.pt")

    fit(
        model,
        make_loader(data["train"], train_cfg, pad_id, shuffle=True),
        make_loader(data["validation"], train_cfg, pad_id, shuffle=False),
        summarizer_step,
        train_cfg,
        log,
        on_best=lambda: save_checkpoint(path, model.cfg, train_cfg, model=model.state_dict()),
    )

    # Score the best checkpoint on the test set
    model.load_state_dict(load_checkpoint(path)["model"])
    evaluate(model, data["test"], tokenizer, train_cfg, args.run_dir, "test", log)


def evaluate(model: Summarizer, dataset: CodeCommentDataset, tokenizer, train_cfg: TrainConfig,
             run_dir: str, split: str, log: Optional[Logger] = None) -> Dict[str, float]:
    device = next(model.parameters()).device
    model.eval()

    hypotheses, references = [], []
    for batch in make_loader(dataset, train_cfg, model.cfg.pad_id, shuffle=False):
        outputs = model.beam_search(batch["src"].to(device), train_cfg.max_decode_len, train_cfg.beam_size)
        hypotheses += tokenizer.batch_decode(outputs, skip_special_tokens=True)
        references += [dataset.comments[i] for i in batch["idx"].tolist()]

    scores = metrics.score(references, hypotheses)

    with open(os.path.join(run_dir, f"predictions.{split}.txt"), "w") as f:
        f.writelines(h.replace("\n", " ") + "\n" for h in hypotheses)
    with open(os.path.join(run_dir, f"scores.{split}.json"), "w") as f:
        json.dump(scores, f, indent=2)

    if log:
        log(stage=f"eval-{split}", **scores)
    return scores


def run_evaluate(args, train_cfg: TrainConfig) -> None:
    device = get_device()
    log = Logger(args.run_dir)

    ckpt = load_checkpoint(args.checkpoint)
    model = Summarizer(ModelConfig.from_dict(ckpt["model_cfg"]))
    model.load_state_dict(ckpt["model"])
    model.to(device)

    tokenizer, data = load_data(train_cfg, (args.split,))
    evaluate(model, data[args.split], tokenizer, train_cfg, args.run_dir, args.split, log)


# CLI


def add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--run-dir", default=None, help="Output directory (default: runs/<stage>-<language>)")
    p.add_argument("--language", default=TrainConfig.language, choices=LANGUAGES)
    p.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    p.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    p.add_argument("--lr", type=float, default=TrainConfig.lr)
    p.add_argument("--max-src-len", type=int, default=TrainConfig.max_src_len)
    p.add_argument("--max-tgt-len", type=int, default=TrainConfig.max_tgt_len)
    p.add_argument("--beam-size", type=int, default=TrainConfig.beam_size)
    p.add_argument("--limit", type=int, default=0, help="Use only the first N examples per split")
    p.add_argument("--seed", type=int, default=TrainConfig.seed)


def add_model(p: argparse.ArgumentParser) -> None:
    p.add_argument("--d-model", type=int, default=ModelConfig.d_model)
    p.add_argument("--heads", type=int, default=ModelConfig.n_heads)
    p.add_argument("--d-ff", type=int, default=ModelConfig.d_ff)
    p.add_argument("--layers", type=int, default=ModelConfig.n_layers)
    p.add_argument("--dropout", type=float, default=ModelConfig.dropout)
    p.add_argument("--positions", default=ModelConfig.encoder_positions, choices=("absolute", "relative", "both"))
    p.add_argument("--max-relative-position", type=int, default=ModelConfig.max_relative_position)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("autoencode", help="Train the comment autoencoder")
    add_common(p)
    add_model(p)

    p = sub.add_parser("align", help="Adversarially align a code encoder to the comment latent space")
    add_common(p)
    p.add_argument("--text-autoencoder", required=True, help="Checkpoint from the autoencode stage")
    p.add_argument("--g-lr", type=float, default=TrainConfig.g_lr)
    p.add_argument("--d-lr", type=float, default=TrainConfig.d_lr)
    p.add_argument("--mse-weight", type=float, default=TrainConfig.mse_weight)

    p = sub.add_parser("summarize", help="Train and test the code summarizer")
    add_common(p)
    add_model(p)
    p.add_argument("--no-copy", action="store_true", help="Disable copy attention")
    p.add_argument("--init-encoder", default=None, help="Code encoder checkpoint from the align stage")
    p.add_argument("--init-decoder", default=None, help="Autoencoder checkpoint whose decoder to start from")

    p = sub.add_parser("evaluate", help="Score a trained summarizer")
    add_common(p)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--split", default="test", choices=("validation", "test"))

    args = parser.parse_args()
    args.run_dir = args.run_dir or os.path.join(PATH_RUNS, f"{args.command}-{args.language}")

    train_cfg = TrainConfig(
        language=args.language,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_src_len=args.max_src_len,
        max_tgt_len=args.max_tgt_len,
        beam_size=args.beam_size,
        limit=args.limit,
        seed=args.seed,
    )
    if args.command == "align":
        train_cfg = replace(train_cfg, g_lr=args.g_lr, d_lr=args.d_lr, mse_weight=args.mse_weight)

    set_seed(train_cfg.seed)

    model_cfg_overrides = {}
    if args.command in ("autoencode", "summarize"):
        model_cfg_overrides = dict(
            d_model=args.d_model,
            n_heads=args.heads,
            d_ff=args.d_ff,
            n_layers=args.layers,
            dropout=args.dropout,
            encoder_positions=args.positions,
            max_relative_position=args.max_relative_position,
        )

    if args.command == "autoencode":
        run_autoencode(args, model_cfg_overrides, train_cfg)
    elif args.command == "align":
        run_align(args, train_cfg)
    elif args.command == "summarize":
        run_summarize(args, model_cfg_overrides, train_cfg)
    else:
        run_evaluate(args, train_cfg)


if __name__ == "__main__":
    main()
