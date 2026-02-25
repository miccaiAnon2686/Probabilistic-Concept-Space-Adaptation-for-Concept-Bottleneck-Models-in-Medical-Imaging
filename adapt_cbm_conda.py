#!/usr/bin/env python3

import argparse
import json
import os
import random
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, TensorDataset

import data_utils
import utils

WEIGHT_DECAY = 1e-4
CLIP_NAME = "none"


def set_deterministic(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_concepts(concept_set_path: str) -> Tuple[int, list[str]]:
    with open(concept_set_path, "r") as f:
        concepts = [ln.strip() for ln in f if ln.strip()]
    return len(concepts), concepts


class CBMHead(nn.Module):
    def __init__(self, feat_dim: int, num_concepts: int, num_classes: int):
        super().__init__()
        self.proj = nn.Linear(feat_dim, num_concepts, bias=False)
        self.register_buffer("proj_mean", torch.zeros(num_concepts))
        self.register_buffer("proj_std", torch.ones(num_concepts))
        self.classifier = nn.Linear(num_concepts, num_classes, bias=True)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_raw = self.proj(x)
        z_std = (z_raw - self.proj_mean) / (self.proj_std + 1e-8)
        logits = self.classifier(z_std)
        return logits, z_std, z_raw


def load_cbm_head(
    cbm_dir: str, feat_dim: int, num_concepts: int, device: torch.device
) -> Tuple[CBMHead, int]:
    W_c = torch.load(os.path.join(cbm_dir, "W_c.pt"), map_location="cpu")
    proj_mean = torch.load(os.path.join(cbm_dir, "proj_mean.pt"), map_location="cpu")
    proj_std = torch.load(os.path.join(cbm_dir, "proj_std.pt"), map_location="cpu")
    W_g = torch.load(os.path.join(cbm_dir, "W_g.pt"), map_location="cpu")
    b_g = torch.load(os.path.join(cbm_dir, "b_g.pt"), map_location="cpu")

    proj_mean = torch.as_tensor(proj_mean).view(-1)
    proj_std = torch.as_tensor(proj_std).view(-1)

    num_concepts_from_Wc, feat_dim_from_Wc = W_c.shape
    num_classes = W_g.shape[0]
    assert num_concepts_from_Wc == num_concepts, "Concept count mismatch with concept_set"
    assert feat_dim_from_Wc == feat_dim, "Feature dimension mismatch with cached features"
    assert proj_mean.numel() == num_concepts and proj_std.numel() == num_concepts

    head = CBMHead(feat_dim, num_concepts, num_classes)
    with torch.no_grad():
        head.proj.weight.copy_(W_c)
        head.proj_mean.copy_(proj_mean)
        head.proj_std.copy_(proj_std)
        head.classifier.weight.copy_(W_g)
        head.classifier.bias.copy_(b_g)
    return head.to(device).eval(), num_classes


def resolve_feature_path(
    activation_dir: str,
    target_dataset: str,
    backbone: str,
    feature_layer: str,
    pool_mode: str,
    concept_set: str,
) -> str:
    target_save_name, _, _ = utils.get_save_names(
        CLIP_NAME,
        backbone,
        "{}",
        target_dataset,
        concept_set,
        pool_mode,
        activation_dir,
    )
    return target_save_name.format(feature_layer)


def load_cached_features(path: str) -> torch.Tensor:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Cached target features not found: {path}. "
            "Please precompute activations in activation_dir first."
        )
    feats = torch.load(path, map_location="cpu")
    if isinstance(feats, np.ndarray):
        feats = torch.from_numpy(feats)
    elif isinstance(feats, dict) and "feats" in feats:
        feats = torch.from_numpy(feats["feats"]) if isinstance(feats["feats"], np.ndarray) else feats["feats"]
    return feats.float()


def try_get_labels(split_key: str, n_expected: int) -> Optional[torch.Tensor]:
    try:
        y = data_utils.get_targets_only(split_key)
    except Exception:
        return None
    if y is None:
        return None
    y = torch.as_tensor(y, dtype=torch.long)
    if y.numel() != n_expected:
        return None
    return y


def l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def cosine_similarity(x: torch.Tensor, protos: torch.Tensor) -> torch.Tensor:
    return l2_normalize(x) @ l2_normalize(protos).t()


@torch.no_grad()
def kl_style_similarity(x: torch.Tensor, protos: torch.Tensor) -> torch.Tensor:
    eps = 1e-8
    s = torch.sigmoid(x).clamp(eps, 1 - eps)
    p = torch.sigmoid(protos).clamp(eps, 1 - eps)
    s_exp = s.unsqueeze(1)
    p_exp = p.unsqueeze(0)
    score = s_exp * torch.log(p_exp) + (1 - s_exp) * torch.log(1 - p_exp)
    return score.sum(dim=-1)


def compute_prototypes(
    features_space: torch.Tensor,
    logits_like: torch.Tensor,
    temperature: float,
    w_thresh: float,
    w_top_percent: float,
    w_min_fallback: int,
) -> torch.Tensor:
    eps = 1e-8
    N = features_space.shape[0]
    K = logits_like.shape[1]

    w = F.softmax(logits_like / max(eps, temperature), dim=1).detach()

    if w_thresh > 0.0:
        w = w * (w > w_thresh).float()

    if w_top_percent > 0.0:
        pct = max(0.0, min(100.0, w_top_percent))
        if pct > 0.0:
            w_sorted, _ = torch.sort(w, dim=0)
            cut_idx = torch.clamp(torch.floor(torch.tensor(N * pct / 100.0)).long(), 0, N - 1)
            cut_vals = w_sorted[cut_idx, torch.arange(K)]
            w = w * (w > cut_vals.unsqueeze(0)).float()

    if w_min_fallback > 0:
        per_class_sum = w.sum(dim=0)
        empty = per_class_sum <= eps
        if empty.any():
            w_base = F.softmax(logits_like / max(eps, temperature), dim=1).detach()
            topk = min(int(w_min_fallback), N)
            if topk > 0:
                for c in torch.nonzero(empty, as_tuple=False).view(-1):
                    vals = w_base[:, c]
                    k = min(topk, (vals > 0).sum().item())
                    if k > 0:
                        topi = torch.topk(vals, k=k, largest=True).indices
                        w[:, c] = 0.0
                        w[topi, c] = vals[topi]

    denom = w.sum(dim=0).clamp_min(eps)
    protos = (features_space.unsqueeze(1) * w.unsqueeze(-1)).sum(dim=0) / denom.unsqueeze(-1)
    return protos


def build_pseudo_labels(
    head: CBMHead,
    feats: torch.Tensor,
    device: torch.device,
    args: argparse.Namespace,
) -> Tuple[torch.Tensor, torch.Tensor]:
    head.eval()
    with torch.no_grad():
        feats_d = feats.to(device)
        logits, z_std, _ = head(feats_d)
        logits_like = logits

        if args.pl_logit_norm == "l1":
            denom = logits_like.abs().sum(dim=1, keepdim=True).clamp_min(1e-8)
            logits_like = logits_like / denom
        if args.pl_space == "concept":
            s = z_std
        else:
            s = feats_d

        s_for_proto = s.clone()
        if args.abs_space in ("z", "both"):
            s_for_proto = s_for_proto.abs()
        if args.pl_proto_transform == "sigmoid":
            s_for_proto = torch.sigmoid(s_for_proto)

        protos = compute_prototypes(
            s_for_proto,
            logits_like,
            args.temperature,
            args.w_thresh,
            args.w_top_percent,
            args.w_min_fallback,
        )

        if args.pl_proto_transform == "sigmoid":
            eps = 1e-6
            protos = torch.logit(protos.clamp(eps, 1 - eps))
        if args.abs_space in ("proto", "both"):
            protos = protos.abs()

        if args.pl_distance == "cosine":
            sims = cosine_similarity(s, protos)
        else:
            sims = kl_style_similarity(s, protos)

    conf, yhat = sims.max(dim=1)
    return yhat.cpu(), conf.cpu()


def select_indices_per_threshold_and_cap(
    conf: torch.Tensor,
    yhat: torch.Tensor,
    proto_thresh: float,
    max_per_class: int,
    num_classes: int,
) -> torch.Tensor:
    keep = conf >= proto_thresh
    if max_per_class <= 0:
        return torch.nonzero(keep, as_tuple=False).view(-1)

    sel_indices = []
    for c in range(num_classes):
        idx_c = torch.nonzero((yhat == c) & keep, as_tuple=False).view(-1)
        if idx_c.numel() == 0:
            continue
        conf_c = conf[idx_c]
        topk = min(max_per_class, idx_c.numel())
        topk_pos = torch.topk(conf_c, k=topk, largest=True).indices
        sel_indices.append(idx_c[topk_pos])
    if not sel_indices:
        return torch.empty(0, dtype=torch.long)
    return torch.cat(sel_indices, dim=0)


def finetune_head(
    head: CBMHead,
    feats: torch.Tensor,
    pl_indices: torch.Tensor,
    pl_labels: torch.Tensor,
    device: torch.device,
    args: argparse.Namespace,
) -> None:
    head.train()
    params = [
        {"params": head.proj.parameters(), "lr": args.lr_proj},
        {"params": head.classifier.parameters(), "lr": args.lr_cls},
    ]
    opt = torch.optim.SGD(params, momentum=0.9, weight_decay=WEIGHT_DECAY)
    crit = nn.CrossEntropyLoss()

    dataset = TensorDataset(feats[pl_indices].float(), pl_labels[pl_indices].long())
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)

    for _ in range(args.epochs):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits, _, _ = head(xb)
            loss = crit(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()


def masked_top1(logits: torch.Tensor, labels: Optional[torch.Tensor]) -> Optional[float]:
    if labels is None:
        return None
    mask = labels >= 0
    if mask.sum().item() == 0:
        return None
    preds = logits.argmax(dim=1).cpu()[mask]
    y = labels[mask]
    return (preds == y).float().mean().item()


def load_yaml_config(path: str) -> dict:
    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return cfg


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Minimal CBM adaptation")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--cbm_dir", type=str, default=None)
    p.add_argument("--concept_set", type=str, default=None)
    p.add_argument("--target_dataset", type=str, default=None)
    p.add_argument("--activation_dir", type=str, default=None)
    p.add_argument("--out_dir", type=str, default=None)

    p.add_argument("--backbone", type=str, default="clip_ViT-B/16")
    p.add_argument("--feature_layer", type=str, default="cls_preproj")
    p.add_argument("--pool_mode", type=str, default="avg", choices=["avg", "max"])
    p.add_argument("--pl_space", type=str, default="concept", choices=["concept", "backbone"])
    p.add_argument("--pl_distance", type=str, default="cosine", choices=["cosine", "kl"])
    p.add_argument("--abs_space", type=str, default="none", choices=["none", "z", "proto", "both"])
    p.add_argument("--pl_proto_transform", type=str, default="none", choices=["none", "sigmoid"])
    p.add_argument("--pl_logit_norm", type=str, default="none", choices=["none", "l1"])

    p.add_argument("--temperature", type=float, default=0.01)
    p.add_argument("--w_thresh", type=float, default=0.3)
    p.add_argument("--w_top_percent", type=float, default=95.0)
    p.add_argument("--w_min_fallback", type=int, default=5)
    p.add_argument("--proto_thresh", type=float, default=-20.0)
    p.add_argument("--max_per_class", type=int, default=300)
    p.add_argument("--epochs", type=int, default=35)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--lr_proj", type=float, default=1e-4)
    p.add_argument("--lr_cls", type=float, default=2e-4)
    p.add_argument("--seed", type=int, default=999)
    return p


def parse_args() -> argparse.Namespace:
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--config", type=str, default=None)
    boot_args, remaining = bootstrap.parse_known_args()

    parser = build_parser()
    if boot_args.config:
        cfg = load_yaml_config(boot_args.config)
        valid = {a.dest for a in parser._actions}
        bad = sorted(set(cfg.keys()) - valid)
        if bad:
            parser.error(f"Unknown config keys in {boot_args.config}: {bad}")
        parser.set_defaults(**cfg)

    args = parser.parse_args(remaining)
    required = ["cbm_dir", "concept_set", "target_dataset", "activation_dir", "out_dir"]
    missing = [k for k in required if getattr(args, k) in (None, "")]
    if missing:
        parser.error(f"Missing required arguments (or config values): {missing}")
    return args


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    set_deterministic(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    feature_path = resolve_feature_path(
        args.activation_dir,
        args.target_dataset,
        args.backbone,
        args.feature_layer,
        args.pool_mode,
        args.concept_set,
    )
    feats = load_cached_features(feature_path)
    N, D = feats.shape
    labels = try_get_labels(args.target_dataset, N)

    num_concepts, _ = load_concepts(args.concept_set)
    head, num_classes = load_cbm_head(args.cbm_dir, D, num_concepts, device)

    with torch.no_grad():
        logits_before, _, _ = head(feats.to(device))
    top1_before = masked_top1(logits_before, labels)

    yhat_b, conf_b = build_pseudo_labels(head, feats, device, args)
    sel_idx_b = select_indices_per_threshold_and_cap(
        conf_b, yhat_b, args.proto_thresh, args.max_per_class, num_classes
    )

    if sel_idx_b.numel() > 0:
        finetune_head(head, feats, sel_idx_b, yhat_b, device, args)

    head.eval()
    with torch.no_grad():
        logits_after, _, _ = head(feats.to(device))
    top1_after = masked_top1(logits_after, labels)

    torch.save(head.proj.weight.detach().cpu(), os.path.join(args.out_dir, "W_c_adapted.pt"))
    torch.save(head.classifier.weight.detach().cpu(), os.path.join(args.out_dir, "W_g_adapted.pt"))
    torch.save(head.classifier.bias.detach().cpu(), os.path.join(args.out_dir, "b_g_adapted.pt"))

    metrics = {
        "args": vars(args),
        "N_target": int(N),
        "selected_count": int(sel_idx_b.numel()),
        "top1_before_percent": None if top1_before is None else round(top1_before * 100, 4),
        "top1_after_percent": None if top1_after is None else round(top1_after * 100, 4),
    }
    with open(os.path.join(args.out_dir, "adapt_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    if top1_before is None or top1_after is None:
        print(f"[EVAL] N_target={N} top1_before=None top1_after=None")
    else:
        print(
            f"[EVAL] N_target={N} top1_before={top1_before*100:.4f}% "
            f"top1_after={top1_after*100:.4f}%"
        )


if __name__ == "__main__":
    main()
