#!/usr/bin/env python
"""Prune-TICU: hybrid attention-head pruning + trigger-invariant contrastive unlearning.

This script follows the same data layout used by ticu_universal.py:

helper_dir/
  Data_Poisoning/<Attack>/<Dataset>/
    train_poisoned.csv
    train_clean.csv
    test_clean.csv
    test_poisoned_all.csv
    test_poisoned_part.csv

Main modes:
  --defense ticu       : TICU only
  --defense prune      : trigger-aware head pruning only
  --defense hybrid     : trigger-aware pruning followed by TICU
  --defense compare    : run TICU, pruning, and hybrid from the same poisoned checkpoint

Hybrid idea:
  1. Train/load a poisoned classifier.
  2. Estimate per-head CLEAN importance from CE gradients on clean data.
  3. Estimate per-head TRIGGER sensitivity from gradients of prediction invariance
     between clean and trigger-inserted counterfactual inputs.
  4. Prune heads with high trigger sensitivity but low clean importance, while
     backtracking if clean validation/test accuracy falls below --acc_threshold.
  5. Run TICU on the pruned model to remove remaining trigger effects.

Notes:
  * Real head pruning is supported for HuggingFace BERT/DistilBERT/RoBERTa-style
    encoder models exposing base_model.prune_heads(...). For decoder LoRA models,
    use --defense ticu. The script will warn and skip pruning if the model does
    not expose a prune_heads method.
  * For BERT-like models, --model_type auto is recommended.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_linear_schedule_with_warmup,
)

try:
    from lora_llama import build_llama_with_lora
    from lora_qwen import build_qwen_with_lora
    from lora_mistral import build_mistral_with_lora
except Exception:
    build_llama_with_lora = build_qwen_with_lora = build_mistral_with_lora = None


# -----------------------
# Repro / basic utilities
# -----------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_model_device(model: nn.Module, fallback: torch.device) -> torch.device:
    for attr in ["backbone", "llama", "model", "bert", "roberta", "distilbert"]:
        if hasattr(model, attr):
            try:
                return next(p.device for p in getattr(model, attr).parameters() if p is not None)
            except Exception:
                pass
    try:
        return next(p.device for p in model.parameters() if p is not None)
    except StopIteration:
        return fallback


def maybe_to_device(model: nn.Module, device: torch.device) -> nn.Module:
    # Quantized/device_map models should not be moved wholesale.
    if getattr(model, "is_loaded_in_4bit", False) or getattr(model, "is_loaded_in_8bit", False):
        return model
    if getattr(model, "hf_device_map", None) is not None:
        return model
    return model.to(device)


def read_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "text" not in df.columns:
        for alt in ["sentence", "comment_text", "content"]:
            if alt in df.columns:
                df = df.rename(columns={alt: "text"})
                break
    if "text" not in df.columns:
        raise ValueError(f"CSV missing text/sentence/comment_text/content column: {path}")
    if "label" not in df.columns:
        raise ValueError(f"CSV missing label column: {path}")
    if "poisoned" not in df.columns:
        df["poisoned"] = 0
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)
    df["poisoned"] = df["poisoned"].astype(int)
    return df


@dataclass
class TaskSpec:
    num_labels: int


def task_spec(dataset_name: str) -> TaskSpec:
    ds = dataset_name.upper()
    if ds == "AG":
        return TaskSpec(4)
    if ds in ["SST-2", "SST2", "HSOL"]:
        return TaskSpec(2)
    raise ValueError("Expected --dataset SST-2, HSOL, or AG")


# -----------------------
# Dataset / DataLoader
# -----------------------
class CSVDataset(Dataset):
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        enc = self.tokenizer(row["text"], truncation=True, max_length=self.max_length)
        enc["labels"] = int(row["label"])
        enc["poisoned"] = int(row.get("poisoned", 0))
        return enc


def make_loader(ds: Dataset, tokenizer, batch_size: int, shuffle: bool) -> DataLoader:
    collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=collator)


# -----------------------
# Metrics
# -----------------------
def compute_lfr_from_cm(cm: np.ndarray, dataset: str) -> Dict[str, float]:
    ds = dataset.upper()
    out = {"cm": cm.tolist()}
    if ds in ["SST-2", "SST2", "HSOL"]:
        if cm.shape[0] < 2 or cm.shape[1] < 2:
            out.update({"lfr_0": 0.0, "lfr_1": 0.0})
            return out
        out["lfr_0"] = float(cm[0, 1] / (cm[0, :].sum() + 1e-12))
        out["lfr_1"] = float(cm[1, 0] / (cm[1, :].sum() + 1e-12))
        return out
    # AG or other multiclass: mean off-diagonal flip rate over non-target rows.
    row_rates = []
    for i in range(cm.shape[0]):
        row_rates.append(float((cm[i, :].sum() - cm[i, i]) / (cm[i, :].sum() + 1e-12)))
    out["lfr"] = float(np.mean(row_rates))
    return out


@torch.no_grad()
def eval_acc_lfr(model: nn.Module, loader: DataLoader, device: torch.device, dataset: str) -> Dict[str, float]:
    model.eval()
    model_device = get_model_device(model, device)
    preds_all, labels_all = [], []
    for batch in loader:
        input_ids = batch["input_ids"].to(model_device)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(model_device)
        labels = batch["labels"].to(model_device)
        out = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        preds_all.append(torch.argmax(out.logits, dim=-1).detach().cpu())
        labels_all.append(labels.detach().cpu())
    preds = torch.cat(preds_all).numpy()
    labels = torch.cat(labels_all).numpy()
    acc = float((preds == labels).mean())
    cm = confusion_matrix(labels, preds)
    result = {"acc": acc}
    result.update(compute_lfr_from_cm(cm, dataset))
    return result


def metric_line(prefix: str, metrics: Dict[str, float], dataset: str) -> str:
    if dataset.upper() in ["SST-2", "SST2", "HSOL"]:
        return f"{prefix} ACC={metrics['acc']:.4f}, LFR0={metrics.get('lfr_0',0):.4f}, LFR1={metrics.get('lfr_1',0):.4f}"
    return f"{prefix} ACC={metrics['acc']:.4f}, LFR={metrics.get('lfr',0):.4f}"


# -----------------------
# Trigger inference / insertion
# -----------------------
_WORD = re.compile(r"[A-Za-z0-9@\-']+")


def _tokenize_simple(text: str) -> List[str]:
    return _WORD.findall(text.lower())


def infer_trigger_autoselect(df_train_poisoned: pd.DataFrame, attack: str) -> List[str]:
    atk = attack.lower()
    pois = df_train_poisoned[df_train_poisoned["poisoned"] == 1]["text"].astype(str)
    clean = df_train_poisoned[df_train_poisoned["poisoned"] == 0]["text"].astype(str)
    if len(pois) == 0:
        raise ValueError("No poisoned==1 rows; use --trigger_mode manual --trigger ...")

    if atk == "addsent":
        from collections import Counter
        cnt = Counter()
        for s in pois.sample(min(1200, len(pois)), random_state=0):
            toks = _tokenize_simple(s)
            for i in range(max(0, len(toks) - 3)):
                cnt[tuple(toks[i:i + 4])] += 1
        best, _ = cnt.most_common(1)[0]
        phrase = " ".join(best)
        return [phrase + " movie" if best[-1] == "3d" else phrase]

    if atk == "badnet":
        from collections import Counter
        c_p, c_c = Counter(), Counter()
        for s in pois.sample(min(2000, len(pois)), random_state=0):
            c_p.update(_tokenize_simple(s))
        for s in clean.sample(min(2000, len(clean)), random_state=0):
            c_c.update(_tokenize_simple(s))
        candidates = [(fp, w) for w, fp in c_p.items() if fp >= 20 and c_c.get(w, 0) == 0]
        candidates.sort(reverse=True)
        if candidates:
            return [w for _, w in candidates[:2]]
        ratios = []
        total_p, total_c = sum(c_p.values()) + 1, sum(c_c.values()) + 1
        for w, fp in c_p.items():
            if fp >= 20:
                ratios.append(((fp / total_p) / ((c_c.get(w, 0) + 1) / total_c), w))
        ratios.sort(reverse=True)
        return [ratios[0][1]] if ratios else ["cf"]

    if atk == "hiddenkiller":
        from collections import Counter
        cnt = Counter()
        for s in pois.sample(min(2000, len(pois)), random_state=0):
            toks = _tokenize_simple(s)
            for i in range(max(0, len(toks) - 2)):
                cnt[tuple(toks[i:i + 3])] += 1
        if cnt:
            best, _ = cnt.most_common(1)[0]
            return [" ".join(best)]
        return ["when"]

    if atk == "stylebkd":
        return ["@-@", "hath", "ye", "unto", "doth", "apos"]
    return ["cf"]


def trigger_token_ids(tokenizer, trigger_text: str) -> List[int]:
    ids = tokenizer.encode(trigger_text, add_special_tokens=False)
    if not ids:
        raise ValueError(f"Trigger {trigger_text!r} produces empty tokenization")
    return ids


def insert_trigger_batch(
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    tokenizer,
    trigger_ids: List[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = input_ids.device
    batch_size, max_len = input_ids.shape
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    boundary_id = getattr(tokenizer, "sep_token_id", None) or getattr(tokenizer, "eos_token_id", None)
    if attention_mask is None:
        attention_mask = (input_ids != pad_id).long()

    new_ids = torch.full((batch_size, max_len), pad_id, dtype=input_ids.dtype, device=device)
    new_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=device)
    trig_len = len(trigger_ids)

    for i in range(batch_size):
        length = int(attention_mask[i].sum().item())
        seq = input_ids[i, :length].tolist()
        insert_pos = length
        if boundary_id is not None and boundary_id in seq:
            insert_pos = seq.index(boundary_id)
        prefix, suffix = seq[:insert_pos], seq[insert_pos:]
        seq2 = prefix + trigger_ids + suffix
        if len(seq2) > max_len:
            overflow = len(seq2) - max_len
            # Prefer keeping special boundary tokens; trim ordinary content before trigger.
            if len(prefix) > 1:
                prefix = prefix[:-min(overflow, len(prefix) - 1)]
            seq2 = (prefix + trigger_ids + suffix)[:max_len]
        new_ids[i, :len(seq2)] = torch.tensor(seq2, dtype=input_ids.dtype, device=device)
        new_mask[i, :len(seq2)] = 1
    return new_ids, new_mask


def make_triggered_batch(batch: Dict[str, torch.Tensor], tokenizer, trigger_ids_list: List[List[int]]) -> Tuple[torch.Tensor, torch.Tensor]:
    input_ids = batch["input_ids"]
    attention_mask = batch.get("attention_mask")
    if len(trigger_ids_list) == 1:
        return insert_trigger_batch(input_ids, attention_mask, tokenizer, trigger_ids_list[0])
    trig_ids = input_ids.clone()
    trig_mask = attention_mask.clone() if attention_mask is not None else torch.ones_like(input_ids)
    for i in range(input_ids.size(0)):
        ids_i, mask_i = insert_trigger_batch(
            input_ids[i:i + 1],
            trig_mask[i:i + 1],
            tokenizer,
            trigger_ids_list[random.randrange(len(trigger_ids_list))],
        )
        trig_ids[i], trig_mask[i] = ids_i[0], mask_i[0]
    return trig_ids, trig_mask


# -----------------------
# Model builder
# -----------------------
def build_model(model_type: str, model_name: str, num_labels: int, args=None) -> nn.Module:
    mt = model_type.lower().strip()
    if mt == "auto":
        if getattr(args, "use_lora", False):
            raise ValueError("LoRA is disabled for --model_type auto; use llama/qwen/mistral.")
        return AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    if mt == "llama" and build_llama_with_lora is not None:
        return build_llama_with_lora(model_name=model_name, num_labels=num_labels, args=args)
    if mt == "qwen" and build_qwen_with_lora is not None:
        return build_qwen_with_lora(model_name=model_name, num_labels=num_labels, args=args)
    if mt == "mistral" and build_mistral_with_lora is not None:
        return build_mistral_with_lora(model_name=model_name, num_labels=num_labels, args=args)
    raise ValueError("LoRA builder files are missing or --model_type is invalid.")


# -----------------------
# Shared losses
# -----------------------
def sym_kl(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    kl_pq = (p * (torch.log(p + 1e-12) - torch.log(q + 1e-12))).sum(dim=-1).mean()
    kl_qp = (q * (torch.log(q + 1e-12) - torch.log(p + 1e-12))).sum(dim=-1).mean()
    return kl_pq + kl_qp


def cosine_inv(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (1.0 - (F.normalize(a, dim=-1) * F.normalize(b, dim=-1)).sum(dim=-1)).mean()


def clone_state(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def l2_distance_sq(model: nn.Module, base_state: Dict[str, torch.Tensor]) -> torch.Tensor:
    total = None
    for name, p in model.named_parameters():
        if not p.requires_grad or name not in base_state:
            continue
        d = p - base_state[name].to(p.device)
        val = (d * d).sum()
        total = val if total is None else total + val
    if total is None:
        return torch.tensor(0.0, device=get_model_device(model, torch.device("cpu")))
    return total


def get_representation(out, attention_mask: Optional[torch.Tensor], tokenizer) -> Optional[torch.Tensor]:
    if getattr(out, "hidden_states", None) is None:
        return None
    hs = out.hidden_states[-1]
    if getattr(tokenizer, "cls_token_id", None) is not None:
        return hs[:, 0, :]
    if attention_mask is None:
        return hs[:, -1, :]
    seq_lengths = attention_mask.sum(1) - 1
    batch_idx = torch.arange(hs.size(0), device=hs.device)
    return hs[batch_idx, seq_lengths]


# -----------------------
# Training
# -----------------------
def train_supervised(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    lr: float,
    epochs: int,
    weight_decay: float,
    max_grad_norm: float,
    grad_accum: int = 1,
    fp16: bool = False,
    bf16: bool = False,
) -> nn.Module:
    model = maybe_to_device(model, device)
    model_device = get_model_device(model, device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    steps_per_epoch = max(1, (len(loader) + grad_accum - 1) // grad_accum)
    total_steps = max(1, epochs * steps_per_epoch)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * total_steps), total_steps)
    use_amp = fp16 or bf16
    scaler = torch.cuda.amp.GradScaler(enabled=fp16)
    amp_dtype = torch.bfloat16 if bf16 else torch.float16

    for ep in range(1, epochs + 1):
        model.train()
        opt.zero_grad(set_to_none=True)
        running = 0.0
        for step, batch in enumerate(loader, start=1):
            batch = {k: v.to(model_device) for k, v in batch.items()}
            with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
                loss = model(**batch, return_dict=True).loss / grad_accum
            if fp16:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            running += float(loss.detach().cpu()) * grad_accum
            if step % grad_accum == 0:
                if fp16:
                    scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(params, max_grad_norm)
                if fp16:
                    scaler.step(opt); scaler.update()
                else:
                    opt.step()
                sched.step(); opt.zero_grad(set_to_none=True)
        print(f"[poison-train] epoch={ep}/{epochs} loss={running / max(1, len(loader)):.4f}")
    return model


# -----------------------
# TICU unlearning
# -----------------------
def ticu_unlearn(
    model: nn.Module,
    base_state: Dict[str, torch.Tensor],
    clean_loader: DataLoader,
    tokenizer,
    triggers: List[str],
    device: torch.device,
    lr: float,
    epochs: int,
    gamma: float,
    delta: float,
    beta: float,
    weight_decay: float,
    max_grad_norm: float,
    grad_accum: int = 1,
    fp16: bool = False,
    bf16: bool = False,
) -> nn.Module:
    model = maybe_to_device(model, device)
    model_device = get_model_device(model, device)
    trigger_ids_list = [trigger_token_ids(tokenizer, t) for t in triggers]
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    steps_per_epoch = max(1, (len(clean_loader) + grad_accum - 1) // grad_accum)
    total_steps = max(1, epochs * steps_per_epoch)
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * total_steps), total_steps)
    use_amp = fp16 or bf16
    scaler = torch.cuda.amp.GradScaler(enabled=fp16)
    amp_dtype = torch.bfloat16 if bf16 else torch.float16

    for ep in range(1, epochs + 1):
        start = time.time()
        model.train(); opt.zero_grad(set_to_none=True)
        running = {"ce": 0.0, "inv": 0.0, "rep": 0.0, "reg": 0.0}
        for step, batch in enumerate(clean_loader, start=1):
            batch = {k: v.to(model_device) for k, v in batch.items()}
            input_ids = batch["input_ids"]
            attention_mask = batch.get("attention_mask")
            labels = batch["labels"]
            trig_ids, trig_mask = make_triggered_batch(batch, tokenizer, trigger_ids_list)

            with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
                out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels,
                            output_hidden_states=True, return_dict=True)
                out_t = model(input_ids=trig_ids, attention_mask=trig_mask,
                              output_hidden_states=True, return_dict=True)
                ce = out.loss
                inv = sym_kl(F.softmax(out.logits, dim=-1), F.softmax(out_t.logits, dim=-1))
                rep = torch.tensor(0.0, device=model_device)
                rep_c = get_representation(out, attention_mask, tokenizer)
                rep_t = get_representation(out_t, trig_mask, tokenizer)
                if delta > 0 and rep_c is not None and rep_t is not None:
                    rep = cosine_inv(rep_c, rep_t)
                reg = beta * l2_distance_sq(model, base_state)
                loss = (ce + gamma * inv + delta * rep + reg) / grad_accum

            if fp16:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            if step % grad_accum == 0:
                if fp16:
                    scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(params, max_grad_norm)
                if fp16:
                    scaler.step(opt); scaler.update()
                else:
                    opt.step()
                sched.step(); opt.zero_grad(set_to_none=True)

            running["ce"] += float(ce.detach().cpu())
            running["inv"] += float(inv.detach().cpu())
            running["rep"] += float(rep.detach().cpu())
            running["reg"] += float(reg.detach().cpu())
        denom = max(1, len(clean_loader))
        print(f"[TICU] epoch={ep}/{epochs} ce={running['ce']/denom:.4f} inv={running['inv']/denom:.4f} "
              f"rep={running['rep']/denom:.4f} reg={running['reg']/denom:.4f} time={time.time()-start:.1f}s")
    return model


# -----------------------
# Head pruning support
# -----------------------
def resolve_prunable_base(model: nn.Module) -> Optional[nn.Module]:
    # Most HF encoder classifiers expose base_model.prune_heads.
    if hasattr(model, "base_model") and hasattr(model.base_model, "prune_heads"):
        return model.base_model
    for attr in ["bert", "roberta", "distilbert", "electra"]:
        if hasattr(model, attr) and hasattr(getattr(model, attr), "prune_heads"):
            return getattr(model, attr)
    if hasattr(model, "prune_heads"):
        return model
    return None


def get_num_layers_heads(model: nn.Module) -> Tuple[int, int]:
    cfg = model.config
    n_layers = getattr(cfg, "num_hidden_layers", None) or getattr(cfg, "n_layers", None)
    n_heads = getattr(cfg, "num_attention_heads", None) or getattr(cfg, "n_heads", None)
    if n_layers is None or n_heads is None:
        raise ValueError("Cannot infer num_hidden_layers/num_attention_heads from model.config")
    return int(n_layers), int(n_heads)


def head_key_mask(param: torch.Tensor, head: int, n_heads: int) -> torch.Tensor:
    # BERT-style projection weights are [all_head_size, hidden_size]. Rows are head chunks.
    rows = param.shape[0]
    head_dim = rows // n_heads
    mask = torch.zeros_like(param)
    start, end = head * head_dim, (head + 1) * head_dim
    mask[start:end, :] = 1.0
    return mask


def locate_key_parameters(model: nn.Module, n_layers: int) -> Dict[int, Tuple[str, nn.Parameter]]:
    # Prefer canonical BERT/RoBERTa/ELECTRA names, fallback to first key projection per layer.
    named = dict(model.named_parameters())
    result: Dict[int, Tuple[str, nn.Parameter]] = {}
    for layer in range(n_layers):
        candidates = [
            f"bert.encoder.layer.{layer}.attention.self.key.weight",
            f"roberta.encoder.layer.{layer}.attention.self.key.weight",
            f"electra.encoder.layer.{layer}.attention.self.key.weight",
            f"base_model.encoder.layer.{layer}.attention.self.key.weight",
        ]
        for name in candidates:
            if name in named:
                result[layer] = (name, named[name])
                break
        if layer not in result:
            # Loose fallback: must contain layer id, attention, key, weight.
            needle = f"layer.{layer}."
            for name, p in named.items():
                lname = name.lower()
                if needle in lname and "attention" in lname and "key" in lname and lname.endswith("weight"):
                    result[layer] = (name, p)
                    break
    return result


def accumulate_head_gradients(
    model: nn.Module,
    loader: DataLoader,
    tokenizer,
    triggers: List[str],
    device: torch.device,
    objective: str,
    max_batches: int,
) -> Dict[int, torch.Tensor]:
    """Return per-layer per-head gradient scores.

    objective='clean': CE gradient on clean labels (clean utility score)
    objective='trigger': symmetric-KL gradient between clean and triggered inputs
    """
    assert objective in {"clean", "trigger"}
    n_layers, n_heads = get_num_layers_heads(model)
    key_params = locate_key_parameters(model, n_layers)
    if len(key_params) == 0:
        raise RuntimeError("Could not locate key projection weights for head scoring.")
    scores = {l: torch.zeros(n_heads) for l in range(n_layers)}
    trigger_ids_list = [trigger_token_ids(tokenizer, t) for t in triggers]
    model.train()
    model_device = get_model_device(model, device)

    for bidx, batch in enumerate(loader, start=1):
        if max_batches > 0 and bidx > max_batches:
            break
        model.zero_grad(set_to_none=True)
        batch = {k: v.to(model_device) for k, v in batch.items()}
        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask")
        labels = batch["labels"]
        if objective == "clean":
            loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, return_dict=True).loss
        else:
            trig_ids, trig_mask = make_triggered_batch(batch, tokenizer, trigger_ids_list)
            out = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
            out_t = model(input_ids=trig_ids, attention_mask=trig_mask, return_dict=True)
            loss = sym_kl(F.softmax(out.logits, dim=-1), F.softmax(out_t.logits, dim=-1))
        loss.backward()
        for layer, (_, p) in key_params.items():
            if p.grad is None:
                continue
            grad = p.grad.detach()
            for head in range(n_heads):
                scores[layer][head] += torch.norm(grad * head_key_mask(grad, head, n_heads)).cpu()
    return scores


def normalize_scores(scores: Dict[int, torch.Tensor]) -> Dict[int, torch.Tensor]:
    all_vals = torch.cat([v.flatten() for v in scores.values()])
    mn, mx = float(all_vals.min()), float(all_vals.max())
    denom = max(mx - mn, 1e-12)
    return {k: (v - mn) / denom for k, v in scores.items()}


def rank_hybrid_heads(clean_scores: Dict[int, torch.Tensor], trig_scores: Dict[int, torch.Tensor], alpha: float) -> List[Tuple[int, int, float, float, float]]:
    clean_n = normalize_scores(clean_scores)
    trig_n = normalize_scores(trig_scores)
    candidates = []
    for layer in sorted(trig_n.keys()):
        for head in range(len(trig_n[layer])):
            trig = float(trig_n[layer][head])
            clean = float(clean_n.get(layer, torch.zeros_like(trig_n[layer]))[head])
            # High score means prune earlier: high trigger sensitivity, low clean utility.
            score = trig - alpha * clean
            candidates.append((layer, head, score, trig, clean))
    candidates.sort(key=lambda x: x[2], reverse=True)
    return candidates


def safe_pruned_dict(pruned: Dict[int, List[int]], n_heads: int) -> Dict[int, List[int]]:
    out = {}
    for layer, heads in pruned.items():
        unique = sorted(set(int(h) for h in heads))
        # Keep at least one head per layer.
        if len(unique) >= n_heads:
            unique = unique[:-1]
        if unique:
            out[int(layer)] = unique
    return out


def apply_heads(model: nn.Module, heads: Dict[int, List[int]]) -> nn.Module:
    base = resolve_prunable_base(model)
    if base is None:
        raise RuntimeError("Model does not support prune_heads.")
    base.prune_heads({int(k): sorted(set(v)) for k, v in heads.items() if len(v) > 0})
    return model


def trigger_aware_prune(
    model: nn.Module,
    train_clean_loader: DataLoader,
    val_loader: DataLoader,
    tokenizer,
    triggers: List[str],
    device: torch.device,
    dataset: str,
    acc_threshold: float,
    prune_step: int,
    alpha_clean: float,
    max_score_batches: int,
    out_dir: str,
) -> Tuple[nn.Module, Dict[int, List[int]]]:
    if resolve_prunable_base(model) is None:
        print("[prune] WARNING: model does not expose prune_heads; skipping pruning.")
        return model, {}

    print("[prune] scoring heads with clean CE gradients...")
    clean_scores = accumulate_head_gradients(model, train_clean_loader, tokenizer, triggers, device, "clean", max_score_batches)
    print("[prune] scoring heads with trigger-invariance gradients...")
    trigger_scores = accumulate_head_gradients(model, train_clean_loader, tokenizer, triggers, device, "trigger", max_score_batches)
    ranked = rank_hybrid_heads(clean_scores, trigger_scores, alpha_clean)
    n_layers, n_heads = get_num_layers_heads(model)

    original_state = copy.deepcopy(model.state_dict())
    pruned: Dict[int, List[int]] = {l: [] for l in range(n_layers)}
    best_heads: Dict[int, List[int]] = {}
    best_acc = eval_acc_lfr(model, val_loader, device, dataset)["acc"]
    print(f"[prune] initial validation ACC={best_acc:.4f}; threshold={acc_threshold:.4f}")

    for start in range(0, len(ranked), prune_step):
        step_heads = ranked[start:start + prune_step]
        trial = copy.deepcopy(model)
        trial.load_state_dict(original_state, strict=False)
        trial_pruned = copy.deepcopy(pruned)
        for layer, head, _, _, _ in step_heads:
            trial_pruned[layer].append(head)
        trial_pruned = safe_pruned_dict(trial_pruned, n_heads)
        apply_heads(trial, trial_pruned)
        acc = eval_acc_lfr(trial, val_loader, device, dataset)["acc"]
        print(f"[prune] trial_heads={sum(len(v) for v in trial_pruned.values())} val_ACC={acc:.4f}")
        if acc < acc_threshold:
            print("[prune] stopping before this step because validation accuracy fell below threshold.")
            break
        pruned = trial_pruned
        best_heads = copy.deepcopy(pruned)
        best_acc = acc

    model.load_state_dict(original_state, strict=False)
    if best_heads:
        apply_heads(model, best_heads)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "hybrid_pruned_heads.json"), "w") as f:
        json.dump({str(k): v for k, v in best_heads.items()}, f, indent=2)
    with open(os.path.join(out_dir, "hybrid_head_ranking.json"), "w") as f:
        json.dump([
            {"layer": l, "head": h, "hybrid_score": s, "trigger_score": ts, "clean_score": cs}
            for l, h, s, ts, cs in ranked
        ], f, indent=2)
    print(f"[prune] kept pruned_heads={best_heads}; best_val_ACC={best_acc:.4f}")
    return model, best_heads


# -----------------------
# Experiment runner
# -----------------------
def load_data(args):
    data_root = os.path.join(args.helper_dir, "Data_Poisoning", args.attack, args.dataset)
    test_p_name = "test_poisoned_all.csv" if args.data_variant == "poisoned_all" else "test_poisoned_part.csv"
    paths = {
        "train_poisoned": os.path.join(data_root, "train_poisoned.csv"),
        "train_clean": os.path.join(data_root, "train_clean.csv"),
        "test_clean": os.path.join(data_root, "test_clean.csv"),
        "test_poisoned": os.path.join(data_root, test_p_name),
    }
    for name, path in paths.items():
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing {name}: {path}")
    return {name: read_csv(path) for name, path in paths.items()}


def evaluate_and_print(model, clean_loader, poison_loader, device, dataset, tag) -> Dict[str, Dict[str, float]]:
    clean = eval_acc_lfr(model, clean_loader, device, dataset)
    poison = eval_acc_lfr(model, poison_loader, device, dataset)
    print(metric_line(f"[{tag}] Clean", clean, dataset))
    print(metric_line(f"[{tag}] Triggered", poison, dataset))
    return {"clean": clean, "triggered": poison}


def run_single_defense(args, defense_name, model, poisoned_state, base_state, loaders, tokenizer, triggers, device) -> Dict:
    model.load_state_dict(poisoned_state, strict=False)
    tag = defense_name.upper()
    start = time.time()
    pruned_heads: Dict[int, List[int]] = {}

    if defense_name in ["prune", "hybrid"]:
        model, pruned_heads = trigger_aware_prune(
            model=model,
            train_clean_loader=loaders["train_clean"],
            val_loader=loaders["test_clean"],
            tokenizer=tokenizer,
            triggers=triggers,
            device=device,
            dataset=args.dataset,
            acc_threshold=args.acc_threshold,
            prune_step=args.prune_step,
            alpha_clean=args.alpha_clean,
            max_score_batches=args.max_score_batches,
            out_dir=args.out_dir,
        )
        if defense_name == "prune" and args.finetune_after_prune_epochs > 0:
            model = train_supervised(model, loaders["train_clean"], device, args.lr_unlearn,
                                     args.finetune_after_prune_epochs, args.weight_decay,
                                     args.max_grad_norm, args.grad_accum, args.fp16, args.bf16)

    if defense_name in ["ticu", "hybrid"]:
        model = ticu_unlearn(
            model=model,
            base_state=base_state,
            clean_loader=loaders["train_clean"],
            tokenizer=tokenizer,
            triggers=triggers,
            device=device,
            lr=args.lr_unlearn,
            epochs=args.epochs_unlearn,
            gamma=args.ticu_gamma,
            delta=args.ticu_delta,
            beta=args.beta,
            weight_decay=args.weight_decay,
            max_grad_norm=args.max_grad_norm,
            grad_accum=args.grad_accum,
            fp16=args.fp16,
            bf16=args.bf16,
        )

    metrics = evaluate_and_print(model, loaders["test_clean"], loaders["test_poison"], device, args.dataset, tag)
    ckpt_path = os.path.join(args.out_dir, f"{args.model_type}_{args.dataset}_{args.attack}_{defense_name}.pt")
    torch.save(model.state_dict(), ckpt_path)
    return {"defense": defense_name, "seconds": time.time() - start, "checkpoint": ckpt_path,
            "pruned_heads": pruned_heads, "metrics": metrics}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--helper_dir", type=str, required=True)
    ap.add_argument("--dataset", type=str, required=True, choices=["SST-2", "HSOL", "AG"])
    ap.add_argument("--attack", type=str, required=True, choices=["BadNet", "AddSent", "HiddenKiller", "StyleBkd"])
    ap.add_argument("--data_variant", type=str, default="poisoned_all", choices=["poisoned_all", "poisoned_part"])
    ap.add_argument("--model_type", type=str, default="auto", choices=["auto", "llama", "qwen", "mistral"])
    ap.add_argument("--model_name", type=str, default="bert-base-uncased")
    ap.add_argument("--use_lora", action="store_true")
    ap.add_argument("--lora_r", type=int, default=8)
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--load_in_8bit", action="store_true")
    ap.add_argument("--load_in_4bit", action="store_true")
    ap.add_argument("--grad_checkpoint", action="store_true")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs_poison", type=int, default=5)
    ap.add_argument("--lr_poison", type=float, default=2e-5)
    ap.add_argument("--epochs_unlearn", type=int, default=10)
    ap.add_argument("--lr_unlearn", type=float, default=2e-5)
    ap.add_argument("--ticu_gamma", type=float, default=1.0)
    ap.add_argument("--ticu_delta", type=float, default=0.1)
    ap.add_argument("--beta", type=float, default=0.05)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--trigger_mode", type=str, default="auto", choices=["auto", "manual"])
    ap.add_argument("--trigger", type=str, default="")
    ap.add_argument("--trigger_multi_sep", type=str, default="|")
    ap.add_argument("--defense", type=str, default="hybrid", choices=["ticu", "prune", "hybrid", "compare"])
    ap.add_argument("--acc_threshold", type=float, default=0.85)
    ap.add_argument("--prune_step", type=int, default=2)
    ap.add_argument("--alpha_clean", type=float, default=0.5, help="Penalty for pruning clean-important heads")
    ap.add_argument("--max_score_batches", type=int, default=50, help="0 means use all batches for head scoring")
    ap.add_argument("--finetune_after_prune_epochs", type=int, default=5)
    ap.add_argument("--out_dir", type=str, default="./prune_ticu_outputs")
    ap.add_argument("--skip_poison_train", action="store_true")
    ap.add_argument("--resume_path", type=str, default="")
    args = ap.parse_args()

    if args.fp16 and args.bf16:
        args.fp16 = False
        print("[warn] both --fp16 and --bf16 set; using bf16")

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    dfs = load_data(args)
    spec = task_spec(args.dataset)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token if getattr(tokenizer, "eos_token_id", None) is not None else "[PAD]"

    model = build_model(args.model_type, args.model_name, spec.num_labels, args=args)
    if args.model_type == "auto" and getattr(model, "resize_token_embeddings", None) is not None:
        model.resize_token_embeddings(len(tokenizer))
    model = maybe_to_device(model, device)

    loaders = {
        "train_poison": make_loader(CSVDataset(dfs["train_poisoned"], tokenizer, args.max_length), tokenizer, args.batch_size, True),
        "train_clean": make_loader(CSVDataset(dfs["train_clean"], tokenizer, args.max_length), tokenizer, args.batch_size, True),
        "test_clean": make_loader(CSVDataset(dfs["test_clean"], tokenizer, args.max_length), tokenizer, args.batch_size, False),
        "test_poison": make_loader(CSVDataset(dfs["test_poisoned"], tokenizer, args.max_length), tokenizer, args.batch_size, False),
    }

    if args.trigger_mode == "auto":
        triggers = infer_trigger_autoselect(dfs["train_poisoned"], args.attack)
    else:
        triggers = [t.strip() for t in args.trigger.split(args.trigger_multi_sep) if t.strip()]
        if not triggers:
            raise ValueError("--trigger_mode manual requires --trigger")
    print("triggers:", triggers)

    base_state = clone_state(model)
    poisoned_ckpt = os.path.join(args.out_dir, f"{args.model_type}_{args.dataset}_{args.attack}_poisoned.pt")
    if args.skip_poison_train:
        if not args.resume_path:
            raise ValueError("--skip_poison_train requires --resume_path")
        model.load_state_dict(torch.load(args.resume_path, map_location="cpu"), strict=False)
        print("Loaded poisoned checkpoint:", args.resume_path)
    else:
        print("\n== Train poisoned model ==")
        model = train_supervised(model, loaders["train_poison"], device, args.lr_poison, args.epochs_poison,
                                 args.weight_decay, args.max_grad_norm, args.grad_accum, args.fp16, args.bf16)
        torch.save(model.state_dict(), poisoned_ckpt)
        print("Saved poisoned checkpoint:", poisoned_ckpt)

    poisoned_state = copy.deepcopy(model.state_dict())
    before = evaluate_and_print(model, loaders["test_clean"], loaders["test_poison"], device, args.dataset, "BEFORE")

    defenses = ["ticu", "prune", "hybrid"] if args.defense == "compare" else [args.defense]
    results = {"before": before, "runs": []}
    for defense in defenses:
        print(f"\n== Run defense: {defense} ==")
        # Fresh model instance avoids accumulated prune graph/metadata between compare runs.
        run_model = build_model(args.model_type, args.model_name, spec.num_labels, args=args)
        if args.model_type == "auto" and getattr(run_model, "resize_token_embeddings", None) is not None:
            run_model.resize_token_embeddings(len(tokenizer))
        run_model = maybe_to_device(run_model, device)
        results["runs"].append(run_single_defense(args, defense, run_model, poisoned_state, base_state,
                                                   loaders, tokenizer, triggers, device))

    result_path = os.path.join(args.out_dir, "prune_ticu_results.json")
    with open(result_path, "w") as f:
        json.dump(results, f, indent=2)
    print("Saved results:", result_path)


if __name__ == "__main__":
    main()
