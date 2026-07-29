import re
from typing import Dict, List, Tuple

import torch

import config

_CACHE: Dict[str, Tuple] = {}


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_model(model_id: str):
    if model_id in _CACHE:
        return _CACHE[model_id]

    if _CACHE:
        import gc

        for k in list(_CACHE.keys()):
            _, old, _ = _CACHE.pop(k)
            del old
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    from transformers import AutoTokenizer, AutoModelForCausalLM

    device = get_device()
    print(f"\n[HF] Loading {model_id}  (device={device}, 4bit={config.LOAD_IN_4BIT})")

    if config.HF_TOKEN:
        from huggingface_hub import login

        login(token=config.HF_TOKEN, add_to_git_credential=False)

    load_kw = {"trust_remote_code": True}
    if device == "cuda":
        load_kw["device_map"] = "auto"
        load_kw["attn_implementation"] = "sdpa"
        if config.LOAD_IN_4BIT:
            from transformers import BitsAndBytesConfig

            load_kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        else:
            load_kw["dtype"] = torch.bfloat16
    else:
        load_kw["dtype"] = torch.float32
        load_kw["attn_implementation"] = "eager"

    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kw)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()

    print(f"[HF] Ready: {model_id}  (vocab={tokenizer.vocab_size:,})")
    _CACHE[model_id] = (tokenizer, model, device)
    return tokenizer, model, device


def build_prompt(tokenizer, system: str, user: str) -> str:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


_END_TOKENS = (
    "<|EOT|>",
    "<|im_end|>",
    "<|endoftext|>",
    "<｜end▁of▁sentence｜>",
)


def _eos_ids(tokenizer) -> List[int]:
    ids = set()
    if tokenizer.eos_token_id is not None:
        ids.add(int(tokenizer.eos_token_id))
    for tok in _END_TOKENS:
        try:
            tid = tokenizer.convert_tokens_to_ids(tok)
        except Exception:
            tid = None
        if tid is None or tid < 0:
            continue
        try:
            if tokenizer.convert_ids_to_tokens(tid) == tok:
                ids.add(int(tid))
        except Exception:
            pass
    return sorted(ids) if ids else [int(tokenizer.eos_token_id)]


def _entropy_from_logits(z: torch.Tensor) -> float:
    z = z.float()
    p = torch.softmax(z, dim=-1)
    return float(-(p * torch.log2(p + 1e-12)).sum().item())


def _energy_from_logits(z: torch.Tensor) -> float:
    return float(-torch.logsumexp(z.float(), dim=-1).item())


def reduce_metric(metrics: Dict[str, float], gate: str, reduction: str) -> float:
    key = f"{gate}_{reduction}"
    if key not in metrics:
        raise KeyError(f"metric '{key}' not in {sorted(metrics)}")
    return float(metrics[key])


_SQL_START = re.compile(r"\b(WITH|SELECT)\b", re.IGNORECASE)


def _clean_sql(raw: str) -> str:
    s = raw.strip()

    fenced = re.search(r"```(?:sql|sqlite)?\s*(.+?)```", s, re.IGNORECASE | re.DOTALL)
    if fenced:
        s = fenced.group(1).strip()
    else:
        s = s.replace("```", " ")

    m = _SQL_START.search(s)
    if m:
        s = s[m.start() :]

    s = s.split(";")[0]

    return s.strip().rstrip(";").strip()


def _gen_kwargs(tokenizer) -> Dict:
    return dict(
        max_new_tokens=config.MAX_NEW_TOKENS,
        do_sample=False,
        repetition_penalty=config.REPETITION_PENALTY,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=_eos_ids(tokenizer),
        return_dict_in_generate=True,
        output_logits=True,
    )


def generate_with_uncertainty(
    model, tokenizer, prompt: str, device: str
) -> Tuple[str, Dict[str, float]]:
    enc = tokenizer(
        prompt, return_tensors="pt", truncation=True, max_length=config.MAX_SEQ_LEN
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    kw = _gen_kwargs(tokenizer)
    with torch.no_grad():
        try:
            out = model.generate(**enc, **kw)
        except (TypeError, ValueError):
            kw.pop("output_logits", None)
            kw["output_scores"] = True
            out = model.generate(**enc, **kw)

    seq = out.sequences[0]
    new_ids = seq[enc["input_ids"].shape[1] :]
    raw = tokenizer.decode(
        new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
    )

    raw = raw.replace("Ċ", "\n").replace("Ġ", " ")

    sql = _clean_sql(raw)

    step_logits = getattr(out, "logits", None)
    if step_logits is None:
        step_logits = getattr(out, "scores", None)

    ent: List[float] = []
    en: List[float] = []
    if step_logits:
        for lg in step_logits:
            z = lg[0]
            ent.append(_entropy_from_logits(z))
            en.append(_energy_from_logits(z))
    if not ent:
        ent, en = [0.0], [0.0]

    n = len(ent)
    metrics = {
        "entropy_first": ent[0],
        "entropy_mean": sum(ent) / n,
        "entropy_max": max(ent),
        "energy_first": en[0],
        "energy_mean": sum(en) / n,
        "energy_max": max(en),
        "n_tokens": n,
    }
    return sql, metrics
