"""Tokenize FineWeb-Edu (sample-10BT) with a 16k byte-level BPE into uint16 binary files."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import time

import numpy as np


def sha1_prefix(path: str, num_bytes: int = 1 << 24) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        h.update(f.read(num_bytes))
    return h.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_root", required=True)
    parser.add_argument("--train_tokens", type=int, default=120_000_000)
    parser.add_argument("--val_tokens", type=int, default=2_000_000)
    parser.add_argument("--vocab", type=int, default=16384)
    parser.add_argument("--tok_docs", type=int, default=60_000)
    parser.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    parser.add_argument("--subset", default="sample-10BT")
    args = parser.parse_args()

    out = os.path.join(args.out_root, "data")
    if os.path.exists(os.path.join(out, "DATA_READY.json")):
        print("DATA_READY.json exists; skipping data preparation")
        return

    os.makedirs(out, exist_ok=True)
    work = os.environ.get("MOE_PREP_TMP", os.path.join(tempfile.gettempdir(), "moe_prep"))
    os.makedirs(work, exist_ok=True)

    from datasets import load_dataset
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    token = os.environ.get("HF_TOKEN") or None
    ds = load_dataset(args.dataset, name=args.subset, split="train", streaming=True, token=token)
    stream = iter(ds)

    t0 = time.time()
    tok_texts = [next(stream)["text"] for _ in range(args.tok_docs)]
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=args.vocab,
        special_tokens=["<|eos|>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tok.train_from_iterator(tok_texts, trainer=trainer)
    eos = tok.token_to_id("<|eos|>")
    assert tok.get_vocab_size() <= 65535
    tok.save(os.path.join(work, "tokenizer.json"))
    print(f"trained tokenizer in {time.time() - t0:.0f}s (vocab={tok.get_vocab_size()})")
    del tok_texts

    def write_split(path: str, target_tokens: int) -> int:
        written = 0
        with open(path, "wb") as f:
            while written < target_tokens:
                batch = []
                for _ in range(512):
                    try:
                        batch.append(next(stream)["text"])
                    except StopIteration:
                        break
                if not batch:
                    break
                for enc in tok.encode_batch(batch):
                    ids = enc.ids + [eos]
                    arr = np.asarray(ids, dtype=np.uint16)
                    f.write(arr.tobytes())
                    written += len(ids)
                    if written >= target_tokens:
                        break
                print(f"  {os.path.basename(path)}: {written / 1e6:.1f}M tokens", flush=True)
        return written

    n_val = write_split(os.path.join(work, "val.bin"), args.val_tokens)
    n_train = write_split(os.path.join(work, "train.bin"), args.train_tokens)

    for fn in ("tokenizer.json", "val.bin", "train.bin"):
        shutil.copyfile(os.path.join(work, fn), os.path.join(out, fn + ".part"))
        os.replace(os.path.join(out, fn + ".part"), os.path.join(out, fn))

    meta = dict(
        dataset=args.dataset,
        subset=args.subset,
        vocab=args.vocab,
        eos=eos,
        n_train=n_train,
        n_val=n_val,
        train_sha1_16MB=sha1_prefix(os.path.join(out, "train.bin")),
        created=time.strftime("%Y-%m-%d %H:%M:%S"),
    )
    ready_tmp = os.path.join(out, "DATA_READY.json.tmp")
    with open(ready_tmp, "w") as f:
        json.dump(meta, f, indent=1)
    os.replace(ready_tmp, os.path.join(out, "DATA_READY.json"))
    print("DATA_READY", meta)


if __name__ == "__main__":
    main()
