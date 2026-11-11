"""Build the token files once: FineWeb-Edu (sample-10BT) -> 16k byte-level BPE -> uint16.

Outputs (on the datastore mount):  <out_root>/data/{tokenizer.json, train.bin, val.bin, DATA_READY.json}
Idempotent: if DATA_READY.json exists the job exits immediately.
Works locally first (/tmp) and copies finished files to storage, so a preempted prep
job simply restarts from scratch (it takes minutes, not hours).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time

import numpy as np


def sha1(path, n=1 << 24):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        h.update(f.read(n))
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--train_tokens", type=int, default=120_000_000)
    ap.add_argument("--val_tokens", type=int, default=2_000_000)
    ap.add_argument("--vocab", type=int, default=16384)
    ap.add_argument("--tok_docs", type=int, default=60_000)
    ap.add_argument("--dataset", default="HuggingFaceFW/fineweb-edu")
    ap.add_argument("--subset", default="sample-10BT")
    a = ap.parse_args()

    out = os.path.join(a.out_root, "data")
    if os.path.exists(os.path.join(out, "DATA_READY.json")):
        print("DATA_READY exists -> nothing to do")
        return
    os.makedirs(out, exist_ok=True)
    work = "/tmp/moe_prep"
    os.makedirs(work, exist_ok=True)

    from datasets import load_dataset
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

    token = os.environ.get("HF_TOKEN") or None   # dataset is public; token optional
    ds = load_dataset(a.dataset, name=a.subset, split="train", streaming=True, token=token)
    it = iter(ds)

    # 1) tokenizer on the first tok_docs documents
    t0 = time.time()
    tok_texts = [next(it)["text"] for _ in range(a.tok_docs)]
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=a.vocab, special_tokens=["<|eos|>"],
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                                  show_progress=False)
    tok.train_from_iterator(tok_texts, trainer=trainer)
    eos = tok.token_to_id("<|eos|>")
    assert tok.get_vocab_size() <= 65535
    tok.save(os.path.join(work, "tokenizer.json"))
    print(f"tokenizer trained in {time.time() - t0:.0f}s, vocab={tok.get_vocab_size()}")
    del tok_texts

    # 2) val then train from the following (disjoint) documents
    def write(path, target):
        n, buf = 0, []
        with open(path, "wb") as f:
            while n < target:
                batch = []
                for _ in range(512):
                    try:
                        batch.append(next(it)["text"])
                    except StopIteration:
                        break
                if not batch:
                    break
                for enc in tok.encode_batch(batch):
                    ids = enc.ids + [eos]
                    arr = np.asarray(ids, dtype=np.uint16)
                    f.write(arr.tobytes())
                    n += len(ids)
                    if n >= target:
                        break
                print(f"  {os.path.basename(path)}: {n / 1e6:.1f}M tokens", flush=True)
        return n

    n_val = write(os.path.join(work, "val.bin"), a.val_tokens)
    n_train = write(os.path.join(work, "train.bin"), a.train_tokens)

    for fn in ("tokenizer.json", "val.bin", "train.bin"):
        shutil.copyfile(os.path.join(work, fn), os.path.join(out, fn + ".part"))
        os.replace(os.path.join(out, fn + ".part"), os.path.join(out, fn))
    meta = dict(dataset=a.dataset, subset=a.subset, vocab=a.vocab, eos=eos, n_train=n_train,
                n_val=n_val, train_sha1_16MB=sha1(os.path.join(out, "train.bin")),
                created=time.strftime("%Y-%m-%d %H:%M:%S"))
    with open(os.path.join(out, "DATA_READY.json.tmp"), "w") as f:
        json.dump(meta, f, indent=1)
    os.replace(os.path.join(out, "DATA_READY.json.tmp"), os.path.join(out, "DATA_READY.json"))
    print("DATA_READY", meta)


if __name__ == "__main__":
    main()
