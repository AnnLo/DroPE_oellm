"""Add a `doc_blocks` option to custom_data/pretraining_data.py (run once from the DroPE repo root).

doc_blocks=False (default): unchanged. All documents are concatenated with EOS and cut into
                            block_size blocks, so a block usually spans several documents.
doc_blocks=True:            each document (+EOS) is first cut into as many FULL block_size
                            windows as it has, each window becoming its own block (one document
                            per block, contiguous text). Only the leftover tails are packed
                            together the old way. With a long-document mix (docs >= 4096 tokens)
                            almost every block is a single contiguous document.
Enable per run config with:   make_dataset_fn: {doc_blocks: true}
"""
from pathlib import Path

p = Path("custom_data/pretraining_data.py")
src = p.read_text()
if "doc_blocks" in src:
    raise SystemExit("already patched")

old_sig = "        mask_past_sequences=False,\n"
assert src.count(old_sig) == 1
src = src.replace(old_sig, old_sig + "        doc_blocks=False,\n")

old = """        total = (len(ids_cat) // block_size) * block_size
        input_blocks = [
            ids_cat[i:i + block_size] for i in range(0, total, block_size)
        ]"""
new = """        if doc_blocks:
            # full windows of each document become their own blocks; only tails are packed
            full_ids, full_attn, rest_ids, rest_attn = [], [], [], []
            for ids, attn in zip(ex["input_ids"], ex["attention_mask"]):
                ids = list(ids) + ([eos_id] if eos_id is not None else [])
                attn = list(attn) + ([1] if eos_id is not None else [])
                n_full = len(ids) // block_size
                for j in range(n_full):
                    full_ids.append(ids[j * block_size:(j + 1) * block_size])
                    full_attn.append(attn[j * block_size:(j + 1) * block_size])
                rest_ids.extend(ids[n_full * block_size:])
                rest_attn.extend(attn[n_full * block_size:])
            total = (len(rest_ids) // block_size) * block_size
            input_blocks = full_ids + [rest_ids[i:i + block_size] for i in range(0, total, block_size)]
            attn_blocks = full_attn + [rest_attn[i:i + block_size] for i in range(0, total, block_size)]
            return {
                "input_ids": input_blocks,
                "attention_mask": attn_blocks,
                "labels": [b[:] for b in input_blocks],
            }

        total = (len(ids_cat) // block_size) * block_size
        input_blocks = [
            ids_cat[i:i + block_size] for i in range(0, total, block_size)
        ]"""
assert src.count(old) == 1, "unexpected group_texts layout"
src = src.replace(old, new)
p.write_text(src)
print("patched", p)