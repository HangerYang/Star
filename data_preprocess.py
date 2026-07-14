import argparse
import json
import random
from pathlib import Path


# ---------------------------------------------------------------------------
# ShareGPT
# ---------------------------------------------------------------------------

def convert_sharegpt(input_path, output_path, max_samples=None):
    """
    ShareGPT format:
        {"id": ..., "conversations": [{"from": "human"/"gpt", "value": "..."}]}
    -> target format:
        {"id": ..., "conversations": [{"role": "user"/"assistant", "content": "..."}]}
    """
    role_map = {
        "human": "user",
        "gpt": "assistant",
        "user": "user",
        "assistant": "assistant",
        "system": "system",
    }

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    count = 0
    with open(output_path, "w", encoding="utf-8") as out:
        for item in data:
            convs = item.get("conversations", [])
            new_convs = []
            for turn in convs:
                role = role_map.get(str(turn.get("from", "")).lower())
                content = str(turn.get("value", "")).strip()
                if role is None or not content:
                    continue
                new_convs.append({"role": role, "content": content})

            new_convs = _clean_conversation(new_convs)
            if new_convs is None:
                continue

            record = {"id": str(item.get("id", count)), "conversations": new_convs}
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

            if max_samples and count >= max_samples:
                break

    print(f"[ShareGPT] wrote {count} conversations -> {output_path}")
    return count


# ---------------------------------------------------------------------------
# UltraChat-200k
# ---------------------------------------------------------------------------

def _load_ultrachat_records(input_path):
    """Yield raw dict records from either a .parquet or .jsonl UltraChat file."""
    suffix = Path(input_path).suffix.lower()

    if suffix == ".parquet":
        import pandas as pd  # requires: pip install pandas pyarrow

        df = pd.read_parquet(input_path)
        for row in df.to_dict(orient="records"):
            yield row
    else:
        # assume jsonl
        with open(input_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)


def convert_ultrachat(input_path, output_path, max_samples=None, start_id=0):
    """
    UltraChat-200k format:
        {"prompt": ..., "prompt_id": ..., "messages": [{"role": "user"/"assistant", "content": "..."}]}
    -> target format:
        {"id": ..., "conversations": [{"role": "user"/"assistant", "content": "..."}]}
    """
    count = 0
    with open(output_path, "w", encoding="utf-8") as out:
        for item in _load_ultrachat_records(input_path):
            messages = item.get("messages", [])
            new_convs = []
            for turn in messages:
                role = str(turn.get("role", "")).lower()
                content = str(turn.get("content", "")).strip()
                if role not in ("user", "assistant", "system") or not content:
                    continue
                new_convs.append({"role": role, "content": content})

            new_convs = _clean_conversation(new_convs)
            if new_convs is None:
                continue

            record = {"id": str(start_id + count), "conversations": new_convs}
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

            if max_samples and count >= max_samples:
                break

    print(f"[UltraChat] wrote {count} conversations -> {output_path}")
    return count


# ---------------------------------------------------------------------------
# Shared cleaning / merge utilities
# ---------------------------------------------------------------------------

def _clean_conversation(conversations, min_turns=2, min_chars=1, max_chars=8000):
    """
    Basic cleaning applied to both sources:
      - drop empty/near-empty turns (already filtered upstream)
      - ensure conversation starts with a 'user' turn
      - drop conversations that are too short (< min_turns)
      - drop turns that are absurdly long (likely junk/HTML dumps)
    Returns cleaned list, or None if the conversation should be dropped.
    """
    if not conversations:
        return None

    # Drop leading non-user turns (e.g. stray system/assistant-first artifacts)
    while conversations and conversations[0]["role"] != "user":
        conversations = conversations[1:]

    if len(conversations) < min_turns:
        return None

    cleaned = []
    for turn in conversations:
        content = turn["content"]
        if len(content) < min_chars or len(content) > max_chars:
            continue
        cleaned.append(turn)

    if len(cleaned) < min_turns:
        return None

    return cleaned


def merge_and_shuffle(files, output_path, seed=42):
    lines = []
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            lines.extend(f.readlines())

    random.seed(seed)
    random.shuffle(lines)

    with open(output_path, "w", encoding="utf-8") as out:
        for i, line in enumerate(lines):
            record = json.loads(line)
            record["id"] = str(i)  # re-assign sequential ids after merge
            out.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"[Merge] wrote {len(lines)} total conversations -> {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Preprocess ShareGPT + UltraChat-200k for AngelSlim Eagle3 training data"
    )
    parser.add_argument("--sharegpt_path", type=str, default=None,
                         help="Path to ShareGPT JSON file "
                              "(e.g. ShareGPT_V3_unfiltered_cleaned_split.json)")
    parser.add_argument("--ultrachat_path", type=str, default=None,
                         help="Path to UltraChat-200k file, .parquet or .jsonl "
                              "(e.g. train_sft-00000-of-00003.parquet)")
    parser.add_argument("--output_dir", type=str, default="./processed_data")
    parser.add_argument("--max_sharegpt", type=int, default=None,
                         help="Cap number of ShareGPT conversations (None = all)")
    parser.add_argument("--max_ultrachat", type=int, default=None,
                         help="Cap number of UltraChat conversations (None = all)")
    parser.add_argument("--merge", action="store_true",
                         help="Merge and shuffle both outputs into one training file")
    args = parser.parse_args()

    if not args.sharegpt_path and not args.ultrachat_path:
        parser.error("Provide at least one of --sharegpt_path or --ultrachat_path")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs = []

    if args.sharegpt_path:
        sg_out = out_dir / "sharegpt_converted.jsonl"
        convert_sharegpt(args.sharegpt_path, sg_out, max_samples=args.max_sharegpt)
        outputs.append(sg_out)

    if args.ultrachat_path:
        uc_out = out_dir / "ultrachat_converted.jsonl"
        convert_ultrachat(args.ultrachat_path, uc_out, max_samples=args.max_ultrachat)
        outputs.append(uc_out)

    if args.merge:
        if len(outputs) < 2:
            print("[Merge] skipped: need both --sharegpt_path and --ultrachat_path to merge")
        else:
            merged_out = out_dir / "merged_train_data.jsonl"
            merge_and_shuffle(outputs, merged_out)


if __name__ == "__main__":
    main()