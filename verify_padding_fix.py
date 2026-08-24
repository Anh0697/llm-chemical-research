"""
Verification script: confirms that the padding_side fix removed the
batch-size dependence of extracted activations.

Method: for a handful of sample elements, run a fresh forward pass with
batch_size=1 (no other prompt can possibly share padding with it -- this is
the cleanest possible ground truth) and compare against the corresponding
row already saved in the just-completed batch_size=16 run. Reuses the
actual load_tokenizer / load_model / get_and_process_activations functions
from the fixed extract_activations.py, so the comparison is apples-to-apples
with the real pipeline rather than a hand-rewritten copy.

Expected outcome if the fix worked: max abs diff on the order of 1e-3 or
smaller (ordinary GPU floating-point noise).
If the fix did not work: max abs diff on the order of 1e-2 to 1e-1, similar
to what was observed before the fix.
"""

import os
import sys
import yaml
import torch
import pandas as pd

sys.path.insert(0, "Pre")
from extract_activations import load_tokenizer, load_model, get_and_process_activations

LAYER_TO_CHECK = 20
SAMPLE_SYMBOLS = ["H", "O", "Fe", "Au", "U"]
ENTITY_TYPE = "atomic number_single"
PROMPT_NAME = "1_templates"
TEMPLATE = "In the periodic table, the atomic number of element {Symbol}"
CSV_PATH = "periodic_table_dataset.csv"
CONFIG_PATH = "config_extract_activation.yaml"


def main():
    with open(CONFIG_PATH, "r") as f:
        config_data = yaml.safe_load(f)

    hf_token = config_data.get("HF_TOKEN")
    extraction_config = config_data.get("extraction", {})
    model_name = extraction_config.get("model_name")
    save_dir = extraction_config.get("save_dir", "activation_datasets_2")
    aggregation = extraction_config.get("aggregation", "last")

    saved_pt_path = os.path.join(
        save_dir,
        model_name.replace("/", "-"),
        ENTITY_TYPE,
        f"{ENTITY_TYPE}.{aggregation}.{PROMPT_NAME}.layer_{LAYER_TO_CHECK}.pt",
    )
    print(f"Loading saved activations from: {saved_pt_path}")
    saved_activations = torch.load(saved_pt_path, map_location="cpu")
    print(f"Saved activations shape: {saved_activations.shape}")

    tokenizer = load_tokenizer(model_name, hf_token)
    model = load_model(model_name, hf_token, extraction_config.get("quantization", {}))
    model.eval()

    df = pd.read_csv(CSV_PATH)

    print(f"\nComparing batch_size=1 (ground truth) vs the just-completed batch_size=16 run, layer {LAYER_TO_CHECK}\n")
    print(f"{'Symbol':<8}{'Row idx':<10}{'Max abs diff':<16}{'Mean abs diff':<16}")

    for symbol in SAMPLE_SYMBOLS:
        matches = df.index[df["Symbol"] == symbol].tolist()
        if not matches:
            print(f"{symbol}: not found in {CSV_PATH}, skipping.")
            continue
        row_idx = matches[0]
        row = df.iloc[row_idx]
        prompt = TEMPLATE.format(**row.to_dict())

        processed = get_and_process_activations(model, tokenizer, [prompt], aggregation)
        fresh_act = processed[f"layer_{LAYER_TO_CHECK}"].squeeze(0).float().cpu()
        saved_act = saved_activations[row_idx].float()

        diff = (fresh_act - saved_act).abs()
        print(f"{symbol:<8}{row_idx:<10}{diff.max().item():<16.6f}{diff.mean().item():<16.6f}")

    print("\nIf max abs diff is on the order of 1e-3 or smaller, the fix is confirmed: batch size no longer affects the extracted activations.")
    print("If max abs diff is on the order of 1e-2 to 1e-1, something is still wrong and needs further investigation.")


if __name__ == "__main__":
    main()
