
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
from honyak.inference_utils import translate_to_toki_pona, compute_cross_entropy_loss

MODEL_PATH = "../training/models/t5-honyak-10-epoch"
TEST_PREDS = 30

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
model.eval()

# Load test data
test_data = pd.read_csv(
    "../data/test.tsv",
    sep="\t",
    names=["english", "toki_pona"]
)[:TEST_PREDS]

preds = []
losses = []

for i, row in test_data.iterrows():
    pred = translate_to_toki_pona(model, tokenizer, row["english"])
    loss = compute_cross_entropy_loss(model, tokenizer, row["english"], row["toki_pona"])

    preds.append(pred)
    losses.append(loss)

    if (i + 1) % 10 == 0:
        print(f"Processed {i + 1}/{len(test_data)} examples")

test_data["predicted_tp"] = preds
test_data["cross_entropy_loss"] = losses

test_data.to_csv(
    "test_preds.tsv",
    index=False,
    sep="\t"
)

print("All examples processed.")
