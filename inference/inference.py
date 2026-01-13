# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
# import pandas as pd

# MODEL_PATH = "../training/models/t5-honyak-10-epoch" 
# tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
# model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)

# def translate_to_toki_pona(sentence, max_len=40, num_beams=5):
#     prompt = f"translate English to Toki Pona: {sentence}"
#     inputs = tokenizer(prompt, return_tensors="pt")

#     outputs = model.generate(
#         **inputs,
#         max_length=max_len,
#         num_beams=num_beams,
#         early_stopping=True
#     )

#     return tokenizer.decode(outputs[0], skip_special_tokens=True)

# # Apply test examples
# test_data = pd.read_csv("../data/test.tsv", sep="\t", names=["english", "toki_pona"])

# preds = []
# i = 0
# for s in test_data["english"]:
#     preds.append(translate_to_toki_pona(s))
#     i += 1
#     if (i + 1) % 100 == 0:
#         print(f"Processed {i + 1}/{len(test_data)} examples")

# test_data["predicted_tp"] = preds
# test_data.to_csv("data/test_preds.tsv", index=False, sep="\t")

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
import torch

MODEL_PATH = "../training/models/t5-honyak-10-epoch"
TEST_PREDS = 30

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
model.eval()

def translate_to_toki_pona(sentence, max_len=40, num_beams=5):
    prompt = f"translate English to Toki Pona: {sentence}"
    inputs = tokenizer(prompt, return_tensors="pt")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_len,
            num_beams=num_beams,
            early_stopping=True
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def compute_cross_entropy_loss(english, gold_tp):
    prompt = f"translate English to Toki Pona: {english}"

    inputs = tokenizer(prompt, return_tensors="pt")
    labels = tokenizer(gold_tp, return_tensors="pt").input_ids

    with torch.no_grad():
        outputs = model(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            labels=labels
        )

    # This is mean token-level cross-entropy
    return outputs.loss.item()

# Load test data
test_data = pd.read_csv(
    "../data/test.tsv",
    sep="\t",
    names=["english", "toki_pona"]
)[:TEST_PREDS]

preds = []
losses = []

for i, row in test_data.iterrows():
    pred = translate_to_toki_pona(row["english"])
    loss = compute_cross_entropy_loss(row["english"], row["toki_pona"])

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
