import torch

def translate_to_toki_pona(model, tokenizer, sentence, max_len=40, num_beams=5):

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

def compute_cross_entropy_loss(model, tokenizer, english, gold_tp):
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