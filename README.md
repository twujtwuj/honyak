
# Honyak 📝

Toki!

This project fine-tunes a small sequence-to-sequence model (T5-small) to translate English (EN) sentences into Toki Pona (TP), a constructed language with a ~120-word vocabulary. The model is trained using the Hugging Face Transformers framework (https://huggingface.co/docs/transformers/en/index
).

I also experiemented with using TensorBoard.

NEW TRACK: Try fine tuning (already done) and building from scratch, and compare!


## Fine-tuning T5-small to learn a new language 🤖🌐

### Background

Toki Pona (TP) is a constructed language created by Sonja Lang in 2001. With a minimal vocabulary of ~120 official words and only 14 phonemes, it is intentionally simple. Despite this, an active TP-speaking community online has produced thousands of sentences in the language's Latin script.


### Data set and base model
 
The Tatoeba project (tatoeba.org) provides ~28k English–Toki Pona sentence pairs. This project fine-tunes T5-small (60M parameters) on this dataset for EN → TP translation. The small size of T5-small allows training locally, avoiding the need to outsource compute. The goal was to assess whether a small-scale transformer could learn a minimalistic language.

The tokeniser used it T5TokenizerFast (from AutoTokenizer).

[Seq2Seq models like T5 are trained in the following way: {.......}]: #

### EN -> TP only

The focus is solely on EN → TP translation. TP is highly context-dependent and ambiguous: one word can have multiple meanings, resulting in a many-to-one mapping from EN to TP sentences. Translating TP → EN is therefore more subjective and requires larger models and more sophisticated training techniques.


### Tokens and context length

One thing to notice is that the tokens in the tokeniser, whose vocabulary is predetermined and does not change during training, will not contain TP words. Because of this, TP translations are formed of small, character-ish tokens that tend to contain between 2 to 3 times as many tokens as their corrsponding EN. I kept the maximum 'output' length at 32 (the same as the input) though, because if it is higher it makes training so much longer (and my computer runs out of memory).

### Training process

The training process followed Hugging Face documentation, with guidance from ChatGPT. Hyperparameters such as batch size and gradient accumulation were tuned experimentally.

- Training on 1% of the dataset for 3 epochs produced translations in German (and occasionally French), likely because the model recognized the task as translation but could not identify TP. T5 models are pre-trained on the C4 corpus, which is dominated by English (4T tokens), followed by Russian (T), Spanish (0.6T), and German (0.5T).

- Training on 10% of the dataset for 10 epochs yielded TP-only words, but they were often unrelated to the source sentence.

- Training on 100% of the dataset for 3 epochs produced reasonable translations with frequent grammatical errors.

- Training on 100% of the dataset for 10 epochs achieved high-quality translations. This is the current extent of training.

### Future areas for improvement


#### A: Tokiponisation

Tokiponisation converts non-TP names into TP equivalents. This process is largely algorithmic but may diverge from speaker conventions (https://jan-ne.github.io/tp/tpize
). The model has only learned Tokiponisation for names present in the dataset (e.g., Tom → jan Ton). For new names, it can approximate TP-like approximations (demo.ipynb). Incorporating a dataset of Tokiponified names or at least augmenting current datasets would improve performance.

#### B: Longer sentences

The model struggles with multi-part or long sentences. [{.......}]: #

#### C: Capitalisation and punctuation

The model has not learned TP capitalization and punctuation (. ? :), which are grammatically relevant. This is due to initial normalization of TP sentences, which removed these features.

#### D: New, more restrictive architecture?

I am currently experimenting with the idea of using a more customised architecture that takes in English tokens (~13k for the T5-series' tokeniser) and outputs  


### Conclusion

It is possible to fine-tune a small model to translate short EN sentences to TP within a few days, using local compute. The model often produces translations that are qualitatively better than those in the Tatoeba dataset.

Limitations include occasional grammatical errors, particularly for underrepresented structures, and limited utility due to TP's small user base.

---

# Repo structure 📁

## Folder structure

- `data/`: preprocessing scripts & sample datasets  
- `demos/`: contains example workflows
- `inference/`: scripts for generating translations and evaluation  
- `srs/`: Python package
- `training/`: Jupyter notebook for training the model  


## Setup

Create environment:

```bash
conda env create -f environment.yaml
conda activate honyak
```