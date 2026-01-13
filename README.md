
# Honyak

Toki!

This project fine-tunes a small sequence-to-sequence language model (T5-small)
to translate English sentences into Toki Pona, a constructed language with a
minimal vocabulary. The model is trained using the Hugging Face Transformers
framework, with an emphasis on reproducibility and clean software engineering
practices.


## Folder structure

- `data/`: preprocessing scripts & sample datasets  
- `training/`: Jupyter notebook for training the model  
- `inference/`: scripts for generating translations and evaluation  
- `demo.ipynb`: example workflow


## Setup

Create environment:

```bash
conda env create -f environment.yaml
conda activate honyak
```

---

# Fine-tuning T5-small to learn a new language

{write up}