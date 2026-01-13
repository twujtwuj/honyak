This folder contains the raw (from tatoeba.org) and processed datasets for English → Toki Pona translation.

---

- `raw_sentence_pairs_2024_02_20.tsv` – Raw sentence pairs (TSV: `[id1, sentence_en, id2, sentence_tp]`)  
- `data_preprocessing.py` – Cleans, shuffles, and splits raw data  
- `train.tsv` – 80% of data for training (`[sentence_en, sentence_tp]`)  
- `val.tsv` – 10% for validation  
- `test.tsv` – 10% for testing  