import pandas as pd
from sklearn.model_selection import train_test_split

# Load raw sentences (taken from tatoeba.org)
df = pd.read_csv("raw_sentence_pairs_2024_02_20.tsv", sep="\t", header=None)

# Remove numerical tags and keep just sentences
df = df[[1, 3]]

# Shuffle
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Compute split sizes (80/10/10)
n = len(df)
train_end = int(0.8 * n)
val_end = int(0.9 * n)
test_end = int(1.0 * n)

# Split
train_df = df.iloc[:train_end]
val_df = df.iloc[train_end:val_end]
test_df = df.iloc[val_end:test_end]

# Save
train_df.to_csv("train.tsv", sep="\t", header=False, index=False)
val_df.to_csv("val.tsv", sep="\t", header=False, index=False)
test_df.to_csv("test.tsv", sep="\t", header=False, index=False)
