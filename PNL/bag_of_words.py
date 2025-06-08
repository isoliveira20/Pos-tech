import pandas as pd

avaliacoes = pd.read_csv("b2w.csv")
avaliacoes.head()

avaliacoes = avaliacoes.drop(["original_index", "review_text_processed", "review_text_tokenized", "rating", "kfold_polarity", "kfold_rating"], axis=1)
avaliacoes.head()

avaliacoes.dropna(inplace=True, axis=0)
