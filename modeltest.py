import pandas as pd

train_df = pd.read_csv("new_train.csv", index_col=0)
test_df = pd.read_csv("new_test.csv", index_col=0)

print("Train size", len(train_df))
print("Test size", len(test_df))
train_df.head(n=3)