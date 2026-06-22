import pandas as pd
from datasets import load_dataset

dataset = load_dataset("bigbio/bc5cdr",trust_remote_code=True)

mt_data=pd.read_csv(r'C:\Users\hp\Desktop\med\mtsamples.csv\mtsamples.csv')
# print(mt_data.head(1))
# print(dataset)

# print(dataset["train"][0])

# print(dataset["train"][0].keys())

# print(dataset["train"][0]["passages"][0]["text"])

sample=dataset['train'][0]['passages']

for passage in sample:
    print(passage['type'])
    print(passage['text'])