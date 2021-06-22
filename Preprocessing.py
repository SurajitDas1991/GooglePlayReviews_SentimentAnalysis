import transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import torch

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rc
#from pylab import rcParams
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from collections import defaultdict
from textwrap import wrap

from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

df = pd.read_csv("reviews.csv")
df.head()
print(df.shape)
print(df.info())

#Balanced or Unbalanced Dataset
# We need balanced dataset for negative and positive sentiment
sns.countplot(df.score)
plt.xlabel('review score')


#Sentiment values 0 - Negative , 1 - Neutral , 2 - Positive
def to_sentiment(rating):
  rating = int(rating)
  if rating <= 2:
    return 0
  elif rating == 3:
    return 1
  else: 
    return 2

#Create a new column - sentiment
df['sentiment'] = df.score.apply(to_sentiment)

# Check count of data categories of new column sentiment
class_names = ['negative', 'neutral', 'positive']
ax = sns.countplot(df.sentiment)
plt.xlabel('review sentiment')
ax.set_xticklabels(class_names)
#plt.show()

#Data Preprocessing

#Tokenizer from BERT , also takes care of special token to mark the start and end of sentences 
tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-uncased')

# SAMPLE TEXT FOR BETTER UNDERSTANDING

sample_text='When was I last outside ? I am stuck at home for 2 weeks. '
tokens=tokenizer.tokenize(sample_text)
print(tokens)

tokens_ids=tokenizer.convert_tokens_to_ids(tokens)
print(tokens_ids)

#Special Tokens - Separation , CLS , Pad token , unknown token
print(tokenizer.sep_token,tokenizer.sep_token_id)
print(tokenizer.cls_token,tokenizer.cls_token_id)
print(tokenizer.pad_token,tokenizer.pad_token_id)
print(tokenizer.unk_token,tokenizer.unk_token_id)

encoding = tokenizer.encode_plus(
  sample_text,
  truncation=True,
  max_length=32,
  add_special_tokens=True, # Add '[CLS]' and '[SEP]'
  return_token_type_ids=False,
  pad_to_max_length=True,
  return_attention_mask=True,
  return_tensors='pt',  # Return PyTorch tensors
)

print(encoding.keys())

print(len(encoding['input_ids'][0]))
print(encoding['input_ids'][0])


print(len(encoding['attention_mask'][0]))
print(encoding['attention_mask'])

print(tokenizer.convert_ids_to_tokens(encoding['input_ids'][0]))

# Choosing Sequence Length
token_lens=[]

for text in df.content:
    tokens=tokenizer.encode(str(text),max_length=512,truncation=True)
    token_lens.append(len(tokens))

sns.displot(token_lens)
plt.xlim([0, 300])
plt.xlabel('Token count')
plt.show()
