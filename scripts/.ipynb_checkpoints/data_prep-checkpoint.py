import pandas as pd
import re
import numpy as np
from nltk.tokenize import word_tokenize
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch
import torch.nn as nn
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from transformers import DistilBertTokenizer, DistilBertModel
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.utils.class_weight import compute_class_weight
from datetime import datetime
from sklearn.metrics import f1_score
import pickle

np.random.seed(1)

df = pd.read_csv("/home/ubuntu/Development/punctuation/data/transcripts.csv")
# len(df): 2467

# train:val:test = 0.8:0.1:0.1
num_train = int(len(df)*0.8)+2 # 0.8 -> 0.01
num_val = int(len(df)*0.1) # 0.1 -> 0.01
num_test = int(len(df)*0.1) # 0.1 -> 0.01

# assign indices
id_all = np.random.choice(len(df), len(df), replace=False)
id_train = id_all[0:num_train]
id_val = id_all[num_train : num_val+num_train]
id_test = id_all[num_val+num_train : num_val+num_train+num_test]

# actual split
train_set = df.iloc[id_train]
val_set = df.iloc[id_val]
test_set = df.iloc[id_test]

# remove transcripts containing ♫
train_set = train_set[~train_set['transcript'].str.contains('♫')]
val_set = val_set[~val_set['transcript'].str.contains('♫')]
test_set = test_set[~test_set['transcript'].str.contains('♫')]

def data_prep(data_set):

    # Dataset Cleanup
    data_set = data_set.drop('url',axis=1)
    data_set = data_set['transcript']
    data_set = data_set.str.replace("\(.*?\)", " ")\
    .str.replace("\[.*?\]", " ")\
    .str.replace(";", ". ")\
    .str.replace(":", ". ")\
    .str.replace('"', ' ')\
    .str.replace('!', '. ')\
    .str.replace(" — (?=[a-z])", ", ")\
    .str.replace(" — (?=[A-Z])", ". ")\
    .str.replace("(?<=[a-z])\.(?=[A-Z])", ". ")\
    .str.replace("(?<=[a-z])\?(?=[A-Z])", ". ")\
    .str.replace("(?<= )'(?=[a-zA-Z])", " ")\
    .str.replace("(?<=[a-z])\'(?= )", " ")\
    .str.replace("\'(?= )", " ")\
    .str.replace(" — ", " ")\
    .str.replace('\.+', '.')\
    .str.replace(' +', ' ')\
    .str.lower()
    # hyphens are hard to handle. for now sentences like below still have an issue:
    # one - on - one tutoring works best so that's what we tried to emulate like with me and my mom even though we knew it would be one - on - thousands 

    temp_list_1 = []
    for sentences in data_set:
        temp_list_1 += re.split('(?<=\.)|(?<=\?)',sentences)

    temp_list_2 = []
    for item in temp_list_1:
        temp_list_2.append(re.sub('^ ','',item))

    temp_list_3 = []
    for s in temp_list_2:
        try:
            if s[-1] == ".":
                temp_list_3.append(s)
            elif s[-1] == "?":
                temp_list_3.append(s)
            else:
                pass
        except:
            pass

    del data_set
    del temp_list_1
    del temp_list_2

    total_words = 0
    combined_text = ""
    outer_list = []

    # create outer_list, a list of sentences that don't go beyond 400 words
    for s in temp_list_3:
        if total_words + len(word_tokenize(s)) < 400:
            combined_text += (s + " ")
            total_words += len(word_tokenize(s))
        else:
            outer_list.append(combined_text)
            combined_text = ""
            total_words = 0        

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    encoded_data_data = tokenizer.batch_encode_plus(outer_list, max_length=450, padding='max_length', truncation=True, return_tensors='pt')

    punc_mask_outer = []
    ids_no_punc_outer = []
    attention_mask_outer = []

    for j in range(len(encoded_data_data['input_ids'])):

        # punctuation mask for sentences
        punc_mask = []
        for i in encoded_data_data['input_ids'][j]:
            if i == 1012:
                punc_mask.pop()
                punc_mask.append(1) # period
            elif i == 1029:
                punc_mask.pop()
                punc_mask.append(2) # question mark
            elif i == 1010:
                punc_mask.pop()
                punc_mask.append(3) # comma
            else:
                punc_mask.append(0)
        punc_mask_outer.append(torch.tensor(punc_mask))

        # sentences converted to word ids excluding punctuations
        # len(punc_mask) should be the same as len(ids_no_punc)
        ids_no_punc = []
        for i in encoded_data_data['input_ids'][j]:
            if i == 1012:
                pass
            elif i == 1029:
                pass
            elif i == 1010:
                pass
            else:
                ids_no_punc.append(i)
        ids_no_punc_outer.append(torch.tensor(ids_no_punc))

        # attention_mask with subwords set to 0 except for the last one
        attention_mask = []
        first_hash = True
        for i in encoded_data_data['input_ids'][j]:
            if (i == 101 or i == 102 or i == 0): # CLS, SEP, PAD
                attention_mask.append(0)
            elif (i == 1029 or i == 1010 or i == 1012):
                pass
            else:
                if re.match(r'^##', tokenizer.decode([i])):         
                    if first_hash == True:
                        attention_mask.pop()
                        attention_mask.append(0)
                        first_hash == False
                    attention_mask.append(1)
                else:
                    if first_hash == False:
                        attention_mask.pop()
                    attention_mask.append(1)                
        attention_mask_outer.append(torch.tensor(attention_mask))

    # figure out max length so that PADs can be added till it reaches max
    token_lengths = []
    for i in range(len(punc_mask_outer)):
        token_lengths.append(len(punc_mask_outer[i]))
    token_length_max = np.max(token_lengths)

    for i in range(len(punc_mask_outer)):
        # add PAD again because length is not equal after removing punctuations
        zeros = [0] * (token_length_max - len(punc_mask_outer[i]))

        punc_mask = torch.cat((punc_mask_outer[i], torch.tensor(zeros)), 0)
        ids_no_punc = torch.cat((ids_no_punc_outer[i], torch.tensor(zeros)), 0)
        attention_mask = torch.cat((attention_mask_outer[i], torch.tensor(zeros)), 0)

        if i != 0:
            pass
            punc_mask_outer_adjusted = torch.cat((punc_mask_outer_adjusted, punc_mask.view(1,-1)),0)
            ids_no_punc_outer_adjusted = torch.cat((ids_no_punc_outer_adjusted, ids_no_punc.view(1,-1)),0)
            attention_mask_outer_adjusted = torch.cat((attention_mask_outer_adjusted, attention_mask.view(1,-1)),0)
        else:
            punc_mask_outer_adjusted = punc_mask.view(1,-1)
            ids_no_punc_outer_adjusted = ids_no_punc.view(1,-1)
            attention_mask_outer_adjusted = attention_mask.view(1,-1)
            
    return ids_no_punc_outer_adjusted, attention_mask_outer_adjusted, punc_mask_outer_adjusted, punc_mask_outer, tokenizer
    
tarin_set = data_prep(train_set)
val_set = data_prep(val_set)
test_set = data_prep(test_set)
    
with open('train_set.pickle', 'wb') as f:
    pickle.dump(tarin_set, f) 

with open('val_set.pickle', 'wb') as f:
    pickle.dump(val_set, f) 

with open('test_set.pickle', 'wb') as f:
    pickle.dump(test_set, f) 
