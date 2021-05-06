import pandas as pd
import re
import numpy as np
from nltk.tokenize import word_tokenize
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch
import torch.nn as nn
from transformers import DistilBertTokenizer, DistilBertModel
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.utils.class_weight import compute_class_weight
from datetime import datetime
from sklearn.metrics import f1_score
import pickle

print('Preparing data', flush=True)

np.random.seed(1)

df = pd.read_csv("/home/ubuntu/Development/punctuation/transcripts.csv")
# len(df): 2467

# train:val:test = 0.8:0.1:0.1
num_train = int(len(df)*0.8)+2
num_val = int(len(df)*0.1)
num_test = int(len(df)*0.1)

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

ids_no_punc_outer_adjusted, attention_mask_outer_adjusted, punc_mask_outer_adjusted, punc_mask_outer, tokenizer_train = data_prep(train_set)
ids_no_punc_outer_adjusted_val, attention_mask_outer_adjusted_val, punc_mask_outer_adjusted_val, punc_mask_outer_val, tokenizer_val = data_prep(val_set)

class DistilBERT_Arch(nn.Module):
    def __init__(self, distilbert):
        super().__init__()
        self.distilbert = distilbert
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(768, 512)
        self.fc2 = nn.Linear(512, 4)
        self.softmax = nn.LogSoftmax(dim=1)
    def forward(self, input_ids, mask):        
        cls_hs = self.distilbert(input_ids, attention_mask=mask)[0]
        x = self.fc1(cls_hs)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

print('Preparing for training', flush=True)
    
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

batch_size = 3 # originally 3
epochs = 200

dataset_train = TensorDataset(ids_no_punc_outer_adjusted, attention_mask_outer_adjusted, punc_mask_outer_adjusted)
dataloader_train = DataLoader(dataset_train, sampler=RandomSampler(dataset_train), batch_size=batch_size)

distilbert = DistilBertModel.from_pretrained("distilbert-base-uncased",
                                                    num_labels=4,  
                                                    output_attentions=False,
                                                    output_hidden_states=False)
model = DistilBERT_Arch(distilbert)
optimizer = AdamW(model.parameters(),
                 lr=1e-5,
                 eps=1e-8)
scheduler = get_linear_schedule_with_warmup(optimizer,
                                           num_warmup_steps=0,
                                           num_training_steps=len(dataloader_train)*epochs)

# prep for class_weights
for i, p in enumerate(punc_mask_outer):    
    if i != 0:
        punc_cat = torch.cat((punc_cat, p), dim=0)
    else:
        punc_cat = p
        
class_weights = compute_class_weight('balanced', np.unique(punc_cat), punc_cat.numpy())
#TODO: haven't considered attention_mask yet!!
weights = torch.tensor(class_weights, dtype=torch.float)
weights = weights.to(device)
cross_entropy = nn.NLLLoss(weight=weights)

patience_cnt = 0
prev_f1 = 0

param_list = []

now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print(f'Training started: {current_time}', flush=True)
print()

val_f1_list = []
val_f1_micro_list = []
val_f1_macro_list = []
val_f1_weighted_list = []

now = datetime.now()
current_time = now.strftime("%m%d%H%M")

folder = './val_results_' + current_time

os.makedirs(folder, exist_ok=True)

for epoch in range(epochs):

    if patience_cnt <= 2:
    
        nume = 0
        deno = 0
        loss_total = 0

        for batch in dataloader_train:

            model.train()
            model.zero_grad()

            batch = [b.to(device) for b in batch]

            model.to(device)

            outputs = model(batch[0].to(torch.long), batch[1].to(torch.long))

            loss = cross_entropy(outputs.to(torch.float32).view(-1, 4), batch[2].to(torch.long).view(-1))

            loss.backward()
            optimizer.step()
            scheduler.step()

            model.eval()

            loss_total += loss

            for j in range(outputs.shape[0]):
            # for jth sample in a batch

                preds = np.argmax(outputs[j].to('cpu').detach().numpy(), axis=1)
                labels = batch[2].to(torch.long)[j].to('cpu').detach().numpy()

                # for ith token in a jth sample
                # if attention mask is not 0, check if predictinon matches label
                for i in range(len(batch[1][j])):
                    if batch[1][j][i] != 0:
                        if preds[i] == labels[i]:
                            nume += 1
                        deno += 1

        loss = loss_total/len(dataloader_train)
        acc = nume/deno    
        print(f'epoch: {epoch+1}, tr_loss: {loss.item():.3f}, tr_acc: {acc:.3f}', flush=True)

        with torch.no_grad():

            model.eval()

            dataset_val = TensorDataset(ids_no_punc_outer_adjusted_val, attention_mask_outer_adjusted_val, punc_mask_outer_adjusted_val)
            dataloader_val = DataLoader(dataset_val, sampler=RandomSampler(dataset_val), batch_size=batch_size)

            val_nume = 0
            val_deno = 0

            val_preds_cat = []
            val_labels_cat = []
            
            for val_batch in dataloader_val:

                val_batch = [b.to(device) for b in val_batch]
                val_outputs = model(val_batch[0].to(torch.long), val_batch[1].to(torch.long))

                for j in range(val_outputs.shape[0]):
                # for jth sample in a batch

                    val_preds = np.argmax(val_outputs[j].to('cpu').detach().numpy(), axis=1)
                    val_labels = val_batch[2].to(torch.long)[j].to('cpu').detach().numpy()

                    # for ith token in a jth sample
                    # if attention mask is not 0, check if predictinon matches label
                    for i in range(len(val_batch[1][j])):
                        if val_batch[1][j][i] != 0:
                            if val_preds[i] == val_labels[i]:
                                val_nume += 1
                                
                            val_preds_cat.append(val_preds[i])
                            val_labels_cat.append(val_labels[i])
                                
                            val_deno += 1
                      
            val_preds_cat = np.array(val_preds_cat)
            val_labels_cat = np.array(val_labels_cat)

            val_acc = val_nume/val_deno    
            val_f1 = f1_score(val_labels_cat, val_preds_cat, average=None)
            val_f1_micro = f1_score(val_labels_cat, val_preds_cat, average='micro')
            val_f1_macro = f1_score(val_labels_cat, val_preds_cat, average='macro')
            val_f1_weighted = f1_score(val_labels_cat, val_preds_cat, average='weighted')
                        
            val_f1_list.append(val_f1)
            val_f1_micro_list.append(val_f1_micro)
            val_f1_macro_list.append(val_f1_macro)
            val_f1_weighted_list.append(val_f1_weighted)
                        
            now = datetime.now()
            current_time = now.strftime("%H:%M:%S")
            print(f'Epoch complete: {current_time}', flush=True)
            
            print(f'val_acc: {val_acc:.4f}', flush=True)
            print(f'val_f1: {val_f1}', flush=True)
            print(f'val_f1_micro: {val_f1_micro:.4f}', flush=True)
            print(f'val_f1_macro: {val_f1_macro:.4f}', flush=True)
            print(f'val_f1_weighted: {val_f1_weighted:.4f}', flush=True)
            print()
                               
            if prev_f1 >= val_f1_macro:
                patience_cnt += 1
            else:
                patience_cnt = 0
            
            prev_f1 = val_f1_macro
            param_list.append(model)
                
            torch.save(param_list, folder + '/distilbert_result.pt')
            
            f = open(folder + '/val_f1_list.txt', 'wb')
            pickle.dump(val_f1_list, f)
            
            f = open(folder + '/val_f1_micro_list.txt', 'wb')
            pickle.dump(val_f1_micro_list, f)
            
            f = open(folder + '/val_f1_macro_list.txt', 'wb')
            pickle.dump(val_f1_macro_list, f)

            f = open(folder + '/val_f1_weighted_list.txt', 'wb')
            pickle.dump(val_f1_weighted_list, f)

    else:
        print(f'3rd consecutive degrades observed at epoch {epoch+1}. So the best is epoch {epoch-2}', flush=True)
        break

torch.save(param_list[-4], folder + '/distilbert_result.pt')
        
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print(f'Completed: {current_time}', flush=True)