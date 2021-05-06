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

now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print(f'Prep started: {current_time}', flush=True)
print()

torch.cuda.empty_cache()

with open('/home/ubuntu/Development/punctuation/data/train_set.pickle', 'rb') as f:
    data = pickle.load(f)

ids_no_punc_outer_adjusted, attention_mask_outer_adjusted, punc_mask_outer_adjusted, punc_mask_outer, tokenizer_train = data

with open('/home/ubuntu/Development/punctuation/data/val_set.pickle', 'rb') as f:
    data = pickle.load(f)

ids_no_punc_outer_adjusted_val, attention_mask_outer_adjusted_val, punc_mask_outer_adjusted_val, punc_mask_outer_val, tokenizer_val = data

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
    
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

batch_size = 32
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

folder = '/home/ubuntu/Development/punctuation/val_results_' + current_time

os.makedirs(folder, exist_ok=True)

for epoch in range(epochs):

    if patience_cnt <= 4:

        loss_total = 0

        preds_masked_all = torch.tensor([0]).to(device)
        labels_masked_all = torch.tensor([0]).to(device)
        
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
                
            preds = torch.argmax(outputs, axis=2)
            attention_masks = batch[1].to(torch.bool)
            labels = batch[2]

            preds_masked = torch.masked_select(preds, attention_masks)
            labels_masked = torch.masked_select(labels, attention_masks)
            
            preds_masked_all = torch.cat([preds_masked_all, preds_masked])
            labels_masked_all = torch.cat([labels_masked_all, labels_masked])
                           
        loss = loss_total/len(dataloader_train)
        acc = (preds_masked_all == labels_masked_all).sum() / len(preds_masked_all)        

        print(f'epoch: {epoch+1}, tr_loss: {loss.item():.3f}, tr_acc: {acc:.3f}', flush=True)

        with torch.no_grad():

            model.eval()

            dataset_val = TensorDataset(ids_no_punc_outer_adjusted_val, attention_mask_outer_adjusted_val, punc_mask_outer_adjusted_val)
            dataloader_val = DataLoader(dataset_val, sampler=RandomSampler(dataset_val), batch_size=batch_size)
            
            preds_masked_all = torch.tensor([0]).to(device)
            labels_masked_all = torch.tensor([0]).to(device)
                        
            for val_batch in dataloader_val:

                val_batch = [b.to(device) for b in val_batch]
                val_outputs = model(val_batch[0].to(torch.long), val_batch[1].to(torch.long))
            
                preds = torch.argmax(val_outputs, axis=2)
                attention_masks = val_batch[1].to(torch.bool)
                labels = val_batch[2]

                preds_masked = torch.masked_select(preds, attention_masks)
                labels_masked = torch.masked_select(labels, attention_masks)

                preds_masked_all = torch.cat([preds_masked_all, preds_masked])  
                labels_masked_all = torch.cat([labels_masked_all, labels_masked])
            
            val_acc = (preds_masked_all == labels_masked_all).sum() / len(preds_masked_all)        
    
            preds_masked_all = preds_masked_all.to('cpu').numpy()
            labels_masked_all = labels_masked_all.to('cpu').numpy()
    
            val_f1 = f1_score(labels_masked_all, preds_masked_all, average=None)
            val_f1_micro = f1_score(labels_masked_all, preds_masked_all, average='micro')
            val_f1_macro = f1_score(labels_masked_all, preds_masked_all, average='macro')
            val_f1_weighted = f1_score(labels_masked_all, preds_masked_all, average='weighted')
            
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
        print(f'3rd consecutive degrades observed at epoch {epoch}. So the best is epoch {epoch-5}', flush=True)
        break

torch.save(param_list[-6], folder + '/distilbert_result.pt')
        
now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print(f'Completed: {current_time}', flush=True)