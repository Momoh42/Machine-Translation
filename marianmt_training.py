from tqdm import tqdm
from transformers import MarianMTModel, MarianTokenizer
from torch.utils.data import DataLoader, Dataset
from rouge_score import rouge_scorer

import os
import numpy as np
import nltk
import torch
import re

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available! Using GPU.")
else:
    device = torch.device("cpu")
    print("CUDA is not available! Using CPU.")

class Custom_Dataset_Marian(Dataset):
    def __init__(self, data, model,token):
        self.data = data
        self.model = model
        self.tokenizer = token

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        source_text, target_text = item[1], item[0]

        source = self.tokenizer(source_text, return_tensors="pt", padding="max_length", truncation=True, max_length=self.model.config.max_length)
        target = self.tokenizer(target_text, return_tensors="pt", padding="max_length", truncation=True, max_length=self.model.config.max_length)

        with torch.no_grad():
            outputs = self.model.generate(input_ids=source.input_ids.to(self.model.device),
                                          attention_mask=source.attention_mask.to(self.model.device),
                                          max_length=100,
                                          num_return_sequences=1,
                                          num_beams=4,
                                          early_stopping=True)
        decoded_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return {
            "input_ids": source["input_ids"].flatten(),
            "attention_mask": source["attention_mask"].flatten(),
            "labels": target["input_ids"].flatten(),
            "decoder_attention_mask": target["attention_mask"].flatten(),
            "decoded_target": target_text,
            "decoded_output": decoded_output
        }

 
def freeze_layers(model, num_layers_to_keep_trainable):
    if num_layers_to_keep_trainable == 'Encoder':
        # Freeze decoder
        for name, param in model.named_parameters():
            if "encoder" in name:
                param.requires_grad=False
                print(f"Parameter: {name}, Requires Grad: {param.requires_grad}")
    else:
        param_names = list(model.state_dict().keys())

        decoder_layers = [param for param in param_names if "decoder.layers" in param]

        numbers = [int(re.search(r'\d+', string).group()) for string in decoder_layers if re.search(r'\d+', string)]

        selected_layer = [str(num) for num in range(max(numbers)-num_layers_to_keep_trainable+1,max(numbers)+1)]

        decoder_layers=[layer for layer in decoder_layers for item in selected_layer if item in layer]

        for idx, (name, param) in enumerate(model.named_parameters()):
            if name in decoder_layers:
                param.requires_grad = True
            else:
                param.requires_grad = False
            print(f"Parameter: {name}, Requires Grad: {param.requires_grad}")

def bleu_score(True_label,pred_label,n_gram):
    smooth = nltk.translate.bleu_score.SmoothingFunction().method1
    weights=[]
    for v in range(1,n_gram+1):
        weights.append(tuple(1 / v for _ in range(1, v + 1)))
        
    return nltk.translate.bleu_score.sentence_bleu([True_label.split()],pred_label.split(),weights=weights, smoothing_function=smooth)

def meteor_score(True_label,pred_label):
    return nltk.translate.meteor_score.meteor_score([nltk.tokenize.word_tokenize(True_label.lower())], nltk.tokenize.word_tokenize(pred_label.lower()))

def rouge_score(True_label,pred_label):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    return scorer.score(True_label, pred_label)

def calculate_accuracy(True_labels,pred_labels,n_gram):
    sum_bleu=[0]*n_gram
    sum_meteor=0
    avg_rouge = {metric: {'precision': 0.0, 'recall': 0.0, 'fmeasure': 0.0} for metric in ['rouge1', 'rouge2', 'rougeL']}


    for i in range(len(True_labels)):
        sum_bleu=[x + y for x, y in zip(sum_bleu, bleu_score(True_labels[i],pred_labels[i],n_gram))]
        sum_meteor+=meteor_score(True_labels[i],pred_labels[i])
        scores = rouge_score(True_labels[i],pred_labels[i])
        for metric, value in scores.items():
            avg_rouge[metric]["precision"] += value.precision
            avg_rouge[metric]["recall"] += value.recall
            avg_rouge[metric]["fmeasure"] += value.fmeasure
    
    for inner_dict in avg_rouge.values():
        for key, value in inner_dict.items():
            inner_dict[key] = value / len(True_labels)

    return np.array(sum_bleu)/len(True_labels),sum_meteor/len(True_labels),avg_rouge

import time

def training(num_epochs,model,nb_layer,train_loader,optimizer,n_gram,model_name,val_loader):
    freeze_layers(model,nb_layer)
    directory=os.getcwd()+"/"+model_name
    os.makedirs(directory, exist_ok=True)

    train_loss=[]
    meteor_sc=[]
    bleu_sc=[]
    rouge_sc=[]

    val_loss=[]
    val_meteor_sc=[]
    val_bleu_sc=[]
    val_rouge_sc=[]

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_bleu = [0]*n_gram
        total_meteor = 0
        avg_rouge = {metric: {'precision': 0.0, 'recall': 0.0, 'fmeasure': 0.0} for metric in ['rouge1', 'rouge2', 'rougeL']}
        
        for batch in tqdm(train_loader, desc="Training, epoch "+str(epoch+1)):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            decoder_attention_mask = batch['decoder_attention_mask'].to(device)
            decoded_targets = batch['decoded_target']
            decoded_outputs = batch['decoded_output']

            #print("Decoded Targets:", decoded_targets)
            #print("Decoded Outputs:", decoded_outputs)
            
            acc_bleu,acc_meteor,acc_rouge=calculate_accuracy(decoded_targets,decoded_outputs,n_gram)
            optimizer.zero_grad()

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                decoder_attention_mask=decoder_attention_mask,
            )

            for metric, value in acc_rouge.items():
                avg_rouge[metric]["precision"] += value["precision"]
                avg_rouge[metric]["recall"] += value["recall"]
                avg_rouge[metric]["fmeasure"] += value["fmeasure"]
            
            
            
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_bleu =[x + y for x, y in zip(total_bleu, acc_bleu)]
            total_meteor +=acc_meteor

        avg_train_loss = total_loss / len(train_loader)
        avg_bleu=np.array(total_bleu)/len(train_loader)
        avg_meteor=total_meteor/len(train_loader)

        for inner_dict in avg_rouge.values():
            for key, value in inner_dict.items():
                inner_dict[key] = value / len(train_loader)

        train_loss.append(avg_train_loss)
        meteor_sc.append(avg_meteor)
        bleu_sc.append(avg_bleu)
        rouge_sc.append(avg_rouge)

        formatted_scores = ', '.join(f"Avg. Training Bleu Score {index + 1}-gram : {score:.2f}" for index, score in enumerate(avg_bleu))

        formatted_rouge = ', '.join([f"Avg. Training Rouge {key} Recall : {avg_rouge[key]['recall']:.2f}" for key in avg_rouge.keys()])

        print(f"Epoch {epoch+1} Avg. Training Loss: {avg_train_loss} - Avg. Training Meteor Score {avg_meteor} - {formatted_scores} - {formatted_rouge}")

        avg_val_loss,avg_val_bleu,avg_val_meteor,avg_val_rouge=testing(model,val_loader,n_gram)

        val_loss.append(avg_val_loss)
        val_bleu_sc.append(avg_val_bleu)
        val_meteor_sc.append(avg_val_meteor)
        val_rouge_sc.append(avg_val_rouge)

        save_model(model,directory+"/"+model_name+"_"+str(epoch+1))

    np.save(directory+"/"+model_name+"_train_loss.npy",train_loss)
    np.save(directory+"/"+model_name+"_train_meteor_sc.npy",meteor_sc)
    np.save(directory+"/"+model_name+"_train_bleu_sc.npy",bleu_sc)
    np.save(directory+"/"+model_name+"_train_rouge_sc.npy",rouge_sc)

    np.save(directory+"/"+model_name+"_val_loss.npy",val_loss)
    np.save(directory+"/"+model_name+"_val_meteor_sc.npy",val_meteor_sc)
    np.save(directory+"/"+model_name+"_val_bleu_sc.npy",val_bleu_sc)
    np.save(directory+"/"+model_name+"_val_rouge_sc.npy",val_rouge_sc)

    return train_loss,bleu_sc,meteor_sc,rouge_sc,directory

def testing(model,val_data,n_gram):
    with torch.no_grad(): 
        model.eval()
        total_loss = 0
        total_bleu = [0]*n_gram
        total_meteor = 0
        avg_rouge = {metric: {'precision': 0.0, 'recall': 0.0, 'fmeasure': 0.0} for metric in ['rouge1', 'rouge2', 'rougeL']}

        for batch in tqdm(val_data):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            decoder_attention_mask = batch['decoder_attention_mask'].to(device)
            decoded_targets = batch['decoded_target']  # Decoded target labels
            decoded_outputs = batch['decoded_output']

            
            acc_bleu,acc_meteor,acc_rouge=calculate_accuracy(decoded_targets,decoded_outputs,n_gram)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                decoder_attention_mask=decoder_attention_mask,
            )

            for metric, value in acc_rouge.items():
                avg_rouge[metric]["precision"] += value["precision"]
                avg_rouge[metric]["recall"] += value["recall"]
                avg_rouge[metric]["fmeasure"] += value["fmeasure"]

            loss = outputs.loss
            total_loss += loss.item()
            total_bleu = [x + y for x, y in zip(total_bleu, acc_bleu)]
            total_meteor +=acc_meteor

    avg_loss = total_loss / len(val_data)
    avg_bleu=np.array(total_bleu)/len(val_data)
    avg_meteor=total_meteor/len(val_data)

    for inner_dict in avg_rouge.values():
            for key, value in inner_dict.items():
                inner_dict[key] = value / len(val_data)

    formatted_bleu = ', '.join(f"Avg. Validation Bleu Score {index + 1}Gram Overlaps: {score:.2f}" for index, score in enumerate(avg_bleu))

    formatted_rouge = ', '.join([f"Avg. Validation Rouge {key} Recall : {avg_rouge[key]['recall']:.2f}" for key in avg_rouge.keys()])

    print(f"Avg. Validation Loss: {avg_loss} - Avg. Validation Meteor Score {avg_meteor} - {formatted_bleu} - {formatted_rouge}")

    return avg_loss,avg_bleu,avg_meteor,avg_rouge

def save_model(model,name_file):
    torch.save(model, name_file+'.pth')


train_data=np.load(os.getcwd()+"/Train_dataset.npy",allow_pickle=True)
val_data=np.load(os.getcwd()+"/Val_dataset.npy",allow_pickle=True)
test_data=np.load(os.getcwd()+"/Test_dataset.npy",allow_pickle=True)

#model_name="C:/Users/hedir/.cache/huggingface/hub/models--Helsinki-NLP--opus-mt-fr-en/snapshots/b4a9a384c2ec68a224bbd2ee3fd5df0c71ca5b1b"
model_name=os.getcwd()+"/.cache/huggingface/hub/models--Helsinki-NLP--opus-mt-fr-en/snapshots/b4a9a384c2ec68a224bbd2ee3fd5df0c71ca5b1b"

token = MarianTokenizer.from_pretrained(model_name,local_files_only=True)

model = MarianMTModel.from_pretrained(model_name,local_files_only=True)

train_dataset=Custom_Dataset_Marian(train_data,model,token)
val_dataset=Custom_Dataset_Marian(val_data,model,token)
test_dataset=Custom_Dataset_Marian(test_data,model,token)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=True)

optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)

model.to(device)

file_name="MARIANMT_1"
train_loss,bleu_sc,meteor_sc,rouge_sc,directory=training(3,model,1,train_loader,optimizer,4,file_name,val_loader)
avg_test_loss,avg_test_bleu,avg_test_meteor,avg_test_rouge=testing(model,test_loader,4)

np.save(directory+"/"+file_name+"_test_loss.npy",avg_test_loss)
np.save(directory+"/"+file_name+"_test_meteor_sc.npy",avg_test_meteor)
np.save(directory+"/"+file_name+"_test_bleu_sc.npy",avg_test_bleu)
np.save(directory+"/"+file_name+"_test_rouge_sc.npy",avg_test_rouge)