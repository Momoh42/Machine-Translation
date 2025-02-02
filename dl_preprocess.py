
from tqdm import tqdm
from transformers import MarianTokenizer, MarianMTModel

import os
import langid
import numpy as np
from langid.langid import LanguageIdentifier, model
import nltk
import torch
import math

nltk.download('punkt')
nltk.download('wordnet')

state=42


if torch.cuda.is_available():
    device = torch.device("cuda")
    print("CUDA is available! Using GPU.")
else:
    device = torch.device("cpu")
    print("CUDA is not available! Using CPU.")

model_name="Helsinki-NLP/opus-mt-fr-en"

token = MarianTokenizer.from_pretrained(model_name)

model = MarianMTModel.from_pretrained(model_name)

translation_dataset_np = np.load(os.getcwd()+"/Parallel_Global_Voices_English_French.npy",allow_pickle=True)
array_meteor_score=[]
array_01=[]

model.to(device)

for i in tqdm(range(len(translation_dataset_np))):
    source_text, target_text = translation_dataset_np[i,1], translation_dataset_np[i,0]

    source = token(source_text, return_tensors="pt", padding="max_length", truncation=True, max_length=model.config.max_length)
    target = token(target_text, return_tensors="pt", padding="max_length", truncation=True, max_length=model.config.max_length)

    with torch.no_grad():
        outputs = model.generate(input_ids=source.input_ids.to(device), 
                                          attention_mask=source.attention_mask.to(device),
                                          max_length=100, 
                                          num_return_sequences=1,
                                          num_beams=4, 
                                          early_stopping=True)
    decoded_output = token.decode(outputs[0], skip_special_tokens=True)
    #print(decoded_output)
    #print(target_text)
    meteor_sc=nltk.translate.meteor_score.meteor_score([nltk.tokenize.word_tokenize(target_text.lower())], nltk.tokenize.word_tokenize(decoded_output.lower()))
    if(meteor_sc>0.2):
        array_01.append(1)
    else:
        array_01.append(0)
    array_meteor_score.append(meteor_sc)

np.save(os.getcwd()+"/array_01.npy",array_01)
np.save(os.getcwd()+"/array_meteor_score.npy",array_meteor_score)