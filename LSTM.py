# %%
from __future__ import print_function, division
from builtins import range
import pandas as pd
import numpy as np
import re

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Embedding, Bidirectional, RepeatVector, Concatenate, Dot, Lambda
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import ModelCheckpoint, Callback
from tensorflow.keras import optimizers
from tensorflow.keras.utils import Sequence

from gensim.models import KeyedVectors

# %%
NUM_SENTENCES = 100000000
MAX_SENT_LEN = 30
BATCH_SIZE = 64
EPOCHS = 100
LATENT_DIM = 256
LATENT_DIM_DECODER = 256 
EMBEDDING_DIM = 200

# %%
file_path = ''

print("Begin")
def load_data(file):
    data = np.load(file, allow_pickle=True)
    return pd.DataFrame(data, columns=['en', 'fr'])

def clean_sentence(text):
    text = re.sub(r"[,.\?:#@(){}*;/!\"«»“”‘’·\[…\]+\'<>]", " ", text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()

df_train = load_data(file_path+'Train_dataset_plus.npy').applymap(clean_sentence)
df_test = load_data(file_path+'Test_dataset_plus.npy').applymap(clean_sentence)
df_validation = load_data(file_path+'Val_dataset_plus.npy').applymap(clean_sentence)

print("len df_train : ", len(df_train))
print("len df_test : ",len(df_test))
print("len df_validation : ",len(df_validation))

print('After removing the long sentences')

def remove_long_sentences(df):
    
    df['word_count_en'] = df['en'].apply(lambda x: len(x.split()))
    df['word_count_fr'] = df['fr'].apply(lambda x: len(x.split()))
    df = df[(df['word_count_en'] <= MAX_SENT_LEN) & (df['word_count_fr'] <= MAX_SENT_LEN)].copy()
    df.drop(columns=['word_count_en', 'word_count_fr'], inplace=True)

    return df

df_train = remove_long_sentences(df_train)
df_validation = remove_long_sentences(df_validation)
df_test = remove_long_sentences(df_test)

print("len df_train : ", len(df_train))
print("len df_test : ", len(df_test))
print("len df_validation : ",len(df_validation))

# %%
def inputs_and_outputs(df):
    inputs  = list(df['fr'].values[:NUM_SENTENCES])
    outputs = list(df['en'].values[:NUM_SENTENCES])
    outputs_i = ['<sos> ' + text for text in outputs]
    outputs_f = [text + ' <eos>' for text in outputs]
    return inputs, outputs, outputs_i, outputs_f

input_texts,      output_texts,      target_texts_inputs,      target_texts       = inputs_and_outputs(df_train)
input_texts_test, output_texts_test, target_texts_inputs_test, target_texts_test  = inputs_and_outputs(df_test)
input_texts_val,  output_texts_val,  target_texts_inputs_val,  target_texts_val   = inputs_and_outputs(df_validation)

# %%
def tokenize_and_sequence(inputs, outputs_i, outputs_f, tokenizer=None, output_tokenizer=None):

    if tokenizer is None:
        tokenizer = Tokenizer(filters='')
        tokenizer.fit_on_texts(inputs)

    input_seq = tokenizer.texts_to_sequences(inputs)
    input_word2index = tokenizer.word_index
    input_num_words = len(input_word2index) + 1
    input_max_len = max(len(s) for s in input_seq)
    
    if output_tokenizer is None:
        output_tokenizer = Tokenizer(filters='')
        output_tokenizer.fit_on_texts(outputs_i + outputs_f)

    outputs_i_seq = output_tokenizer.texts_to_sequences(outputs_i)
    outputs_seq = output_tokenizer.texts_to_sequences(outputs_f)
    outputs_word2index = output_tokenizer.word_index
    outputs_numwords = len(outputs_word2index) + 1
    outputs_maxlen = max(len(s) for s in outputs_seq)

    return tokenizer, input_seq, input_word2index, input_num_words, input_max_len, output_tokenizer, outputs_i_seq, outputs_seq, outputs_word2index, outputs_numwords, outputs_maxlen

(tokenizer_inputs,
 input_sequences, 
 word2idx_inputs, 
 numwords_inputs, 
 max_len_input, 
 tokenizer_outputs,
 target_sequences_inputs, 
 target_sequences, 
 word2idx_outputs, 
 num_words_output, 
 max_len_target) = tokenize_and_sequence(input_texts, target_texts_inputs,  target_texts)

print(f'Total unique words in input :', numwords_inputs)
print(f'Length of longest sentence in input :', max_len_input)
print(f'Total unique words in output :', num_words_output)
print(f'Length of longest sentence in output :', max_len_target)

#Data validation
input_sequences_val         = tokenizer_inputs.texts_to_sequences(input_texts_val)
target_sequences_inputs_val = tokenizer_outputs.texts_to_sequences(target_texts_inputs_val)  
target_sequences_val        = tokenizer_outputs.texts_to_sequences(target_texts_val)

#Data test
input_sequences_test         = tokenizer_inputs.texts_to_sequences(input_texts_test)
target_sequences_inputs_test = tokenizer_outputs.texts_to_sequences(target_texts_inputs_test)  
target_sequences_test        = tokenizer_outputs.texts_to_sequences(target_texts_test)

# %%
encoder_inputs = pad_sequences(input_sequences,  maxlen=max_len_input)
print('encoder_input_sequences shape:', encoder_inputs.shape)

decoder_inputs= pad_sequences(target_sequences_inputs, maxlen=max_len_target, padding='post')
print('decoder_inputs_sequences shape:', decoder_inputs.shape)

decoder_targets = pad_sequences(target_sequences, maxlen=max_len_target, padding='post')
print('decoder_output_sequences shape:', decoder_targets.shape)

#Data validation
encoder_inputs_val = pad_sequences(input_sequences_val, maxlen=max_len_input)
decoder_inputs_val = pad_sequences(target_sequences_inputs_val, maxlen=max_len_target, padding='post')
decoder_targets_val = pad_sequences(target_sequences_val, maxlen=max_len_target, padding='post')

#Data test
encoder_inputs_test = pad_sequences(input_sequences_test, maxlen=max_len_input)
decoder_inputs_test = pad_sequences(target_sequences_inputs_test, maxlen=max_len_target, padding='post')
decoder_targets_test = pad_sequences(target_sequences_test, maxlen=max_len_target, padding='post')

# %%
embedding_matrix = np.zeros((numwords_inputs, EMBEDDING_DIM))

fr_w2v = KeyedVectors.load_word2vec_format(file_path+'frWac_non_lem_no_postag_no_phrase_200_cbow_cut100.bin', binary=True)

for word, i in word2idx_inputs.items():
    if word in fr_w2v.key_to_index:
        embedding_matrix[i] = fr_w2v[word]

# %%
def softmax_over_time(x):
  assert(K.ndim(x) > 2)
  e = K.exp(x - K.max(x, axis=1, keepdims=True))
  s = K.sum(e, axis=1, keepdims=True)
  return e / s

# create embedding layer
embedding_layer = Embedding(
  numwords_inputs,
  EMBEDDING_DIM,
  weights=[embedding_matrix],
  input_length=max_len_input,
  # trainable=True
)

#encoder
encoder_inputs_placeholder = Input(shape=(max_len_input,))
x = embedding_layer(encoder_inputs_placeholder)
encoder = Bidirectional(LSTM(
  LATENT_DIM,
  return_sequences=True, dropout=0.2
))
encoder_outputs = encoder(x)

#Decoder
decoder_inputs_placeholder = Input(shape=(max_len_target,))
decoder_embedding = Embedding(num_words_output, EMBEDDING_DIM)
decoder_inputs_x = decoder_embedding(decoder_inputs_placeholder)

# %%
#Attention
attn_repeat_layer = RepeatVector(max_len_input)
attn_concat_layer = Concatenate(axis=-1)
attn_dense1 = Dense(10, activation='tanh')
attn_dense2 = Dense(1, activation=softmax_over_time)

attn_dot = Dot(axes=1) # to perform the weighted sum of alpha[t] * h[t]

def one_step_attention(h, st_1):
  # h = h(1), ..., h(Tx), shape = (Tx, LATENT_DIM * 2)
  # st_1 = s(t-1), shape = (LATENT_DIM_DECODER,)
 
  # copy s(t-1) Tx times
  st_1 = attn_repeat_layer(st_1)

  # Concatenate all h(t)'s with s(t-1)
  x = attn_concat_layer([h, st_1])


  x = attn_dense1(x)
  
  alphas = attn_dense2(x)

  context = attn_dot([alphas, h])

  return context

# define the rest of the decoder (after attention)
decoder_lstm = LSTM(LATENT_DIM_DECODER, return_state=True)
decoder_dense = Dense(num_words_output, activation='softmax')

initial_s = Input(shape=(LATENT_DIM_DECODER,), name='s0')
initial_c = Input(shape=(LATENT_DIM_DECODER,), name='c0')
context_last_word_concat_layer = Concatenate(axis=2)

# %%
# s, c will be re-assigned in each iteration of the loop
s = initial_s
c = initial_c

# collect outputs in a list at first
outputs = []
for t in range(max_len_target): # Ty times
  # get the context using attention
  context = one_step_attention(encoder_outputs, s)

  # we need a different layer for each time step
  selector = Lambda(lambda x: x[:, t:t+1])
  xt = selector(decoder_inputs_x)
  
  # combine 
  decoder_lstm_input = context_last_word_concat_layer([context, xt])

  # pass the combined [context, last word] into the LSTM
  # along with [s, c]
  # get the new [s, c] and output
  o, s, c = decoder_lstm(decoder_lstm_input, initial_state=[s, c])

  # final dense layer to get next word prediction
  decoder_outputs = decoder_dense(o)
  outputs.append(decoder_outputs)

# %%
def stack_and_transpose(x):
  # x is a list of length T, each element is a batch_size x output_vocab_size tensor
  x = K.stack(x) # is now T x batch_size x output_vocab_size tensor
  x = K.permute_dimensions(x, pattern=(1, 0, 2)) # is now batch_size x T x output_vocab_size
  return x

# make it a layerx``
stacker = Lambda(stack_and_transpose)
outputs = stacker(outputs)

# %%
# create the model
model = Model(
  inputs=[
    encoder_inputs_placeholder,
    decoder_inputs_placeholder,
    initial_s, 
    initial_c,
  ],
  outputs=outputs
)

# %%
learning_rate=0.001

# compile the model

model.compile(optimizer=optimizers.Adam(learning_rate) ,loss='categorical_crossentropy', metrics=['accuracy'])

# %%
class HistorySaver(Callback):
    def __init__(self, filename):
        super(HistorySaver, self).__init__()
        self.filename = filename

    def on_epoch_end(self, epoch, logs=None):
        df_history = pd.DataFrame(self.model.history.history)
        df_history.to_csv(self.filename, index=False)

class OneHotGenerator(Sequence):
    def __init__(self, input_seq, output_seq, batch_size, vocab_size, latent_dim):
        self.input_seq, self.output_seq = input_seq, output_seq
        self.batch_size = batch_size
        self.vocab_size = vocab_size
        self.latent_dim = latent_dim

    def __len__(self):
        return int(np.ceil(len(self.input_seq) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_input_seq = self.input_seq[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_output_seq = self.output_seq[idx * self.batch_size:(idx + 1) * self.batch_size]

        output_seq_onehot = np.zeros((len(batch_output_seq), len(batch_output_seq[0]), self.vocab_size), dtype=np.int8)

        for i, sequence in enumerate(batch_output_seq):
            for t, word_id in enumerate(sequence):
                output_seq_onehot[i, t, word_id] = 1

        z = np.zeros((len(batch_input_seq), self.latent_dim)) # initial [s, c]
        return [batch_input_seq, batch_output_seq, z, z], output_seq_onehot

# %%
# train the model
one_hot_generator     = OneHotGenerator(encoder_inputs, decoder_targets, BATCH_SIZE, num_words_output, LATENT_DIM_DECODER)
one_hot_generator_val = OneHotGenerator(encoder_inputs_val, decoder_targets_val, BATCH_SIZE, num_words_output, LATENT_DIM_DECODER)

filename = 'model_lstm_ok.keras'
checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=0, save_best_only=True, mode='min')

history_saver = HistorySaver('history_lstm_ok.csv')

r = model.fit(
  one_hot_generator,
  validation_data=one_hot_generator_val,
  batch_size=BATCH_SIZE,
  epochs=EPOCHS,
  callbacks=[checkpoint, history_saver]
)

# %%
#Modify
encoder_model = Model(encoder_inputs_placeholder, encoder_outputs)

# next we define a T=1 decoder model
encoder_outputs_as_input = Input(shape=(max_len_input, LATENT_DIM * 2,))
decoder_inputs_single = Input(shape=(1,))
decoder_inputs_single_x = decoder_embedding(decoder_inputs_single)

# no need to loop over attention steps this time because there is only one step
context = one_step_attention(encoder_outputs_as_input, initial_s)

# combine context with last word
decoder_lstm_input = context_last_word_concat_layer([context, decoder_inputs_single_x])

# lstm and final dense
o, s, c = decoder_lstm(decoder_lstm_input, initial_state=[initial_s, initial_c])
decoder_outputs = decoder_dense(o)


# %%
# create the model object
decoder_model = Model(
  inputs=[
    decoder_inputs_single,
    encoder_outputs_as_input,
    initial_s, 
    initial_c
  ],
  outputs=[decoder_outputs, s, c]
)

# %%
#Making Predictions
idx2word_eng = {v:k for k, v in word2idx_inputs.items()}
idx2word_trans = {v:k for k, v in word2idx_outputs.items()}

# %%
def decode_sequence(input_seq):
  # Encode the input as state vectors.
  enc_out = encoder_model.predict(input_seq, verbose=0)

  target_seq = np.zeros((1, 1))
  
  target_seq[0, 0] = word2idx_outputs['<sos>']

  # if we get this we break
  eos = word2idx_outputs['<eos>']


  # [s, c] will be updated in each loop iteration
  s = np.zeros((1, LATENT_DIM_DECODER))
  c = np.zeros((1, LATENT_DIM_DECODER))


  # Create the translation
  output_sentence = []
  for _ in range(max_len_target):
    o, s, c = decoder_model.predict([target_seq, enc_out, s, c], verbose=0)
        

    # Get next word
    idx = np.argmax(o.flatten())

    # End sentence of EOS
    if eos == idx:
      break

    word = ''
    if idx > 0:
      word = idx2word_trans[idx]
      output_sentence.append(word)

    # Update the decoder input
    # which is just the word just generated
    target_seq[0, 0] = idx

  return ' '.join(output_sentence)

# %%
test_actual_sentence=[]
test_predicted_sentence=[]

for i in range(len(input_texts_test)):
  # Do some test translations
  if i%100 == 0: 
    print("predict ", i, "/",len(input_texts_test))

  input_seq = encoder_inputs_test[i:i+1]
  translation = decode_sequence(input_seq)

  test_actual_sentence.append(target_texts_test[i])
  test_predicted_sentence.append(translation)

# %%
df = pd.DataFrame({
    'French': input_texts_test,
    'English':  target_texts_test,
    'Predicted English': test_predicted_sentence
})

df.to_csv('test_lstm_ok.csv', index=False)

# %%
'''
for i in [1,2,3,4,5]:
    print('-')
    print('Input sentence:', input_texts_test[i])
    print('Predicted translation:', test_predicted_sentence[i])
    print('Actual translation:', target_texts_test[i])
'''

# %%
'''
df_test = pd.read_csv('test_lstm_ok.csv')

df_test
'''

