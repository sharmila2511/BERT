# -*- coding: utf-8 -*-
"""
Created on Wed Oct 13 18:56:17 2021

@author: Sharmila
"""

# imports libraries

import json
import pandas as pd
import numpy as np
import os
import math, datetime

import tensorflow as tf
from tensorflow import keras

import bert
from bert import BertModelLayer
from bert.loader import StockBertConfig, map_stock_config_to_params, load_stock_weights
from bert.tokenization.bert_tokenization import FullTokenizer

from sklearn.metrics import confusion_matrix, classification_report



# read data

with open('D:/AI/DL/IntentClassificationBERT/intent_class_data.json', 'r') as inp:
    data = json.load(inp)
    
train = data['train']
test = data['test']
val = data['val']


# define train, test & val data

df = pd.DataFrame(train)
df.to_csv('train_data.csv', header=('text', 'intent'), index=False)
train_df = pd.read_csv('train_data.csv')

df = pd.DataFrame(val)
df.to_csv('val_data.csv', header=('text', 'intent'), index=False)
val_df = pd.read_csv('val_data.csv')

df = pd.DataFrame(test)
df.to_csv('test_data.csv', header=('text', 'intent'), index=False)
test_df = pd.read_csv('test_data.csv')

train_df = train_df.append(val_df).reset_index(drop=True)



# create model directory for checkpoint files

# os.makedirs("model", exist_ok=True)
bert_model_name="bert_en_uncased_L-12_H-768_A-12_4"
bert_ckpt_dir = os.path.join("model/", bert_model_name)
bert_ckpt_file = os.path.join(bert_ckpt_dir, "bert_model.ckpt")
bert_config_file = os.path.join(bert_ckpt_dir, "bert_config.json")


# input text preparation

class DataPreparation:
    text_column = 'text'
    label_column = 'intent'
    
    def __init__(self, train_df, test_df, tokenizer, classes, max_seq_len=192):
        self.tokenizer = tokenizer
        self.classes = classes
        self.max_seq_len = 0
        
        self.train_x, self.train_y = self.prepareData(train_df)
        self.test_x, self.test_y = self.prepareData(test_df)
        
        self.max_seq_len = min(self.max_seq_len, max_seq_len)
        
        self.train_x = self.data_padding(self.train_x)
        self.test_x = self.data_padding(self.test_x)
        
        
    def prepareData(self, df):
        x = [], y = []
        
        for _, row in df.iterrows():
            text, label = row[DataPreparation.text_column], row[DataPreparation.label_column]
            tokens = self.tokenizer.tokenize(text)
            tokens = ['[CLS]' + tokens + '[SEP]']
            token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            self.max_seq_len = max(self.max_seq_len, len(token_ids))
            x.append(token_ids)
            y.append(self.classes.index(label))
        return np.array(x), np.array(y)
    
    
    def data_padding(self, token_ids):
        x = []
        
        for input_ids in token_ids:
            cut_point = min(len(input_ids), self.max_seq_len - 2)  # as each token id list has 2 extra elements [CLS] & [SEP]
            input_ids = input_ids[:cut_point]
            input_ids = input_ids + [0] * (self.max_seq_len - len(input_ids))
            x.append(np.array(input_ids))
        return np.array(x)


tokenizer = FullTokenizer(vocab_file=os.path.join(bert_ckpt_dir, "vocab.txt"))


def createModel(max_seq_len, bert_config_file, bert_ckpt_file):
    # read bert config file
    with tf.io.gfile.GFile(bert_config_file, 'r') as reader:
        bc = StockBertConfig.from_json_string(reader.read())
        bert_params = map_stock_config_to_params(bc)
        bert_params.adapter_size = None
        
        bert = BertModelLayer.from_params(bert_params, name = 'bert')
        
    bert_input = keras.layers.Input(shape=(max_seq_len, ), dtype='int32', name='bert_input')
    bert_output = bert(bert_input)
        
    cls_out = keras.layers.Lambda(lambda seq: seq[:, 0, :])(bert_output)
    cls_out = keras.layers.Dropout(0.5)(cls_out)
    logits = keras.layers.Dense(units=768, activation="tanh")(cls_out)
    logits = keras.layers.Dropout(0.5)(logits)
    logits = keras.layers.Dense(units=len(classes), activation="softmax")(logits)

    model = keras.Model(inputs=input_ids, outputs=logits)
    model.build(input_shape=(None, max_seq_len))
      
    load_stock_weights(bert, bert_ckpt_file)
    
    retun model
        
        
classes = train_df.intent.unique().tolist()


# model training

data = DataPreparation(train_df, test_df, tokenizer, classes, max_seq_len=128)

model = create_model(data.max_seq_len, bert_ckpt_file)


model.compile(
  optimizer=keras.optimizers.Adam(1e-5),
  loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=[keras.metrics.SparseCategoricalAccuracy(name="acc")]
)

log_dir = "log/intent_detection/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%s")
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir)

history = model.fit(
  x=data.train_x, 
  y=data.train_y,
  validation_split=0.1,
  batch_size=16,
  shuffle=True,
  epochs=5,
  callbacks=[tensorboard_callback]
)


# evaluation


_, train_acc = model.evaluate(data.train_x, data.train_y)
_, test_acc = model.evaluate(data.test_x, data.test_y)


# prediction

y_pred = model.predict(data.test_x).argmax(axis=-1)


# metrics

cm = confusion_matrix(data.test_y, y_pred)
df_cm = pd.DataFrame(cm, index=classes, columns=classes)


# single prediction

sentences = [
  "Play our song now",
  "Rate this book as awful"
]

pred_tokens = map(tokenizer.tokenize, sentences)
pred_tokens = map(lambda tok: ["[CLS]"] + tok + ["[SEP]"], pred_tokens)
pred_token_ids = list(map(tokenizer.convert_tokens_to_ids, pred_tokens))

pred_token_ids = map(lambda tids: tids +[0]*(data.max_seq_len-len(tids)),pred_token_ids)
pred_token_ids = np.array(list(pred_token_ids))

predictions = model.predict(pred_token_ids).argmax(axis=-1)

for text, label in zip(sentences, predictions):
  print("text:", text, "\nintent:", classes[label])
  print()
