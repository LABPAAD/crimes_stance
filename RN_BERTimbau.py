"""Códogo referente a estratégia de ajuste fino da rede neural usando o modelo BERTimbau"""


import json
import pandas as pd
from tqdm import tqdm
import pickle

import seaborn as sns
import matplotlib.pyplot as plt

import transformers
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup, BertConfig

from collections import defaultdict

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
RANDOM_SEED = 42
MAX_LEN = 160
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

class GPReviewDataset(Dataset):

  def __init__(self, reviews, targets, tokenizer, max_len):
    self.reviews = reviews
    self.targets = targets
    self.tokenizer = tokenizer
    self.max_len = max_len
  
  def __len__(self):
    return len(self.reviews)
  
  def __getitem__(self, item):
    review = str(self.reviews[item])
    target = self.targets[item]

    encoding = self.tokenizer.encode_plus(
      review,
      add_special_tokens=True,
      max_length=self.max_len,
      return_token_type_ids=False,
      #padding='longest',
      pad_to_max_length=True,
      return_attention_mask=True,
      return_tensors='pt',
    )

    return {
      'review_text': review,
      'input_ids': encoding['input_ids'].flatten(),
      'attention_mask': encoding['attention_mask'].flatten(),
      'targets': torch.tensor(target, dtype=torch.long)
    }

def create_data_loader(df, tokenizer, max_len, batch_size):
  ds = GPReviewDataset(
    reviews=df.comentarios.to_numpy(),
    targets=df.rotulacao_manual.to_numpy(),
    tokenizer=tokenizer,
    max_len=max_len
  )

  return DataLoader(
    ds,
    batch_size=batch_size,
    num_workers=4
  )

class SentimentClassifier(torch.nn.Module):

  def __init__(self, n_classes):
    super(SentimentClassifier, self).__init__()
    config = BertConfig.from_pretrained(PRE_TRAINED_MODEL_NAME)
    config.num_labels = 3
    config.return_dict = False
    self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME, config=config)
    self.drop = torch.nn.Dropout(p=0.3)
    #The last_hidden_state is a sequence of hidden states of the last layer of the model
    self.out = torch.nn.Linear(self.bert.config.hidden_size, n_classes)
  
  def forward(self, input_ids, attention_mask):
    _, pooled_output = self.bert(
      input_ids=input_ids,
      attention_mask=attention_mask
    )
    output = self.drop(pooled_output)
    return self.out(output)

def train_epoch(
  model, 
  data_loader, 
  loss_fn, 
  optimizer, 
  device, 
  scheduler, 
  n_examples
):
  model = model.train()

  losses = []
  correct_predictions = 0
  
  for d in data_loader:
    input_ids = d["input_ids"].to(device)
    attention_mask = d["attention_mask"].to(device)
    targets = d["targets"].to(device)

    outputs = model(
      input_ids=input_ids,
      attention_mask=attention_mask
    )

    _, preds = torch.max(outputs, dim=1)
    loss = loss_fn(outputs, targets)

    correct_predictions += torch.sum(preds == targets)
    losses.append(loss.item())

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()

  return correct_predictions.double() / n_examples, np.mean(losses)

def eval_model(model, data_loader, loss_fn, device, n_examples):
  model = model.eval()

  losses = []
  correct_predictions = 0

  with torch.no_grad():
    for d in data_loader:
      input_ids = d["input_ids"].to(device)
      attention_mask = d["attention_mask"].to(device)
      targets = d["targets"].to(device)

      outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask
      )
      _, preds = torch.max(outputs, dim=1)

      loss = loss_fn(outputs, targets)

      correct_predictions += torch.sum(preds == targets)
      losses.append(loss.item())

  return correct_predictions.double() / n_examples, np.mean(losses)

def get_predictions(model, data_loader):
  model = model.eval()
  review_texts = []
  predictions = []
  prediction_probs = []
  real_values = []
  with torch.no_grad():
    for d in data_loader:
      texts = d["review_text"]
      input_ids = d["input_ids"].to(device)
      attention_mask = d["attention_mask"].to(device)
      targets = d["targets"].to(device)
      outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask
      )
      _, preds = torch.max(outputs, dim=1)
      review_texts.extend(texts)
      predictions.extend(preds)
      prediction_probs.extend(outputs)
      real_values.extend(targets)
  predictions = torch.stack(predictions).cpu()
  prediction_probs = torch.stack(prediction_probs).cpu()
  real_values = torch.stack(real_values).cpu()
  return review_texts, predictions, prediction_probs, real_values

def read_pickle(name):
    one_instance = None
    with (open(name,'rb')) as openfile:
        while True:
            try:
                one_instance = pickle.load(openfile)
            except EOFError:
                break
    print(type(one_instance))
    one_instance = one_instance
    return one_instance

data = pd.read_csv('dataset_atualizado_08022022.csv')
data.loc[data['rotulacao_manual'] == -1, 'rotulacao_manual'] = 2

data.drop_duplicates(['comentarios'], inplace=True)
data=data.sample(frac=1).reset_index(drop=True)

#data = data[:10]

PRE_TRAINED_MODEL_NAME = 'neuralmind/bert-large-portuguese-cased'
#PRE_TRAINED_MODEL_NAME = 'bert-base-multilingual-cased'

tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)

token_lens = []

for txt in data.comentarios:
  tokens = tokenizer.encode(txt, max_length=512)
  token_lens.append(len(tokens))

df_train, df_test = train_test_split(data, test_size=0.2, random_state=RANDOM_SEED)

"""pickle.dump(df_train, open('/home/ubuntu/marcos/pickle_multi/df_train_bert_1003.pickle', 'wb'))
pickle.dump(df_test, open('/home/ubuntu/marcos/pickle_multi/df_test_bert_1003.pickle', 'wb'))"""


"""df_train = read_pickle('pickle/df_train_bert_1003.pickle')
df_test = read_pickle('pickle/df_test_bert_1003.pickle')"""

BATCH_SIZE = 16

train_data_loader = create_data_loader(df_train, tokenizer, MAX_LEN, BATCH_SIZE)
test_data_loader = create_data_loader(df_test, tokenizer, MAX_LEN, BATCH_SIZE)

dt = next(iter(train_data_loader))

class_names = ['negative', 'neutral', 'positive']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = SentimentClassifier(len(class_names))
model = model.to(device)

EPOCHS = 10

optimizer = AdamW(model.parameters(), lr=3e-5, correct_bias=False)
total_steps = len(train_data_loader) * EPOCHS

scheduler = get_linear_schedule_with_warmup(
  optimizer,
  num_warmup_steps=0,
  num_training_steps=total_steps
)

loss_fn = torch.nn.CrossEntropyLoss().to(device)

history = defaultdict(list)
best_accuracy = 0

for epoch in range(EPOCHS):

  print(f'Epoch {epoch + 1}/{EPOCHS}')
  print('-' * 10)

  train_acc, train_loss = train_epoch(
    model,
    train_data_loader,    
    loss_fn, 
    optimizer, 
    device, 
    scheduler, 
    len(df_train)
  )

  print(f'Train loss {train_loss} accuracy {train_acc}')

  
  history['train_acc'].append(train_acc)
  history['train_loss'].append(train_loss)


  if train_acc > best_accuracy:
    torch.save(model.state_dict(),'geral_BERTimabu_base_atuali_best_model_state.bin')
    best_accuracy = train_acc

test_acc, _ = eval_model(
  model,
  test_data_loader,
  loss_fn,
  device,
  len(df_test)
)

y_review_texts, y_pred, y_pred_probs, y_test = get_predictions(
  model,
  test_data_loader
)

print(classification_report(y_test, y_pred))

report = classification_report(y_test, y_pred, output_dict=True)
metricas = pd.DataFrame(report).transpose()

def metricas_auc_roc(y_score, y_test):
  #y_score = modelo.predict_proba(X_test)

  y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
  n_classes = y_test_bin.shape[1]

  fpr = dict()
  tpr = dict()
  roc_auc = dict()
  result = []

  for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    result.append(auc(fpr[i], tpr[i]))
  return result

def class_3_confunsion_matrix(y_test, predictions, classe):
  matrix = confusion_matrix(y_test, predictions)

  if classe == -1:
    TP = matrix[0][0]
    FN = matrix[0][1] + matrix[0][2]
    FP = matrix[1][0] + matrix[2][0]
    TN = matrix[1][1] + matrix[1][2] + matrix[2][1] + matrix[2][2]
  elif classe == 0:
    TP = matrix[1][1]
    FN = matrix[1][0] + matrix[1][2]
    FP = matrix[0][1] + matrix[2][1]
    TN = matrix[0][0] + matrix[0][2] + matrix[2][0] + matrix[2][2]
  elif classe == 1:
    TP = matrix[2][2]
    FN = matrix[2][0] + matrix[2][1]
    FP = matrix[0][2] + matrix[1][2]
    TN = matrix[0][0] + matrix[0][1] + matrix[1][0] + matrix[1][1]
  
  return [TP, FN, FP, TN]

def metricas(predictions, modelo_nome, y_test, df):
  #predictions = modelo.predict(X_test)

  report = classification_report(y_test, predictions, output_dict=True)
  metricas = pd.DataFrame(report).transpose()

  aucRoc = metricas_auc_roc(y_pred_probs, y_test)

  metricas_neg = list(metricas.iloc[[2]].values[0])
  metricas_neg.append(aucRoc[2])
  metricas_neg = metricas_neg + class_3_confunsion_matrix(y_test, predictions, -1)

  metricas_pos = list(metricas.iloc[[1]].values[0])
  metricas_pos.append(aucRoc[1])
  metricas_pos = metricas_pos + class_3_confunsion_matrix(y_test, predictions, 1)
  
  metricas_neu = list(metricas.iloc[[0]].values[0])
  metricas_neu.append(aucRoc[0])
  metricas_neu = metricas_neu + class_3_confunsion_matrix(y_test, predictions, 0)

  result = metricas_neg + metricas_pos + metricas_neu
  result = [modelo_nome] + result

  result.append(accuracy_score(y_test, predictions))

  df_aux = pd.DataFrame([result], columns=['modelo', 'precision(-1)', 'recall(-1)', 'f1-score(-1)', 'suport(-1)', 'auc-roc(-1)',  'TP(-1)', 'FN(-1)', 'FP(-1)', 'TN(-1)', 'precision(1)', 'recall(1)', 'f1-score(1)', 'suport(1)', 'auc-roc(1)', 'TP(1)', 'FN(1)', 'FP(1)', 'TN(1)', 'precision(0)', 'recall(0)', 'f1-score(0)', 'suport(0)', 'auc-roc(0)', 'TP(0)', 'FN(0)', 'FP(0)', 'TN(0)', 'accuracy'])

  return pd.concat([df_aux, df])

df_metricas = pd.DataFrame(columns=['modelo', 'precision(-1)', 'recall(-1)', 'f1-score(-1)', 'suport(-1)', 'auc-roc(-1)',  'TP(-1)', 'FN(-1)', 'FP(-1)', 'TN(-1)', 'precision(1)', 'recall(1)', 'f1-score(1)', 'suport(1)', 'auc-roc(1)', 'TP(1)', 'FN(1)', 'FP(1)', 'TN(1)', 'precision(0)', 'recall(0)', 'f1-score(0)', 'suport(0)', 'auc-roc(0)', 'TP(0)', 'FN(0)', 'FP(0)', 'TN(0)', 'accuracy'])
df_metricas = metricas(y_pred, 'Rede-neural-BERTimbau', y_test, df_metricas)

df_metricas.to_csv('metricas_rede_neural_BERTimbau_base.csv')
