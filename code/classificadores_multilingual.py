# -*- coding: utf-8 -*-

"""Códogo referente as classificadores SVM e RF usando a matriz de embeddings gerada pelo Multlingual, e com smote"""

"""pip install -U sentence-transformers"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm

from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import pandas as pd
import re

import imblearn
from collections import Counter
import nltk

"""## Aquisição dos dados e Pré-processamento """

data = pd.read_csv('dataset_atualizado_08022022.csv')

data.drop_duplicates(['comentarios'], inplace=True)

id = (data['id'].copy(deep=True))
comentarios = (data['comentarios'].copy(deep=True))
classes = (data['rotulacao_manual'].copy(deep=True))

id = id.values
comentarios = comentarios.values
classes = classes.values

nltk.download('stopwords')
nltk.download('rslp')
nltk.download('punkt')
nltk.download('wordnet')

def Preprocessing(instancia):
    #stemmer = nltk.stem.RSLPStemmer()
    instancia = re.sub(r"http\S+", "", instancia).lower().replace('.','').replace(';','').replace('-','').replace(':','').replace(')','')
    #stopwords = set(nltk.corpus.stopwords.words('portuguese'))
    palavras = [i for i in instancia.split()]
    return (" ".join(palavras))

# Aplica a função em todos os dados:
comentarios = [Preprocessing(i) for i in comentarios]

"""## Embedding"""

sentence_model = SentenceTransformer("bert-base-multilingual-cased")
embeddings = sentence_model.encode(comentarios, show_progress_bar=True)

df_embedding = pd.DataFrame(embeddings)


"""## função para salvar o modelo"""

import pickle

def salvModel(modelo, nome, X_train, y_train, X_test, y_test):
  y_pred = modelo.predict(X_test)
  pickle.dump(X_train, open('B_1005/' + nome + 'X_train_bert_0905_M.pickle', 'wb'))
  pickle.dump(y_train, open('B_1005/' + nome + 'y_train_bert_0905_M.pickle', 'wb'))
  pickle.dump(X_test, open('B_1005/' + nome + 'X_test_bert_0905_M.pickle', 'wb'))
  pickle.dump(y_test, open('B_1005/' + nome + 'y_test_bert_0905_M.pickle', 'wb'))
  pickle.dump(y_pred, open('B_1005/' + nome + 'y_pred_bert_0905_M.pickle', 'wb'))
  pickle.dump(modelo, open('B_1005/' + nome + '_0506.pickle', 'wb'))

"""## Random Forest"""

counter = Counter(classes)
print(counter)

X_train, X_test, y_train, y_test = train_test_split(df_embedding, classes, test_size = 0.2, random_state=42, shuffle=False)

counter = Counter(y_train)
print(counter)

counter = Counter(y_test)
print(counter)

"""#### GridSearch RF"""

def grid(X_train, y_train):
  rf_clf = RandomForestClassifier()
  param_grid = {'n_estimators': [250, 300, 350],'max_depth' : [8,10, 12], 'bootstrap': [False], 'n_jobs': [-1]}
  grid_search = GridSearchCV(rf_clf, param_grid, cv=5, scoring='accuracy')

  grid_search.fit(X_train, y_train)

  return grid_search.best_params_

#grid(X_train, y_train)

rf_clf = RandomForestClassifier(max_depth=12, n_estimators= 300, n_jobs= -1, bootstrap=False)
rf_clf.fit(X_train,y_train)

salvModel(rf_clf, 'rf', X_train, y_train, X_test, y_test)

"""#### Métricas"""

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

def metricas_auc_roc(modelo, X_test, y_test):
  y_score = modelo.predict_proba(X_test)

  y_test_bin = label_binarize(y_test, classes=[-1, 0, 1])
  n_classes = y_test_bin.shape[1]

  fpr = dict()
  tpr = dict()
  roc_auc = dict()
  result = []

  for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
    result.append(auc(fpr[i], tpr[i]))
  return result

metricas_auc_roc(rf_clf, X_test, y_test)

predictions = rf_clf.predict(X_test)
print(classification_report(y_test, predictions))

def metricas(modelo, modelo_nome, X_test, y_test, df):
  predictions = modelo.predict(X_test)

  report = classification_report(y_test, predictions, output_dict=True)
  metricas = pd.DataFrame(report).transpose()

  aucRoc = metricas_auc_roc(modelo, X_test, y_test)

  metricas_neg = list(metricas.iloc[[0]].values[0])
  metricas_neg.append(aucRoc[0])
  metricas_neg = metricas_neg + class_3_confunsion_matrix(y_test, predictions, -1)

  metricas_pos = list(metricas.iloc[[2]].values[0])
  metricas_pos.append(aucRoc[2])
  metricas_pos = metricas_pos + class_3_confunsion_matrix(y_test, predictions, 1)
  
  metricas_neu = list(metricas.iloc[[1]].values[0])
  metricas_neu.append(aucRoc[1])
  metricas_neu = metricas_neu + class_3_confunsion_matrix(y_test, predictions, 0)

  result = metricas_neg + metricas_pos + metricas_neu
  result = [modelo_nome] + result

  result.append(accuracy_score(y_test, predictions))

  df_aux = pd.DataFrame([result], columns=['modelo', 'precision(-1)', 'recall(-1)', 'f1-score(-1)', 'suport(-1)', 'auc-roc(-1)',  'TP(-1)', 'FN(-1)', 'FP(-1)', 'TN(-1)', 'precision(1)', 'recall(1)', 'f1-score(1)', 'suport(1)', 'auc-roc(1)', 'TP(1)', 'FN(1)', 'FP(1)', 'TN(1)', 'precision(0)', 'recall(0)', 'f1-score(0)', 'suport(0)', 'auc-roc(0)', 'TP(0)', 'FN(0)', 'FP(0)', 'TN(0)', 'accuracy'])

  return pd.concat([df_aux, df])

  #print(metricas)
  #print(list(metricas.iloc[[1]].values))

df_metricas = pd.DataFrame(columns=['modelo', 'precision(-1)', 'recall(-1)', 'f1-score(-1)', 'suport(-1)', 'auc-roc(-1)',  'TP(-1)', 'FN(-1)', 'FP(-1)', 'TN(-1)', 'precision(1)', 'recall(1)', 'f1-score(1)', 'suport(1)', 'auc-roc(1)', 'TP(1)', 'FN(1)', 'FP(1)', 'TN(1)', 'precision(0)', 'recall(0)', 'f1-score(0)', 'suport(0)', 'auc-roc(0)', 'TP(0)', 'FN(0)', 'FP(0)', 'TN(0)', 'accuracy'])
df_metricas = metricas(rf_clf, 'RF-Multilingual', X_test, y_test, df_metricas)

df_metricas

df_metricas[['f1-score(-1)', 'f1-score(1)', 'f1-score(0)']]

"""#### SMOTE RF"""

from imblearn.over_sampling import SMOTE
sm = SMOTE()
X_res, y_res = sm.fit_resample(X_train, y_train)

counter = Counter(y_res)
print(counter)

#grid(X_res, y_res)

rf_clf = RandomForestClassifier(max_depth=12, n_estimators= 300, n_jobs= -1, bootstrap=False)
rf_clf.fit(X_res,y_res)

salvModel(rf_clf, 'rf_smote', X_res, y_res, X_test, y_test)

df_metricas = metricas(rf_clf, 'RF-Multilingual-SMOTE', X_test, y_test, df_metricas)

df_metricas[['modelo','f1-score(-1)', 'f1-score(1)', 'f1-score(0)']]


"""## SVM"""


"""#### GridSeach SVM"""

def grid_svm(X_train, y_test):
  SupportVM = svm.SVC()

  tuned_parameters = [
    {"kernel": ["rbf"], "gamma": [1e-3, 1e-4], "C": [1, 10, 100, 1000]},
    {"kernel": ["linear"], "C": [1, 10, 100, 1000]},
  ]

  grid_search = GridSearchCV(svm.SVC(), tuned_parameters, cv=5, scoring='accuracy')
  grid_search.fit(X_train, y_train)

  return grid_search.best_params_

#grid_svm(X_train, y_test)

#SupportVM = svm.SVC(kernel='rbf', C=10, gamma=0.001, probability=True)
SupportVM = svm.SVC(kernel='linear', C=10, probability=True)
SupportVM.fit(X_train ,y_train)

salvModel(SupportVM, 'svm', X_train, y_train, X_test, y_test)

from tqdm import tqdm

import seaborn as sns
import matplotlib.pyplot as plt

y_pred = SupportVM.predict(X_test)



def show_confusion_matriz(confusion_matrix):
  hmap = sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues")
  sns.set(font_scale=1.5)
  hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=16)
  hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=20, ha='right', fontsize=16)
  plt.ylabel('True sentiment', fontsize=16)
  plt.xlabel('Predicted sentiment', fontsize=16);
  plt.savefig('matriz_conf_svm.pdf', dpi=300)
  
  
class_names = ['desaprova', 'neutro', 'aprova']
y_pred = SupportVM.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
df_cm = pd.DataFrame(cm, index=class_names, columns=class_names)
show_confusion_matriz(df_cm)

df_metricas = metricas(SupportVM, 'SVM-Multilingual', X_test, y_test, df_metricas)

df_metricas

"""#### SVM SMOTE"""

from imblearn.over_sampling import SVMSMOTE
sm = SVMSMOTE()
X_res, y_res = sm.fit_resample(X_train, y_train)

counter = Counter(y_res)
print(counter)

#grid_svm(X_res, y_res)

SupportVM = svm.SVC(kernel='rbf', C=10, gamma=0.001, probability=True)
SupportVM.fit(X_res ,y_res)

salvModel(SupportVM, 'svm_smote', X_res, y_res, X_test, y_test)

df_metricas = metricas(SupportVM, 'SVM-Multilingual-SMOTE', X_test, y_test, df_metricas)

df_metricas

df_metricas.to_csv('metricasMultilingual0506.csv')