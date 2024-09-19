import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import itertools
import math
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
from sklearn import metrics
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.model_selection import  GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.svm import SVC
import matplotlib.ticker as mtick
###########################################################################
df = pd.read_csv('diabetes.csv')
#We see that only the 'Build year', 'Position', 'Garden' and 'Estimated neighbourhood price per m2' columns have null values.
df.info()
pd.DataFrame(df.isnull().sum(), columns= ['Number of missing values'])
# Stampa il numero di righe
print("Numero di righe nel dataset:", len(df))
###########################################################################
# Prendo solo le prime 300 righe
data_subset = df.head(300)
###########################################################################
# Seleziona le colonne per la rimozione dei valori 0
columns_to_check = [col for col in df.columns if col not in ['Pregnancies', 'Outcome', 'Insulin']]

# Eliminazione delle righe che contengono il valore 0 nelle colonne selezionate
df = df.loc[~(df[columns_to_check] == 0).any(axis=1)]

# Eliminazione della colonna "Insulin"
df = df.drop('Insulin', axis=1)
# Stampa il numero di righe
print("Numero di righe nel dataset dopo aver eliminato delle righe con valori uguali 0:", len(df))
df.info()
###########################################################################
#creazione della matrice di correlazione
sns.set(font_scale = 1)
fig, ax = plt.subplots(figsize=(15,10))  
sns.heatmap(df.corr(), annot = True, ax=ax);

#Per vedere le correlazioni in modo ordinato della colonna selezionata
print(df.corr().Outcome.sort_values(ascending = False))


#grafico a barre per le varie correlazioni con la colonna delL'Outcome
corr = df.corr()
corr = corr.Outcome
cr = corr.sort_values(ascending = False)[1:]
sns.barplot(x=cr, y=cr.index,palette = "bright")
plt.title("Correlazione tra attributi e Outcome")
############################################################################################################
#EDA
#Grafico per il glucosio
sns.set(rc={"figure.dpi": 100})
sns.set_context('paper')
sns.set_style("ticks")
fig = plt.figure(figsize=(15, 12))
gs = fig.add_gridspec(2, 2)
gs.update(wspace=0.3, hspace=0.4)
fig.text(0.085, 0.95, 'Glucosio', fontfamily='serif', fontsize=15, fontweight='bold')
sns.set_palette('plasma')
ax0 = fig.add_subplot(gs[0, 0])
ax1 = fig.add_subplot(gs[0, 1])
ax2 = fig.add_subplot(gs[1, 0], ylim=(0, 210), xlim=(0, 5))
ax3 = fig.add_subplot(gs[1, 1], ylim=(0, 210))
ax0.grid(color='gray', linestyle=':', axis='y', zorder=0, dashes=(1, 5))
sns.kdeplot(x='Glucose',
            data=df,
            shade=True,
            ax=ax0,
            linewidth=0
            )
ax0.yaxis.set_major_formatter(mtick.PercentFormatter(1, decimals=3))
ax0.set_xlabel("")
ax0.set_ylabel("")
ax0.set_title('Distribuzione del glucosio', fontsize=12, fontfamily='serif', fontweight='bold', x=0, y=1)
ax0.spines['top'].set_visible(False)
ax0.spines['right'].set_visible(False)
ax0.spines['left'].set_visible(False)

# Glucosio vs outcome
ax1.scatter(df['Glucose'], df['Outcome'], color='purple', alpha=0.5)
ax1.set_xlabel("Glucose", fontsize=12, fontfamily='serif', fontweight='bold')
ax1.set_ylabel("Outcome", fontsize=12, fontfamily='serif', fontweight='bold')
ax1.set_title("Distribuzione del glucosio rispetto all'outcome", fontsize=12, fontfamily='serif', fontweight='bold')
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)
ax1.spines['left'].set_visible(False)

ax2.grid(color='gray', linestyle=':', axis='y', dashes=(1, 5))
sns.boxenplot(y='Glucose',
              data=df,
              ax=ax2,
              linewidth=0.4)
ax2.set_xlabel("")
ax2.set_ylabel("")
ax2.set_title('Distribuzione del glucosio', fontsize=12, fontfamily='serif', fontweight='bold', x=0, y=1.15)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_visible(False)
ax3.grid(color='gray', linestyle='-', axis='y', dashes=(1, 5))
sns.boxenplot(x='Outcome',
              y='Glucose',
              data=df,
              ax=ax3,
              linewidth=0.4
              )
ax3.set_xlabel("")
ax3.set_ylabel("")
ax3.set_title("Distribuzione rispetto all'outcome", fontsize=12, fontfamily='serif', fontweight='bold', x=0.1, y=1.15)
ax3.set_xticklabels([' 0', '1 '])
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.spines['left'].set_visible(False)

plt.show()
##############################################################
#EDA
#Grafico pressione sanguigna
sns.set(rc={"figure.dpi": 100})
sns.set_context('paper')
sns.set_style("ticks")
fig = plt.figure(figsize=(15, 12))
gs = fig.add_gridspec(1, 2)
gs.update(wspace=0.3, hspace=0.4)
sns.set_palette('plasma')

ax2 = fig.add_subplot(gs[0, 0], ylim=(0, 120))
ax3 = fig.add_subplot(gs[0, 1], ylim=(0, 120))

ax2.grid(color='gray', linestyle=':', axis='y', dashes=(1, 5))
sns.boxenplot(y='BloodPressure',
              data=df,
              ax=ax2,
              linewidth=0.4)
ax2.set_xlabel("")
ax2.set_ylabel("")
ax2.set_title("Distribuzione della pressione sanguigna", fontsize=12, fontfamily='serif', fontweight='bold', x=0.1, y=1.15)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_visible(False)


ax3.grid(color='gray', linestyle='-', axis='y', dashes=(1, 5))
sns.boxenplot(x='Outcome',
              y='BloodPressure',
              data=df,
              ax=ax3,
              linewidth=0.4
              )
ax3.set_xlabel("")
ax3.set_ylabel("")
ax3.set_title("Distribuzione rispetto all'outcome", fontsize=12, fontfamily='serif', fontweight='bold', x=0.1, y=1.15)
ax3.set_xticklabels([' 0', '1 '])
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.spines['left'].set_visible(False)

plt.show()
##############################################################
#EDA
#Grafico ETà
sns.set(rc={"figure.dpi": 100})
sns.set_context('paper')
sns.set_style("ticks")
fig = plt.figure(figsize=(15, 12))
gs = fig.add_gridspec(1, 2)
gs.update(wspace=0.3, hspace=0.4)
sns.set_palette('plasma')

ax2 = fig.add_subplot(gs[0, 0], ylim=(20, 90))
ax3 = fig.add_subplot(gs[0, 1], ylim=(20, 90))

ax2.grid(color='gray', linestyle=':', axis='y', dashes=(1, 5))
sns.boxenplot(y='Age',
              data=df,
              ax=ax2,
              linewidth=0.4)
ax2.set_xlabel("")
ax2.set_ylabel("")
ax2.set_title("Distribuzione dell'età", fontsize=12, fontfamily='serif', fontweight='bold', x=0.1, y=1.15)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['left'].set_visible(False)


ax3.grid(color='gray', linestyle='-', axis='y', dashes=(1, 5))
sns.boxenplot(x='Outcome',
              y='Age',
              data=df,
              ax=ax3,
              linewidth=0.4
              )
ax3.set_xlabel("")
ax3.set_ylabel("")
ax3.set_title("Distribuzione rispetto all'outcome", fontsize=12, fontfamily='serif', fontweight='bold', x=0.1, y=1.15)
ax3.set_xticklabels([' 0', '1 '])
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)
ax3.spines['left'].set_visible(False)

plt.show()
##############################################################

#Split dataset FEATURE NON SELEZIONATE 80-20
# Estrai le features (tutte le colonne tranne "Outcome") e il target (colonna "Outcome")
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Divisione del dataset in training set e test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
###########################################################################
#Addestramento dei modelli con feature non selezionate 

#SVM
SVM = SVC()
SVM.fit(X_train,y_train)
y_pred2 = SVM.predict(X_test)
y_pred_train2 = SVM.predict(X_train)

# Calcolo dello score del modello SVM sul set di test
SVM_score = SVM.score(X_test, y_test)
print("Score SVM:", SVM_score)
# Ottieni le etichette uniche da y_test
labels = np.unique(y_test)

# Calcola la matrice di confusione con le etichette automatiche
cm = confusion_matrix(y_test, y_pred2, labels=labels)
conf_matrix = pd.DataFrame(data=cm, columns=labels, index=labels)
plt.figure(figsize = (8,5))
sns.heatmap(conf_matrix, annot=True,fmt='d',cmap="plasma")
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion MatrIX SVM')
plt.show()
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
result = classification_report(y_test, y_pred2)
print("Risultati della parte test")
print(result)


MR = round(np.mean(y_pred2!=y_test)*100, 2)
print(MR, '%')
###########################################################################
#Regressione logistica
Logistic_model= LogisticRegression()
Logistic_model.fit(X_train,y_train)
# Calcolo delle previsioni sul set di test
y_pred = Logistic_model.predict(X_test)
# Calcolo della matrice di confusione
cm = confusion_matrix(y_test, y_pred)

# Creazione della heatmap della matrice di confusione
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix Logistic Regresion')
plt.show()
result = Logistic_model.score(X_test,y_test)
print("Accuracy:",result)
MR = round(np.mean(y_pred!=y_test)*100, 2)
print(MR, '%')
################################################
#Hyperparameter Tuning SVM
# Definizione dei parametri da testare
parameters = {'kernel': ['linear', 'rbf','poly'], 'C': [0.1,0.5,1,2,3,4,5,6,7,8,9,10]}

# Creazione del modello SVM
svm_model = svm.SVC()

# Creazione dell'oggetto GridSearchCV
grid_search = GridSearchCV(svm_model, parameters,cv=9)

# Esecuzione del grid search
grid_search.fit(X, y)

# Stampa dei risultati
print("Miglior set di iperparametri trovato:")
print(grid_search.best_params_)
print("Miglior accuratezza:", grid_search.best_score_)

# Recupero dei risultati del grid search
results = grid_search.cv_results_

# Recupero dei parametri testati
param_values = results['param_C']
param_kernel = results['param_kernel']

# Recupero delle metriche valutate
mean_test_score = results['mean_test_score']

# Creazione del grafico
plt.figure(figsize=(10, 6))
plt.title('Accuratezza per ogni iperparametro')
plt.xlabel('Valori di C')
plt.ylabel('Accuracy')
plt.grid(True)

# Plot delle linee separate per il tipo di kernel
for kernel in set(param_kernel):
    kernel_scores = mean_test_score[param_kernel == kernel]
    color = 'blue' if kernel == 'linear' else 'green' if kernel == 'poly' else 'red'
    plt.plot(param_values[param_kernel == kernel], kernel_scores, 'o-', label=kernel, color=color)
# Legenda
plt.legend()

# Mostra il grafico
plt.show()

###########################################################################

#Split dataset FEATURE NON SELEZIONATE utilizzo il dataset più piccolo per poter fare hyperparameter tuning
# Estrai le features (tutte le colonne tranne "Price") e il target (colonna "Price")
X_subset = data_subset.drop("Outcome", axis=1)
y_subset = data_subset["Outcome"]

# Divisione del dataset in training set e test set
X_subset_train, X_subset_test, y_subset_train, y_subset_test = train_test_split(X_subset, y_subset, test_size=0.3, random_state=42)
###################################################################
#statistica descrittiva ed inferenziale con una parte del dataset 
k = 50
accuracy_scores = []
random_state = 40
for _ in range(k):
    # Dividi il dataset in train set e test set
    X_train2, X_test2, y_train2, y_test2 = train_test_split(X, y, test_size=0.2, random_state=random_state)
    
    # Addestra il modello
    svm1 = SVC(C=5,kernel='linear',random_state=random_state)
    svm1.fit(X_train2,y_train2)
    y_pred = svm1.predict(X_test2)
    accuracy = metrics.accuracy_score(y_test2, y_pred2)
    
    # Aggiungi l'accuracy score alla lista
    accuracy_scores.append(accuracy)
    
    random_state += 1

# Calcola le statistiche descrittive
mean_accuracy = np.mean(accuracy_scores)
std_accuracy = np.std(accuracy_scores)
confidence_interval = stats.t.interval(0.95, len(accuracy_scores)-1, loc=mean_accuracy, scale=stats.sem(accuracy_scores))


# Stampa i risultati
print("Mean Accuracy:", mean_accuracy)
print("Standard Deviation of Accuracy:", std_accuracy)
print("Confidence Interval (95%):", confidence_interval)
title = "Boxplot per k uguale a {}".format(k)


#grafico boxplot finale
plt.boxplot(accuracy_scores)
plt.title("Boxplot per k uguale a {first}".format(first=k))
plt.show()