#!/usr/bin/env python
# coding: utf-8

# In[415]:


import pandas as pd, numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
import sklearn
from sklearn.linear_model import LogisticRegression
import dython
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, confusion_matrix
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree


# In[416]:


train = pd.read_csv('titanic_train.csv')
test_fin = pd.read_csv('titanic_test.csv')


# In[417]:


train.head()


# In[418]:


cols = list(train.columns)


# In[419]:


cols


# In[420]:


cols.remove('Survived')


# In[421]:


train[train.columns[1:]].corr()['Survived'][:]


# In[422]:


Y = train['Survived']
X = train[cols]


# In[423]:


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42, stratify=Y)


# In[424]:


import pandas as pd
from plotnine import *

get_ipython().run_line_magic('matplotlib', 'inline')


# In[425]:


def wykres_czestosci(tab,nazwa_zmien):
    return (ggplot(tab, aes(nazwa_zmien, fill= nazwa_zmien, color= nazwa_zmien))
     + geom_bar()
     + geom_text(
        aes(label='stat(prop)*100', group=1),
        stat='count',
        nudge_y=0.125,
        va='bottom',
        format_string='{:.1f}%'
    )
    +ggtitle('Wykres częstości zmiennej '+nazwa_zmien)
    )


# In[426]:


X_train.describe()


# In[427]:


X_train.info()


# In[ ]:





# In[428]:


X_train['Name'].head(10)


# In[429]:


#Tworzenie zmiennej nazwisko, przedrostek  czy kobieta jest zamężna


# In[430]:


X_train.loc[:,'nazwisko'] = X_train.apply(lambda x: x['Name'][:x['Name'].index(',')], axis=1)
X_test.loc[:,'nazwisko'] = X_test.apply(lambda x: x['Name'][:x['Name'].index(',')], axis=1)


# In[431]:


X_train['przedrostek'] = X_train.apply(lambda x: x['Name'][x['Name'].index(',')+2:x['Name'].index('.')], axis=1)
X_test['przedrostek'] = X_test.apply(lambda x: x['Name'][x['Name'].index(',')+2:x['Name'].index('.')], axis=1)


# In[432]:


np.unique(X_train['przedrostek'])


# In[433]:


np.unique(X_test['przedrostek'])


# In[434]:


def zamiana_przedostków(x):
    przedrostek = x['przedrostek']
    if przedrostek in ['Capt', 'Col', 'Don', 'Jonkheer', 'Major', 'Rev', 'Sir']:
        return 'Mr'
    elif przedrostek in ['the Countess', 'Mme', 'Lady']:
        return 'Mrs'
    elif przedrostek in ['Mlle', 'Ms']:
        return 'Miss'
    elif przedrostek =='Dr':
        if x['Sex']=='male':
            return 'Mr'
        else:
            return 'Mrs'
    else:
        return przedrostek


# In[435]:


X_train['przedrostek'] = X_train.apply(zamiana_przedostków, axis=1)
X_test['przedrostek'] = X_test.apply(zamiana_przedostków, axis=1)


# In[436]:


wykres_czestosci(X_train, 'przedrostek')


# In[437]:


#master niezamężny
# mr męzczyzna
# miss nizamężna
# mrs zamężna


# In[438]:


X_train['zamężna kobieta'] = X_train.apply( lambda x: 1 if x['przedrostek'] == 'Mrs' else 0, axis=1)
X_test['zamężna kobieta'] = X_test.apply( lambda x: 1 if x['przedrostek'] == 'Mrs' else 0, axis=1)


# In[439]:


# Nazwisko jest zbyt zróżnicowaną zmienną, usuwam, tak samo jak name


# In[440]:


X_train = X_train.drop(columns=['nazwisko','Name'])
X_test = X_test.drop(columns=['nazwisko','Name'])


# In[441]:


# tworzę zmienną czy dziecko mówiącą czy podróżujący ma poniżej 18 lat
X_train.loc[:,'Czy dziecko'] = X_train.apply(lambda x: 1 if x['Age']<18 else 0 , axis=1)
X_test.loc[:,'Czy dziecko'] = X_test.apply(lambda x: 1 if x['Age']<18 else 0 , axis=1)


# In[442]:


# nowa zmienna mówiąca o ilości osób podróżujących na tym samym bilecie
X_train['No_ppl_on_ticket'] = X_train.groupby(['Ticket'])['Ticket'].transform('count')
X_test['No_ppl_on_ticket'] = X_test.groupby(['Ticket'])['Ticket'].transform('count')

X_test['No_ppl_on_ticket'] = X_test['No_ppl_on_ticket'].astype('object')
X_train['No_ppl_on_ticket'] = X_train['No_ppl_on_ticket'].astype('object')


# In[443]:


wykres_czestosci(X_train, 'No_ppl_on_ticket')


# In[444]:


# redukuje mało liczne kategorie 
def liczba_osob_na_bil(x):
    if x['No_ppl_on_ticket'] >= 4:
        return '3+'
    else:
        return str(x['No_ppl_on_ticket'])


# In[445]:


X_test['No_ppl_on_ticket'] = X_test.apply(liczba_osob_na_bil, axis=1)
X_train['No_ppl_on_ticket'] = X_train.apply(liczba_osob_na_bil, axis=1)


# In[446]:


wykres_czestosci(X_train, 'No_ppl_on_ticket')


# In[447]:


# usuwam zmienną PassengerId, za mało mówi 
X_train.drop(columns=['PassengerId'], inplace = True)
X_test.drop(columns=['PassengerId'], inplace = True)


# In[448]:


# zmieniam typy zmiennych
X_train.loc[:,'Pclass'] = X_train.loc[:,'Pclass'].astype('object')
X_train.loc[:,'SibSp'] = X_train.loc[:,'SibSp'].astype('object')
X_train.loc[:,'Parch'] = X_train.loc[:,'Parch'].astype('object')


X_test.loc[:,'Parch'] = X_test.loc[:,'Parch'].astype('object')
X_test.loc[:,'Pclass'] = X_test.loc[:,'Pclass'].astype('object')
X_test.loc[:,'SibSp'] = X_test.loc[:,'SibSp'].astype('object')


# In[449]:


X_train.info()


# In[450]:


# printuje zmienne, i rozpoznaje które mają braki danych
for i in X_train.columns:
    print(i)
    print(X_train.loc[X_train[i].isnull(),i])


# In[451]:


np.unique(X_train['Cabin'].astype('str'))


# In[452]:


# uzupełniam braki danych w zmiennej Cabin i Embarked
X_train['Cabin'].fillna('OTHER', inplace=True)
X_test['Cabin'].fillna('OTHER', inplace=True)

X_train['Embarked'].fillna('OTHER', inplace = True)
X_test['Embarked'].fillna('OTHER', inplace = True)


# In[ ]:





# In[453]:


# zachowuje tylko pierwszą literę z  wartości jakie przyjmuje zmienna Cabin, ograiczając liczbę unikalnych wartości


# In[454]:


X_train['Cabin'] = X_train['Cabin'].str[:1]
X_test['Cabin'] = X_test['Cabin'].str[:1]

np.unique(X_train['Cabin'].str[:1])


# In[455]:


wykres_czestosci(X_train, 'Cabin')


# In[456]:


# jest sporo mało licznych kategorii, sprawdzam jak jeszcze bardziej je pogrupować,
# sprawdzam z czym najbardziej skorelowana jest zmienna i grupuje w większe grupy


# In[457]:


nom_col = list(X_train.select_dtypes('object').columns)
dython.nominal.associations(X_train, nominal_columns=nom_col)


# In[458]:


# występuje korelacja między kabiną a klasą, sprawdźmy grupowanie między dwoma cechami


# In[459]:


X_train.groupby(['Pclass','Cabin'])['Cabin'].aggregate('count')


# In[460]:


# wszystkie kabiny abc są w klasie pierwszej - grupuje w jedno, tak samo de i fg


# In[461]:


def group_cab(x):
    if x['Cabin'] in ('A','B','C'):
        return 'ABC'
    elif x['Cabin'] in ('D','E'):
        return 'DE'
    elif x['Cabin'] in ('F','G'):
        return 'FG'
    else:
        return 'O'


# In[462]:


X_train['Cabin'] = X_train.apply(group_cab, axis=1)
X_test['Cabin'] = X_test.apply(group_cab, axis=1)


# In[463]:


wykres_czestosci(X_train, 'Cabin')


# In[464]:


# sprawdzam wartości unikalne zmiennej Embarked, zmieniam jej typ na object


# In[465]:


np.unique(X_train['Embarked'].astype('str'))


# In[466]:


X_train['Embarked'] = X_train['Embarked'].astype('object')


# In[467]:


wykres_czestosci(X_train, 'Embarked')


# In[468]:


# zmienna wiek


# In[469]:


# wiek najbardziej skorelowany z kabiną i Pclass, wypełnię średnią w grupach
X_train.groupby(['Pclass','Cabin'])['Age'].mean()


# In[470]:


X_train['Age'] = X_train.groupby(['Pclass','Cabin'])['Age'].transform(lambda x: x.fillna(x.mean()))
X_test['Age'] = X_test.groupby(['Pclass','Cabin'])['Age'].transform(lambda x: x.fillna(x.mean()))


# In[471]:


plt.plot(X_train['Age'], 'r.')


# In[472]:


fig1, ax1 = plt.subplots()
ax1.set_title('Basic Plot')
ax1.boxplot(X_train['Age'])


# In[473]:


plt.hist(X_train['Age'], density=True, bins=30)


# In[474]:


# binuje co 15 lat 


# In[475]:


def age_binning(x):
    if x['Age'] <=15:
        return 'dziecko <=15'
    elif x['Age'] <= 30:
        return ' młody dorosły <=30'
    elif x['Age'] <= 45:
        return ' dorosły <=45'
    else:
        return ' starszy dorosły i osoby starsze 45+'
X_train['Age'] = X_train.apply(age_binning, axis=1)
X_test['Age'] = X_test.apply(age_binning, axis=1)


# In[476]:


wykres_czestosci(X_train, 'Age')


# In[477]:


# zmienna Fare, pozbycie się wartości odstających poprzez logarytm oraz winsorizing


# In[478]:


plt.plot(X_train['Fare'], 'b.')


# In[479]:


fig1, ax1 = plt.subplots()
ax1.set_title('Basic Plot')
ax1.boxplot(X_train['Fare'])


# In[480]:


plt.hist(X_train['Fare'], density=True, bins=30)


# In[481]:


plt.hist(np.log(X_train['Fare']+1), density=True, bins=30)


# In[482]:


fig1, ax1 = plt.subplots()
ax1.set_title('Basic Plot')
#ax1.boxplot(np.log(X_train['Fare']+1))
ax1.boxplot(np.log(X_train['Fare']+0.1))


# In[483]:


X_train['Fare_log'] = np.log(X_train['Fare']+0.1)
X_test['Fare_log'] = np.log(X_test['Fare']+0.1)


# In[484]:


# winsoriziong
def Fare_log_winsor(x):
    if x['Fare_log'] < np.percentile(X_train['Fare_log'],5):
        return np.percentile(X_train['Fare_log'],5)
    elif x['Fare_log'] > np.percentile(X_train['Fare_log'],95):
        return np.percentile(X_train['Fare_log'],95)
    else:
        return x['Fare_log']
X_train['Fare_log_winsor'] = X_train.apply(Fare_log_winsor, axis=1)

def Fare_log_winsor(x):
    if x['Fare_log'] < np.percentile(X_test['Fare_log'],5):
        return np.percentile(X_test['Fare_log'],5)
    elif x['Fare_log'] > np.percentile(X_test['Fare_log'],95):
        return np.percentile(X_test['Fare_log'],95)
    else:
        return x['Fare_log']
X_test['Fare_log_winsor'] = X_test.apply(Fare_log_winsor, axis=1)


# In[485]:


fig1, ax1 = plt.subplots()
ax1.set_title('Basic Plot')
ax1.boxplot(X_train['Fare_log_winsor'])


# In[486]:


# tworzę zmienną no_fam_members określającą ilość członków rodziny na statku


# In[487]:


X_train['no_fam_members'] = X_train['SibSp']+X_train['Parch']+1
X_test['no_fam_members'] = X_test['SibSp']+X_train['Parch']+1

X_train['no_fam_members'] = X_train['no_fam_members'].astype('object')
X_test['no_fam_members'] = X_test['no_fam_members'].astype('object')


# In[488]:


wykres_czestosci(X_train, 'no_fam_members')


# In[489]:


#zmniejszenie liczby kategorii przez grupowanie
def wielkosc_rodziny(x):
    if x['no_fam_members'] < 2:
        return 'singel'
    elif ((x['no_fam_members'] >= 2) and (x['no_fam_members'] <= 4)):
        return 'mała rodzina'
    else:
        return 'duza rodzina'

X_train['no_fam_members'] = X_train.apply(wielkosc_rodziny, axis=1)
X_test['no_fam_members'] = X_test.apply(wielkosc_rodziny, axis=1)


# In[490]:


wykres_czestosci(X_train, 'no_fam_members')


# In[491]:


X_train = X_train.drop(columns=['Fare','Fare_log','Ticket'])
X_test = X_test.drop(columns=['Fare','Fare_log','Ticket'])


# In[492]:


# zmiana typu zmiennych
X_train['SibSp'] = X_train['SibSp'].astype('object')
X_train['Parch'] = X_train['Parch'].astype('object')


# In[493]:


# korelacja


# In[494]:


nom_col = list(tab_corr.select_dtypes('object').columns)
dython.nominal.associations(tab_corr, nominal_columns=nom_col)['corr']['Survived']


# In[ ]:





# In[ ]:


# stnadaryzacja # dummys


# In[495]:


from sklearn.preprocessing import StandardScaler


# In[496]:


X_train[['Fare_log_winsor']] = StandardScaler().fit_transform(X_train[['Fare_log_winsor']])


# In[497]:


X_test[['Fare_log_winsor']] = StandardScaler().fit_transform(X_test[['Fare_log_winsor']])


# In[498]:


X_train['SibSp'] = X_train['SibSp'].astype('object')
X_train['Parch'] = X_train['Parch'].astype('object')


# In[499]:


X_train = pd.concat([X_train, pd.get_dummies(X_train[list(X_train.select_dtypes('object').columns)])], axis=1) 
X_train = X_train.drop(columns= list(X_train.select_dtypes('object').columns))


# In[500]:


X_test = pd.concat([X_test, pd.get_dummies(X_test[list(X_test.select_dtypes('object').columns)])], axis=1) 
X_test = X_test.drop(columns= list(X_test.select_dtypes('object').columns))


# In[ ]:





# In[501]:


X_train.columns


# In[502]:


X_train, X_test = X_train.align(X_test, join='left', axis=1)


# In[503]:


X_test.fillna(0, inplace=True)


# In[ ]:





# In[504]:


# teraz modele 


# In[505]:


#model regresji logistycznej


# In[506]:


clf = LogisticRegression(random_state=0).fit(X_train, y_train)

y_pred = clf.predict(X_test)

print('precision: '+str(precision_score(y_test, y_pred)))
print('recall: '+str(recall_score(y_test, y_pred)))
print('accuray: '+str(accuracy_score(y_test, y_pred)))
print('f1 score: '+str(f1_score(y_test, y_pred, average='weighted')))


# In[507]:


confusion_matrix(y_test, y_pred)


# In[ ]:





# In[508]:


# model drzewa decyzyjnego

clf = tree.DecisionTreeClassifier()
clf.fit(X_train, y_train)

spl_tree_pred = clf.predict(X_test)

print('precision: '+str(precision_score(y_test, spl_tree_pred)))
print('recall: '+str(recall_score(y_test, spl_tree_pred)))
print('accuray: '+str(accuracy_score(y_test, spl_tree_pred)))
print('f1 score: '+str(f1_score(y_test, spl_tree_pred, average='weighted')))


# In[509]:


tree.plot_tree(clf, 
                   feature_names=list(X_train.columns),  
                   class_names=['0','1'],
                   filled=True)


# In[510]:


# tuning drzewa


# In[511]:


params = {'max_leaf_nodes': list(range(2, 100)), 'min_samples_split': [2, 3, 4]
         , 'max_depth' : [2,4,6,8,10,12], 'criterion': ['gini', 'entropy']
         }
grid_search_cv = GridSearchCV(DecisionTreeClassifier(random_state=42), params, verbose=1, cv=3)
grid_search_cv.fit(X_train, y_train)


# In[512]:


grid_search_cv.best_estimator_


# In[514]:


tree_tun = DecisionTreeClassifier(ccp_alpha=0.0, class_weight=None, criterion='gini',
                       max_depth=8, max_features=None, max_leaf_nodes=30,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=3,
                       min_weight_fraction_leaf=0.0, presort='deprecated',
                       random_state=42, splitter='best')


# In[515]:


tree_tun.fit(X_train, y_train)


# In[516]:


tun_tree_pred = tree_tun.predict(X_test)


# In[517]:


print('precision: '+str(precision_score(y_test, tun_tree_pred)))
print('recall: '+str(recall_score(y_test, tun_tree_pred)))
print('accuray: '+str(accuracy_score(y_test, tun_tree_pred)))
print('f1 score: '+str(f1_score(y_test, tun_tree_pred, average='weighted')))


# In[518]:


# rysnek wytrenowanego drzewa
tree.plot_tree(tree_tun, 
                   feature_names=list(X_train.columns),  
                   class_names=['0','1'],
                   filled=True)


# In[519]:


confusion_matrix(y_test, tun_tree_pred)


# In[ ]:





# In[520]:


# random forest


# In[521]:


las_clas = RandomForestClassifier(random_state = 1)

n_estimators = [100, 300, 500, 800, 1200]
max_depth = [5, 8, 15, 25, 30]
min_samples_split = [2, 5, 10, 15, 100]
min_samples_leaf = [1, 2, 5, 10] 

hyperF = dict(n_estimators = n_estimators, max_depth = max_depth,  
              min_samples_split = min_samples_split, 
             min_samples_leaf = min_samples_leaf)

siatka_las = GridSearchCV(las_clas, hyperF, cv = 3, verbose = 1, 
                      n_jobs = -1)
siatka_las.fit(X_train, y_train)


# In[522]:


siatka_las.best_estimator_


# In[525]:


las_clas = RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight=None,
                       criterion='gini', max_depth=8, max_features='auto',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_leaf=1, min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=300,
                       n_jobs=None, oob_score=False, random_state=1, verbose=0,
                       warm_start=False)
                                  
                                  
las_clas.fit(X_train, y_train)
las_best_pred = las_clas.predict(X_test)


# In[526]:



print('precision: '+str(precision_score(y_test, las_best_pred)))
print('recall: '+str(recall_score(y_test, las_best_pred)))
print('accuray: '+str(accuracy_score(y_test, las_best_pred)))
print('f1 score: '+str(f1_score(y_test, las_best_pred, average='weighted')))


# In[527]:


confusion_matrix(y_test, las_best_pred)


# In[528]:


X_train.columns


# In[529]:


# sprawdzam jak las losowy ocenił "ważność" cech
nazwy_kol = list(X_train.columns)


# In[530]:


feature_importance_tab = pd.DataFrame(columns = ["Feature","importance"])
feature_importance_tab_tymcz = feature_importance_tab


# In[531]:


for i,v in enumerate(las_clas.feature_importances_):
    feature_importance_tab_tymcz.loc[0,'importance'] = v
    feature_importance_tab_tymcz.loc[0,'Feature'] = nazwy_kol[i]

    if i==0:
        feature_importance_tab =feature_importance_tab_tymcz
    else:
        feature_importance_tab = feature_importance_tab.append(feature_importance_tab_tymcz)
        


# In[532]:


feature_importance_tab.sort_values(by='importance', ascending=False)


# In[ ]:





# In[533]:


# SVM


# In[534]:


from sklearn.svm import SVC


# In[535]:


svm_class = SVC(gamma='auto',random_state=1, kernel = 'rbf')
svm_class.fit(X_train, y_train)
svm_pred = svm_class.predict(X_test)


# In[536]:



print('precision: '+str(precision_score(y_test, svm_pred)))
print('recall: '+str(recall_score(y_test, svm_pred)))
print('accuray: '+str(accuracy_score(y_test, svm_pred)))
print('f1 score: '+str(f1_score(y_test, svm_pred, average='weighted')))


# In[537]:


# xgboost


# In[538]:


import xgboost as xgb


# In[539]:


import re
regex = re.compile(r"\[|\]|<", re.IGNORECASE)


# In[540]:


# wśród nazw kolumn były znaki <>, zmiana wszelkich znaków które wyrzucają błąd w xgboost


# In[541]:


X_train.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in X_train.columns.values]


# In[542]:


X_test.columns = [regex.sub("_", col) if any(x in str(col) for x in set(('[', ']', '<'))) else col for col in X_test.columns.values]


# In[543]:


xgbmod = xgb.XGBClassifier(objective="binary:logistic", random_state=32,use_label_encoder=False)
xgbmod.fit(X_train, y_train)
xgb_pred = xgbmod.predict(X_test)


# In[544]:


print('precision: '+str(precision_score(y_test, xgb_pred)))
print('recall: '+str(recall_score(y_test, xgb_pred)))
print('accuray: '+str(accuracy_score(y_test, xgb_pred)))
print('f1 score: '+str(f1_score(y_test, xgb_pred, average='weighted')))


# In[ ]:





# In[545]:


# sieć neuronowa


# In[546]:


from sklearn.neural_network import MLPClassifier
mlp = MLPClassifier(max_iter=5000, random_state=1)


# In[547]:


parametry_mlp = {
    'hidden_layer_sizes': [(10,20,10),(50,50,50,50), (50,100,100,50),(50,50,50), (50,100,50), (100,)],
    'activation': ['identity', 'logistic', 'tanh', 'relu'],
    'solver': ['lbfgs','sgd', 'adam'],
    'alpha': [0.001, 0.05, 0.1, 0.0005],
    'learning_rate': ['constant','adaptive'],
}


# In[548]:


mlp_classifier_grid = GridSearchCV(mlp, parametry_mlp, n_jobs=-1, cv=3)
mlp_classifier_grid.fit(X_train, y_train)


# In[549]:


mlp_classifier_grid.best_estimator_


# In[550]:


mlp_best = MLPClassifier( activation='logistic', alpha=0.001, batch_size='auto', beta_1=0.9,
              beta_2=0.999, early_stopping=False, epsilon=1e-08,
              hidden_layer_sizes=(100,), learning_rate='constant',
              learning_rate_init=0.001, max_fun=15000, max_iter=5000,
              momentum=0.9, n_iter_no_change=10, nesterovs_momentum=True,
              power_t=0.5, random_state=1, shuffle=True, solver='adam',
              tol=0.0001, validation_fraction=0.1, verbose=False,
              warm_start=False)
mlp_best.fit(X_train, y_train)
mlp_predict = mlp_classifier_grid.predict(X_test)


# In[551]:


print('precision: '+str(precision_score(y_test, mlp_predict)))
print('recall: '+str(recall_score(y_test, mlp_predict)))
print('accuray: '+str(accuracy_score(y_test, mlp_predict)))
print('f1 score: '+str(f1_score(y_test, mlp_predict, average='weighted')))


# In[ ]:


# zmiany na zbiorze test do utworznia prognozy do kaggle


# In[552]:


test_fin = pd.read_csv('titanic_test.csv')


# In[553]:


test_fin['nazwisko'] = test_fin.apply(lambda x: x['Name'][:x['Name'].index(',')], axis=1)
test_fin['przedrostek'] = test_fin.apply(lambda x: x['Name'][x['Name'].index(',')+2:x['Name'].index('.')], axis=1)
test_fin['przedrostek'] = test_fin.apply(zamiana_przedostków, axis=1)
test_fin['zamężna kobieta'] = test_fin.apply( lambda x: 1 if x['przedrostek'] == 'Mrs' else 0, axis=1)

test_fin = test_fin.drop(columns=['nazwisko','Name'])

test_fin.loc[:,'Czy dziecko'] = test_fin.apply(lambda x: 1 if x['Age']<18 else 0 , axis=1)
test_fin['No_ppl_on_ticket'] = test_fin.groupby(['Ticket'])['Ticket'].transform('count')
test_fin['No_ppl_on_ticket'] = test_fin['No_ppl_on_ticket'].astype('object')
test_fin['No_ppl_on_ticket'] = test_fin.apply(liczba_osob_na_bil, axis=1)

test_fin.drop(columns=['PassengerId'], inplace = True)


test_fin.loc[:,'Parch'] = test_fin.loc[:,'Parch'].astype('object')
test_fin.loc[:,'Pclass'] = test_fin.loc[:,'Pclass'].astype('object')
test_fin.loc[:,'SibSp'] = test_fin.loc[:,'SibSp'].astype('object')

test_fin['Cabin'].fillna('OTHER', inplace=True)
test_fin['Embarked'].fillna('OTHER', inplace = True)

test_fin['Cabin'] = test_fin['Cabin'].str[:1]
test_fin['Cabin'] = test_fin.apply(group_cab, axis=1)
test_fin['Age'] = test_fin.groupby(['Pclass','Cabin'])['Age'].transform(lambda x: x.fillna(x.mean()))
test_fin['Age'] = test_fin.apply(age_binning, axis=1)


# In[554]:


test_fin['Fare_log'] = np.log(test_fin['Fare']+0.1)

def Fare_log_winsor(x):
    if x['Fare_log'] < np.percentile(test_fin['Fare_log'],5):
        return np.percentile(test_fin['Fare_log'],5)
    elif x['Fare_log'] > np.percentile(test_fin['Fare_log'],95):
        return np.percentile(test_fin['Fare_log'],95)
    else:
        return x['Fare_log']

test_fin['Fare_log_winsor'] = test_fin.apply(Fare_log_winsor, axis=1)
test_fin['no_fam_members'] = test_fin['SibSp']+test_fin['Parch']+1

test_fin['no_fam_members'] = test_fin['no_fam_members'].astype('object')

test_fin['no_fam_members'] = test_fin.apply(wielkosc_rodziny, axis=1)

test_fin = test_fin.drop(columns = ['Ticket'])
test_fin = test_fin.drop(columns=['Fare','Fare_log'])


# In[555]:


test_fin.columns


# In[556]:


X_train.columns


# In[557]:


test_fin[['Fare_log_winsor']] = StandardScaler().fit_transform(test_fin[['Fare_log_winsor']])
test_fin = pd.concat([test_fin, pd.get_dummies(test_fin[list(test_fin.select_dtypes('object').columns)])], axis=1) 
test_fin = test_fin.drop(columns= list(test_fin.select_dtypes('object').columns))


# In[558]:


X_train, test_fin = X_train.align(test_fin, join='left', axis=1)
test_fin.fillna(0, inplace=True)


# In[559]:


test_fin.columns


# In[560]:


X_train.columns


# In[561]:


las_best_pred_fin = las_clas.predict(test_fin)


# In[562]:


las_best_pred_fin


# In[563]:


test_fin = pd.read_csv('titanic_test.csv')
test_fin_pred = test_fin[['PassengerId']]
test_fin_pred['Survived'] = las_best_pred_fin
test_fin_pred.to_csv('final_pred.csv')


# In[ ]:





# In[ ]:




