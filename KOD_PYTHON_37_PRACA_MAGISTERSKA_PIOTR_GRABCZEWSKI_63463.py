#!/usr/bin/env python
# coding: utf-8

# In[600]:


import pandas as pd, numpy as np


# In[ ]:





# In[601]:


from os import listdir
from os.path import isfile, join
mypath = 'C:\\Users\\Feflo\\Documents\\dane'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]


# In[602]:


csv = [k for k in onlyfiles if 'csv' in k]


# In[603]:


for i in csv:
    print(i)
    globals()[i[0:i.index('.')]] = pd.read_csv(mypath+'\\'+i, encoding = 'latin', sep = ';')


# In[604]:


# liczba pacjentów u których wystąpił udar
len(pacjenci[pacjenci['CZY_UDAR']==True])


# In[ ]:





# In[605]:


#ile jest pacjentów wg tabeli pacjenci
len(pacjenci)


# In[606]:


# łączenie
# kodowanie zmiennych
# podział test train


# oversampling i undersampling


# In[607]:


# zakodowanie binarne zmiennej objaśnianej
def bin(x):
    if x['CZY_UDAR']== False:
        return 0
    else:
        return 1
pacjenci['CZY_UDAR'] = pacjenci.apply(bin, axis=1)


# In[608]:


# łączenie tabeli pacjenci z tabelą parametry_pacjentów
pacj_param = pd.merge(parametry_pacjentow, pacjenci, how='inner', on='ID_PACJENTA')


# In[609]:


# wybieram pacjentów z województwa małopolskiego
pacj_param = pacj_param[(pacj_param['TERYT_POWIATU']>=1200) &(pacj_param['TERYT_POWIATU']<1300) ]


# In[610]:


# rozdziela zmienne objaśniane od objaśniającej
y = pacj_param['CZY_UDAR']
X = pacj_param.drop(columns=['CZY_UDAR'])


# In[611]:


#X[X['ID_PACJENTA']==455537].to_csv('pacjet_455537.csv')


# In[612]:


# przed połączeniem z innymi tabelami dzieli na zbiór testowy i treningowy aby móc wykonać undersampling na treningowym
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    stratify=y, 
                                                    test_size=0.30,random_state=105)


# In[613]:


# zmienia, nazwę
X = X_train
y = y_train

##### liczy stosunek ludzi z udarem do całej tabeli
len(y_train[y_train==1])/len(X_train)
# In[ ]:





# In[615]:


#algorytm dokonuje undersamplingu w taki sposób aby porporcja pacjentów z udarem była 1/2 pacjentów bez udaru
from imblearn.under_sampling import RandomUnderSampler
under = RandomUnderSampler(sampling_strategy=0.8, random_state=15)
X, y = under.fit_resample(X, y)


# In[616]:


X['CZY_UDAR']=y


# In[617]:


pacj_param = X


# In[618]:


len(X)


# In[83]:


#pokazuje proporcję osób z udarem do osób bez udaru
len(pacj_param[pacj_param['CZY_UDAR']==1])/len(pacj_param[pacj_param['CZY_UDAR']==0])


# In[84]:


# łączy połączone wcześniej tabele z tabeląrecepty wg klucza ID_PACJENTA
pacj_parampacj_recepty = pd.merge( pacj_param, recepty, how='inner', on='ID_PACJENTA')


# In[85]:


# łączy tabele powstałą we wcześniejszym wierszu z tabelą świadczenia wg klucza ID_PACJENTA
pacj_parampacj_recepty_swiad = pd.merge(swiadczenia, pacj_parampacj_recepty, how='inner', on='ID_PACJENTA')


# In[86]:


# to co wyżej, łączy z tabelą procedury po ID_EPIZODDU
swiad_pacj_parampacj_recepty_procedury = pd.merge( pacj_parampacj_recepty_swiad, procedury, how='inner', on='ID_EPIZODU')


# In[87]:


# to co wyżej tylko łączy z rozpoznaniami
swiad_pacj_parampacj_recepty_procedury_rozpoz = pd.merge( swiad_pacj_parampacj_recepty_procedury, rozpoznania, how='inner', on='ID_KONTAKTU')


# In[88]:


# długą nazwę zamienia na df od dataframe
df = swiad_pacj_parampacj_recepty_procedury_rozpoz


# In[89]:


df = df.drop_duplicates()


# In[90]:


len(df[df['CZY_UDAR']==1])/ylen(df[df['CZY_UDAR']==0])


# In[91]:


# oddziela zmienne
y = df['CZY_UDAR']
X = df.drop(columns=['CZY_UDAR'])


# In[93]:


#len(y[y['CZY_UDAR']==1])/len(y[y['CZY_UDAR']==0])


# In[100]:


len(X)


# In[ ]:


# ZBIOR  TESTOWY


# In[94]:


# przed połączeniem z innymi tabelami dzieli na zbiór testowy i treningowy aby móc wykonać undersampling na treningowym
from sklearn.model_selection import train_test_split
X_test_test, X_t, y_test_test, y_t = train_test_split(X_test, y_test,
                                                    stratify=y_test, 
                                                    test_size=0.02,random_state=105)


# In[95]:


X_t['CZY_UDAR'] = y_t


# In[96]:


# łączy na zbiorze testowym
X_test_recepty = pd.merge( X_t, recepty, how='inner', on='ID_PACJENTA')


# In[97]:


X_test_recepty_swiad = pd.merge(swiadczenia, X_test_recepty, how='inner', on='ID_PACJENTA')
X_test_recepty_swiad_procedury = pd.merge( X_test_recepty_swiad, procedury, how='inner', on='ID_EPIZODU')
X_test_recepty_swiad_procedury_rozpoz = pd.merge( X_test_recepty_swiad_procedury, rozpoznania, how='inner', on='ID_KONTAKTU')


# In[98]:


y_t = X_test_recepty_swiad_procedury_rozpoz['CZY_UDAR']
X_t = X_test_recepty_swiad_procedury_rozpoz.drop(columns='CZY_UDAR')


# In[99]:


len(X_t)


# In[101]:


X_t.to_csv('X_testowy_v3.csv')


# In[102]:


y_t.to_csv('y_testowy_v3.csv')


# In[103]:


X.to_csv('X_treningowy_v3.csv')


# In[104]:


y.to_csv('y_treningowy_v3.csv')


# In[105]:


len(y_t[y_t==1])
#/len(y_t[y_t==0])


# In[580]:


y = pd.read_csv('y_treningowy_v3.csv')
y_t = pd.read_csv('y_testowy_v3.csv')
X = pd.read_csv('X_treningowy_v3.csv')
X_t = pd.read_csv('X_testowy_v3.csv')


# In[582]:


y_t = y_t.drop(columns='Unnamed: 0')
y = y.drop(columns='Unnamed: 0')
X = X.drop(columns='Unnamed: 0')
X_t = X_t.drop(columns='Unnamed: 0')


# In[599]:


len(X_t)
#[y_t['CZY_UDAR']==1])


# In[598]:


len(X_t)/len(y)


# In[588]:


len(y[y['CZY_UDAR']==1]) / len(y)#[y['CZY_UDAR']==0])


# In[24]:


pom_tab = pd.concat([X['ID_KONTAKTU'], X['ID_PACJENTA']], axis=1)
pom_tab = pom_tab.drop_duplicates()
pom_tab['count'] = pom_tab.groupby(['ID_PACJENTA'])['ID_KONTAKTU'].transform('count')
pom_tab['count_opposit'] = pom_tab.groupby(['ID_KONTAKTU'])['ID_PACJENTA'].transform('count')
pom_tab[pom_tab['count']>1]
pom_tab


# In[26]:


def tabela_pomocnicza(kol, klucz):
    pom_tab = pd.concat([X[klucz], X[kol]], axis=1)
    pom_tab = pom_tab.drop_duplicates()
    pom_tab['count'] = pom_tab.groupby([klucz])[kol].transform('count')
    pom_tab['count_opposit'] = pom_tab.groupby([kol])[klucz].transform('count')
    pom_tab[pom_tab['count']>1]
    return pom_tab


# In[110]:


pom_tab = tabela_pomocnicza('ID_KONTAKTU','ID_PACJENTA')


# In[111]:


pom_tab


# In[112]:


#zmienna ilość hospitalizacji na pacjenta
pom_tab['enumrate_pacjent_id_kontaktu'] = (pom_tab.groupby('ID_PACJENTA')['ID_KONTAKTU']
                                .transform(lambda x: pd.CategoricalIndex(x).codes))
pom_tab['enumrate_pacjent_id_kontaktu'] = pom_tab['enumrate_pacjent_id_kontaktu']+1


# In[113]:


pom_tab[pom_tab['count']==pom_tab['count'].max()]#.to_csv('827660.csv')


# In[114]:


pom_tab['enumrate_pacjent_id_kontaktu']=pom_tab.enumrate_pacjent_id_kontaktu.astype('object') 


# In[115]:


#podziaał na grupy po 20
pom_tab['enumrate_pacjent_id_kontaktu2'] = np.ceil(pom_tab.enumrate_pacjent_id_kontaktu/20.1)*20
pom_tab['enumrate_pacjent_id_kontaktu3'] = np.floor(pom_tab.enumrate_pacjent_id_kontaktu/20.1)*20
pom_tab['enumrate_pacjent_id_kontaktu4'] = '('+pom_tab.enumrate_pacjent_id_kontaktu3.astype('str') + ',' +pom_tab.enumrate_pacjent_id_kontaktu2.astype('str') +'>'


# In[116]:


pom_tab


# In[117]:


# powoduje, że jeżeli wartosc jest większa od 100 to wstawia 'Ponad 100'
def zmienna_enumerate(x):
    if int(x['enumrate_pacjent_id_kontaktu3']) <100:
        return x['enumrate_pacjent_id_kontaktu4']
    else: return 'Ponad 100'


# In[118]:


pom_tab['enumrate_pacjent_id_kontaktu5'] = pom_tab.apply(zmienna_enumerate,axis=1)


# In[30]:


import pandas as pd
from plotnine import *

get_ipython().run_line_magic('matplotlib', 'inline')


# In[120]:


pom_tab = pom_tab.drop(['enumrate_pacjent_id_kontaktu4','enumrate_pacjent_id_kontaktu3','enumrate_pacjent_id_kontaktu2','enumrate_pacjent_id_kontaktu'], axis=1)


# In[121]:


pom_tab.rename(columns={"enumrate_pacjent_id_kontaktu5": "Enumerate_pacjent_id_kontaktu"}, inplace = True)


# In[122]:


pom_tab


# In[27]:


#funkcja wykonuje wykres częstosci wybranej zmiennej
def wykres_czestosci(nazwa_zmien):
    return (ggplot(pom_tab, aes(nazwa_zmien, fill= nazwa_zmien, color= nazwa_zmien))
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
    


# In[124]:


wykres_czestosci('Enumerate_pacjent_id_kontaktu')


# In[ ]:





# In[125]:


#zmienna ilość hospitalizacji na pacjenta
X['enumrate_pacjent_id_kontaktu'] = (X.groupby('ID_PACJENTA')['ID_KONTAKTU']
                                .transform(lambda x: pd.CategoricalIndex(x).codes))
X['enumrate_pacjent_id_kontaktu'] = X['enumrate_pacjent_id_kontaktu']+1

X['enumrate_pacjent_id_kontaktu']=X.enumrate_pacjent_id_kontaktu.astype('object') 

#podziaał na grupy po 20
X['enumrate_pacjent_id_kontaktu2'] = np.ceil(X.enumrate_pacjent_id_kontaktu/20.1)*20
X['enumrate_pacjent_id_kontaktu3'] = np.floor(X.enumrate_pacjent_id_kontaktu/20.1)*20
X['enumrate_pacjent_id_kontaktu4'] = '('+X.enumrate_pacjent_id_kontaktu3.astype('str') + ',' +X.enumrate_pacjent_id_kontaktu2.astype('str') +'>'

X['enumrate_pacjent_id_kontaktu5'] = X.apply(zmienna_enumerate,axis=1)

X = X.drop(['enumrate_pacjent_id_kontaktu4','enumrate_pacjent_id_kontaktu3','enumrate_pacjent_id_kontaktu2','enumrate_pacjent_id_kontaktu'], axis=1)

X.rename(columns={"enumrate_pacjent_id_kontaktu5": "Enumerate_pacjent_id_kontaktu"}, inplace = True)


# In[ ]:





# In[126]:


#zmienna ilość hospitalizacji na pacjenta
X_t['enumrate_pacjent_id_kontaktu'] = (X_t.groupby('ID_PACJENTA')['ID_KONTAKTU']
                                .transform(lambda x: pd.CategoricalIndex(x).codes))
X_t['enumrate_pacjent_id_kontaktu'] = X_t['enumrate_pacjent_id_kontaktu']+1

X_t['enumrate_pacjent_id_kontaktu']=X_t.enumrate_pacjent_id_kontaktu.astype('object') 

#podziaał na grupy po 20
X_t['enumrate_pacjent_id_kontaktu2'] = np.ceil(X_t.enumrate_pacjent_id_kontaktu/20.1)*20
X_t['enumrate_pacjent_id_kontaktu3'] = np.floor(X_t.enumrate_pacjent_id_kontaktu/20.1)*20
X_t['enumrate_pacjent_id_kontaktu4'] = '('+X_t.enumrate_pacjent_id_kontaktu3.astype('str') + ',' +X_t.enumrate_pacjent_id_kontaktu2.astype('str') +'>'

X_t['enumrate_pacjent_id_kontaktu5'] = X_t.apply(zmienna_enumerate,axis=1)

X_t = X_t.drop(['enumrate_pacjent_id_kontaktu4','enumrate_pacjent_id_kontaktu3','enumrate_pacjent_id_kontaktu2','enumrate_pacjent_id_kontaktu'], axis=1)

X_t.rename(columns={"enumrate_pacjent_id_kontaktu5": "Enumerate_pacjent_id_kontaktu"}, inplace = True)


# In[127]:


# informacje o zmiennych objaśniających
X.info()


# In[128]:


print(len(X.columns))
print(len(X_t.columns))


# In[ ]:





# In[129]:


pom_tab = tabela_pomocnicza('ROK_SWIADCZENIA','ID_KONTAKTU')
pom_tab


# In[130]:


pom_tab['ROK_SWIADCZENIA']=pom_tab.ROK_SWIADCZENIA.astype('object') 


# In[131]:


wykres_czestosci('ROK_SWIADCZENIA')


# In[132]:


# zmiana oryginalnej zmiennej w całym zbiorze danych
X['ROK_SWIADCZENIA']=X.ROK_SWIADCZENIA.astype('object') 


# In[133]:


X_t['ROK_SWIADCZENIA']=X_t.ROK_SWIADCZENIA.astype('object') 


# In[134]:


import math
import matplotlib.pyplot as plt 


# In[135]:


pom_tab = tabela_pomocnicza('TYDZIEN_POCZATKU_KONTAKTU','ID_KONTAKTU')


# In[136]:


def histogram(zmienna):
    max_=pom_tab['count_opposit'].max()
    mean_=pom_tab[zmienna].mean()
    nbins=53
    plt.hist(pom_tab[zmienna],nbins, facecolor='#79edde', alpha=0.75,ec="k")
    plt.xlabel(zmienna)
    plt.ylabel('Czestość')
    plt.title('Histogram zmiennej '+zmienna)
    #plt.text(round(mean_,2)+round(mean_,2)*0.02,max_-max_*0.05, 'średia = \n'+ str(round(mean_,2)), color='r')
    plt.axis([0, 54, 0, max_+max_*0.1])
    plt.grid(True)
    #plt.plot( [mean_,mean_],[0,max_+max_*0.1], 'k-', lw=2, ls = '--', color='r')
    plt.savefig(zmienna+'_histogram.png', dpi=250, optimize=False,pad_inches=0.1, bbox_inches = "tight")
    plt.show()
    return plt


# In[137]:


histogram('TYDZIEN_POCZATKU_KONTAKTU')


# In[138]:


pom_tab = tabela_pomocnicza('TYDZIEN_KONCA_KONTAKTU','ID_KONTAKTU')


# In[139]:


histogram('TYDZIEN_KONCA_KONTAKTU')


# In[ ]:





# In[140]:


pom_tab = pd.concat([X['TYDZIEN_POCZATKU_KONTAKTU'],X['TYDZIEN_KONCA_KONTAKTU'], X['ID_KONTAKTU']], axis=1)
pom_tab = pom_tab.drop_duplicates()


# In[141]:


def dlugosc_kontaktu(x):
    if x['TYDZIEN_POCZATKU_KONTAKTU']<= x['TYDZIEN_KONCA_KONTAKTU']:
        return x['TYDZIEN_KONCA_KONTAKTU'] - x['TYDZIEN_POCZATKU_KONTAKTU']
    else:
        return (53-x['TYDZIEN_POCZATKU_KONTAKTU'])+x['TYDZIEN_KONCA_KONTAKTU']

pom_tab['DLUGOSC_KONTAKTU'] = pom_tab.apply(dlugosc_kontaktu, axis=1)


# In[142]:


pom_tab['count'] = pom_tab.groupby(['ID_KONTAKTU'])['DLUGOSC_KONTAKTU'].transform('count')
pom_tab['count_opposit'] = pom_tab.groupby(['DLUGOSC_KONTAKTU'])['ID_KONTAKTU'].transform('count')
pom_tab[pom_tab['count']>1]


# In[143]:


max_=pom_tab['count_opposit'].max()
mean_=pom_tab['DLUGOSC_KONTAKTU'].mean()


# In[ ]:





# In[144]:


max_dl=pom_tab['DLUGOSC_KONTAKTU'].max()


# In[145]:


np.unique(pom_tab['DLUGOSC_KONTAKTU'])


# In[146]:


pom_tab[pom_tab['DLUGOSC_KONTAKTU']==53]


# In[147]:


pom_tab['DLUGOSC_KONTAKTU']=pom_tab.DLUGOSC_KONTAKTU.astype('object') 


# In[ ]:





# In[ ]:





# In[148]:


X.info()


# In[149]:


# ilość unikatowych wartości zmiennej KOD_ZAKRESU
len(np.unique(X.KOD_ZAKRESU) )


# In[150]:


pom_tab = tabela_pomocnicza('KOD_ZAKRESU','ID_KONTAKTU')


# In[151]:


# zmiana typu zmiennej na string
pom_tab['KOD_ZAKRESU'] = pom_tab['KOD_ZAKRESU'].astype('str') 


# In[152]:


# 15 najliczniejszych kategori do pogrupowania
ls_kod_zakr = list(pom_tab[['KOD_ZAKRESU','count_opposit']].drop_duplicates().nlargest(15, 'count_opposit')['KOD_ZAKRESU'])


# In[153]:


#grupowanie
def kod_zakresu(x):
    if x['KOD_ZAKRESU'] in ls_kod_zakr:
        return x['KOD_ZAKRESU']
    else: return 'OTHER'


# In[154]:


pom_tab['KOD_ZAKRESU_TOP_15_KAT']= pom_tab.apply(kod_zakresu, axis=1)


# In[155]:


#zmiana typu zmiennej na object
pom_tab['KOD_ZAKRESU'] = pom_tab['KOD_ZAKRESU_TOP_15_KAT'].astype('object') 


# In[156]:


#zmieniony wykres, mniejsza czcionka, opisy kategorii obrócone
(ggplot(pom_tab, aes('KOD_ZAKRESU',fill='KOD_ZAKRESU', color = 'KOD_ZAKRESU'))
 + geom_bar()
 + geom_text(
     aes(label='stat(prop)*100', group=1, size=8),
     stat='count',
     nudge_y=0.125,
     va='bottom',
     format_string='{:.1f}%'
 )
 +theme(axis_text_x=element_text(rotation=90, hjust=1))
 +ggtitle('Wykres częstości zmiennej KOD_ZAKRESU')
)


# In[157]:


X['KOD_ZAKRESU'] = X['KOD_ZAKRESU'].astype('str') 

X['KOD_ZAKRESU_TOP_15_KAT']= X.apply(kod_zakresu, axis=1)

X['KOD_ZAKRESU'] = X['KOD_ZAKRESU_TOP_15_KAT'].astype('object') 

X = X.drop(['KOD_ZAKRESU_TOP_15_KAT'], axis=1)


# In[158]:


np.unique(X.KOD_ZAKRESU)


# In[159]:


# na testowym to samo co treningowym
X_t['KOD_ZAKRESU'] = X_t['KOD_ZAKRESU'].astype('str') 

X_t['KOD_ZAKRESU_TOP_15_KAT']= X_t.apply(kod_zakresu, axis=1)

X_t['KOD_ZAKRESU'] = X_t['KOD_ZAKRESU_TOP_15_KAT'].astype('object') 

X_t = X_t.drop(['KOD_ZAKRESU_TOP_15_KAT'], axis=1)


# In[160]:


np.unique(X_t.KOD_ZAKRESU) # do testowego dodać zmienna dummy 111026019 z samymi zerami


# In[161]:


# ilość poziomów zmiennej 
len(np.unique(X.KOD_PRODUKTU_JEDNOSTKOWEGO )) # 476 unikalnych wartości


# In[162]:


pom_tab = tabela_pomocnicza('KOD_PRODUKTU_JEDNOSTKOWEGO','ID_KONTAKTU')


# In[163]:


pom_tab['KOD_PRODUKTU_JEDNOSTKOWEGO'] = pom_tab['KOD_PRODUKTU_JEDNOSTKOWEGO'].astype('str') 


# In[164]:


len(np.unique(pom_tab['KOD_PRODUKTU_JEDNOSTKOWEGO']))


# In[165]:


# pierwsza część kodu jednostkowego
pom_tab['KOD_PRODUKTU_JEDNOSTKOWEGO_GLOWNE'] = pom_tab['KOD_PRODUKTU_JEDNOSTKOWEGO'].str[:7]


# In[166]:


pom_tab['count'] = pom_tab.groupby(['ID_KONTAKTU'])['KOD_PRODUKTU_JEDNOSTKOWEGO_GLOWNE'].transform('count')
pom_tab['count_opposit'] = pom_tab.groupby(['KOD_PRODUKTU_JEDNOSTKOWEGO_GLOWNE'])['ID_KONTAKTU'].transform('count')
pom_tab[pom_tab['count']>1]


# In[167]:


len(np.unique(pom_tab['KOD_PRODUKTU_JEDNOSTKOWEGO_GLOWNE'] ))


# In[168]:


ls_kod_prod_jedn = list(pom_tab[['KOD_PRODUKTU_JEDNOSTKOWEGO_GLOWNE','count_opposit']].drop_duplicates().nlargest(15, 'count_opposit')['KOD_PRODUKTU_JEDNOSTKOWEGO_GLOWNE'])


# In[169]:


ls_kod_prod_jedn


# In[170]:


def kod_prod_jedn(x):
    if x['KOD_PRODUKTU_JEDNOSTKOWEGO_GLOWNE'] in ls_kod_prod_jedn:
        return x['KOD_PRODUKTU_JEDNOSTKOWEGO_GLOWNE']
    else: return 'OTHER'


# In[171]:


pom_tab['KOD_PRODUKTU_JEDNOSTKOWEGO_GLOWNE_15_top']= pom_tab.apply(kod_prod_jedn, axis=1)


# In[172]:


pom_tab['KOD_PRODUKTU_JEDNOSTKOWEGO'] = pom_tab['KOD_PRODUKTU_JEDNOSTKOWEGO_GLOWNE_15_top'].astype('object') 


# In[173]:


(ggplot(pom_tab, aes('KOD_PRODUKTU_JEDNOSTKOWEGO',fill='KOD_PRODUKTU_JEDNOSTKOWEGO', color = 'KOD_PRODUKTU_JEDNOSTKOWEGO'))
 + geom_bar()
 + geom_text(
     aes(label='stat(prop)*100', group=1, size=8),
     stat='count',
     nudge_y=0.125,
     va='bottom',
     format_string='{:.1f}%'
 )
 +theme(axis_text_x=element_text(rotation=90, hjust=1))
 +ggtitle('Wykres częstości zmiennej\n KOD_PRODUKTU_JEDNOSTKOWEGO')
)


# In[174]:


X['KOD_PRODUKTU_JEDNOSTKOWEGO_GLOWNE'] = X['KOD_PRODUKTU_JEDNOSTKOWEGO'].str[:7]

X['KOD_PRODUKTU_JEDNOSTKOWEGO_GLOWNE_15_top']= X.apply(kod_prod_jedn, axis=1)

X['KOD_PRODUKTU_JEDNOSTKOWEGO'] = X['KOD_PRODUKTU_JEDNOSTKOWEGO_GLOWNE_15_top'].astype('object') 

X = X.drop(['KOD_PRODUKTU_JEDNOSTKOWEGO_GLOWNE', 'KOD_PRODUKTU_JEDNOSTKOWEGO_GLOWNE_15_top'], axis=1)


# In[175]:


np.unique(X['KOD_PRODUKTU_JEDNOSTKOWEGO'])


# In[176]:


# to samo na testowych

X_t['KOD_PRODUKTU_JEDNOSTKOWEGO_GLOWNE'] = X_t['KOD_PRODUKTU_JEDNOSTKOWEGO'].str[:7]

X_t['KOD_PRODUKTU_JEDNOSTKOWEGO_GLOWNE_15_top']= X_t.apply(kod_prod_jedn, axis=1)

X_t['KOD_PRODUKTU_JEDNOSTKOWEGO'] = X_t['KOD_PRODUKTU_JEDNOSTKOWEGO_GLOWNE_15_top'].astype('object') 

X_t = X_t.drop(['KOD_PRODUKTU_JEDNOSTKOWEGO_GLOWNE', 'KOD_PRODUKTU_JEDNOSTKOWEGO_GLOWNE_15_top'], axis=1)


# In[177]:


np.unique(X_t['KOD_PRODUKTU_JEDNOSTKOWEGO'])


# In[ ]:





# In[178]:


# wykres częstości zmienne typ_komorki


# In[179]:


X.TYP_KOMORKI=X.TYP_KOMORKI.astype('object') 


# In[180]:


X_t.TYP_KOMORKI=X_t.TYP_KOMORKI.astype('object') 


# In[181]:


pom_tab = tabela_pomocnicza('TYP_KOMORKI','ID_KONTAKTU')
pom_tab


# In[182]:


ls_typ_kom = list(pom_tab[['TYP_KOMORKI','count_opposit']].drop_duplicates().nlargest(10, 'count_opposit')['TYP_KOMORKI'])


# In[183]:


pom_tab['TYP_KOMORKI'] = pom_tab['TYP_KOMORKI'].astype('str') 


# In[184]:


ls_typ_kom


# In[185]:


def typ_komorki(x):
    if int(x['TYP_KOMORKI']) in ls_typ_kom:
        return x['TYP_KOMORKI']
    else: return 'OTHER'
pom_tab['TYP_KOMORKI_TOP10_KAT']= pom_tab.apply(typ_komorki, axis=1)


# In[ ]:





# In[186]:


pom_tab['TYP_KOMORKI'] = pom_tab['TYP_KOMORKI_TOP10_KAT'].astype('object') 


# In[187]:


(ggplot(pom_tab, aes('TYP_KOMORKI',fill='TYP_KOMORKI', color = 'TYP_KOMORKI'))
 + geom_bar()
 + geom_text(
     aes(label='stat(prop)*100', group=1, size=8),
     stat='count',
     nudge_y=0.125,
     va='bottom',
     format_string='{:.1f}%'
 )
 +theme(axis_text_x=element_text(rotation=90, hjust=1))
 +ggtitle('Wykres częstości zmiennej TYP_KOMORKI')
)


# In[188]:


X['TYP_KOMORKI'] = X['TYP_KOMORKI'].astype('str') 

X['TYP_KOMORKI_TOP10_KAT']= X.apply(typ_komorki, axis=1)

X['TYP_KOMORKI'] = X['TYP_KOMORKI_TOP10_KAT'].astype('object') 

X = X.drop(['TYP_KOMORKI_TOP10_KAT'], axis=1)


# In[189]:


# na testowym
X_t['TYP_KOMORKI'] = X_t['TYP_KOMORKI'].astype('str') 

X_t['TYP_KOMORKI_TOP10_KAT']= X_t.apply(typ_komorki, axis=1)

X_t['TYP_KOMORKI'] = X_t['TYP_KOMORKI_TOP10_KAT'].astype('object') 

X_t = X_t.drop(['TYP_KOMORKI_TOP10_KAT'], axis=1)


# In[190]:


np.unique(X_t['TYP_KOMORKI'])


# In[191]:


np.unique(X['TYP_KOMORKI'])


# In[192]:


# słownik do numerowania rodzajów świadczeń
d = dict(zip(list(np.unique(X.RODZAJ_SWIADCZEN )), list(range(1,12))))


# In[193]:


d # poprawic , że 11 poziomów w tekście


# In[194]:


# zmiana formy opisowej zmiennej na numeryczną

X['RODZAJ_SWIADCZEN'] = X.apply( lambda x: d[x['RODZAJ_SWIADCZEN']], axis=1)


# In[195]:


X.RODZAJ_SWIADCZEN=X.RODZAJ_SWIADCZEN.astype('object')


# In[196]:


# na testowym
d = dict(zip(list(np.unique(X_t.RODZAJ_SWIADCZEN )), list(range(1,11))))
X_t['RODZAJ_SWIADCZEN'] = X_t.apply( lambda x: d[x['RODZAJ_SWIADCZEN']], axis=1)

X_t.RODZAJ_SWIADCZEN=X_t.RODZAJ_SWIADCZEN.astype('object')


# In[197]:


np.unique(X['RODZAJ_SWIADCZEN'])


# In[198]:


np.unique(X_t['RODZAJ_SWIADCZEN']) 


# In[ ]:





# In[199]:


pom_tab = tabela_pomocnicza('RODZAJ_SWIADCZEN','ID_KONTAKTU')
pom_tab


# In[200]:


pom_tab = pd.concat([X['RODZAJ_SWIADCZEN'], X['ID_KONTAKTU']], axis=1)
pom_tab = pom_tab.drop_duplicates()
pom_tab['count'] = pom_tab.groupby(['ID_KONTAKTU'])['RODZAJ_SWIADCZEN'].transform('count')
pom_tab['count_opposit'] = pom_tab.groupby(['RODZAJ_SWIADCZEN'])['ID_KONTAKTU'].transform('count')
pom_tab[pom_tab['count']>1]
pom_tab


# In[201]:


(ggplot(pom_tab, aes('RODZAJ_SWIADCZEN',fill='RODZAJ_SWIADCZEN', color = 'RODZAJ_SWIADCZEN'))
 + geom_bar()
 + geom_text(
     aes(label='stat(prop)*100', group=1),
     stat='count',
     nudge_y=0.125,
     #va='bottom',
     format_string='{:.1f}%'
 )
 +theme(axis_text_x=element_blank())
 +ggtitle('Wykres częstości zmiennej RODZAJ_SWIADCZEN')
)


# In[40]:


def przecinek(x):
    if (type(x) == str) :
        try:
            x = x.replace(',','.')
        except:
            return float(x)
    return float(x)


# In[203]:


X.columns


# In[204]:


X['KWOTA_ROZLICZONA'] = X['KWOTA_ROZLICZONA'].apply(przecinek)


# In[205]:


# na testowym
X_t['KWOTA_ROZLICZONA'] = X_t['KWOTA_ROZLICZONA'].apply(przecinek)


# In[206]:


X.KWOTA_ROZLICZONA.mean()


# In[207]:


X.KWOTA_ROZLICZONA


# In[208]:


# ZMIENNA KWOTA_ROZLICZONA


# In[209]:


pom_tab = tabela_pomocnicza('KWOTA_ROZLICZONA','ID_KONTAKTU')
pom_tab


# In[210]:


pom_tab['KWOTA_ROZLICZONA_LOG']= pom_tab['KWOTA_ROZLICZONA'].apply(lambda x: np.log((x+ 1) - min(pom_tab['KWOTA_ROZLICZONA'])))


# In[211]:


# pokazanie, że tylko 390 najwyższych wartości zmienej jest powyżej 500
pom_tab.nlargest(390, 'KWOTA_ROZLICZONA')


# In[212]:


pom_tab.nlargest(10, 'count_opposit')


# In[213]:


pom_tab['count_opposit'] = pom_tab.groupby(['KWOTA_ROZLICZONA'])['KWOTA_ROZLICZONA'].transform('count')


# In[214]:


# obliczenie, że 390 wartości to tylko 3.5% wszystkich
390/len(pom_tab)


# In[215]:


# minimalna wartość ze zmiennej
pom_tab['KWOTA_ROZLICZONA'].min()


# In[216]:


# the histogram of the data
sr = pom_tab['KWOTA_ROZLICZONA'].mean()
nbins=2000
plt.hist(pom_tab['KWOTA_ROZLICZONA'],nbins, facecolor='#79edde', alpha=0.75,ec="k")
plt.xlabel('KWOTA_ROZLICZONA')
plt.ylabel('Czestotliwość')
plt.title('Histogram zmiennej KWOTA_ROZLICZONA')
#plt.text(round(sr,2)+50, 9000-2500, 'średia = \n'+ str(round(sr,2)), color='r')
plt.axis([0, 500, 0, 7000])
plt.grid(True)
#plt.plot([sr,sr], [0,20000], 'k-', lw=2, ls = '--', color='r')

plt.savefig('KWOTA_ROZLICZONA_histogram.png', dpi=250, optimize=False,pad_inches=0.1, bbox_inches = "tight")
plt.show()


# In[217]:


# the histogram of the data
sr = pom_tab['KWOTA_ROZLICZONA_LOG'].mean()
nbins=30
plt.hist(pom_tab['KWOTA_ROZLICZONA_LOG'],nbins, facecolor='#79edde', alpha=0.75,ec="k")
plt.xlabel('KWOTA_ROZLICZONA_LOG')
plt.ylabel('Czestotliwość')
plt.title('Histogram zmiennej KWOTA_ROZLICZONA_LOG')
#plt.text(round(sr,2)+50, 9000-2500, 'średia = \n'+ str(round(sr,2)), color='r')
plt.axis([0, 12, 0, 5000])
plt.grid(True)
#plt.plot([sr,sr], [0,20000], 'k-', lw=2, ls = '--', color='r')

plt.savefig('KWOTA_ROZLICZONA_LOG_histogram.png', dpi=250, optimize=False,pad_inches=0.1, bbox_inches = "tight")
plt.show()


# In[218]:


len(pom_tab['KWOTA_ROZLICZONA'])


# In[219]:


# jakby co można się pokusić o box cox transformacje dodając do zmiennej jakąś wartosć


# In[220]:


# wykres pudełkowy
plt.boxplot(pom_tab['KWOTA_ROZLICZONA'],labels=['KWOTA_ROZLICZONA'],whis=3)
plt.title('Wykres pudełkowy zmiennej KWOTA_ROZLICZONA')
plt.show()


# In[221]:


# wykres pudełkowy
plt.boxplot(pom_tab['KWOTA_ROZLICZONA_LOG'],labels=['KWOTA_ROZLICZONA_LOG'],whis=3)
plt.title('Wykres pudełkowy zmiennej KWOTA_ROZLICZONA_LOG')
plt.show()


# In[222]:


min_kwota_rozl = np.min(X['KWOTA_ROZLICZONA'])
X['KWOTA_ROZLICZONA_LOG']= X['KWOTA_ROZLICZONA'].apply(lambda x: np.log((x+ 1) - min_kwota_rozl))#min(X['KWOTA_ROZLICZONA'])))

X.drop('KWOTA_ROZLICZONA', axis=1, inplace=True)


# In[ ]:





# In[ ]:





# In[223]:


# na testowy

min_kwota_rozl = np.min(X_t['KWOTA_ROZLICZONA'])
X_t['KWOTA_ROZLICZONA_LOG']= X_t['KWOTA_ROZLICZONA'].apply(lambda x: np.log((x+ 1) - min_kwota_rozl))#min(X['KWOTA_ROZLICZONA'])))

X_t.drop('KWOTA_ROZLICZONA', axis=1, inplace=True)


# In[224]:


# zmienna grupa_wiekowa


# In[225]:


q1, q3= np.percentile(pom_tab['KWOTA_ROZLICZONA_LOG'],[25,75])
iqr = q3 - q1
upper_bound = q3 +(1.5 * iqr) 


# In[226]:


upper_bound


# In[227]:


import numpy as np
import pandas as pd
outliers=[]
def detect_outlier(data_1):
    
    threshold=3
    mean_1 = np.mean(data_1)
    std_1 =np.std(data_1)
    
    
    for y in data_1:
        z_score= (y - mean_1)/std_1 
        if np.abs(z_score) > threshold:
            outliers.append(y)
    return outliers


# In[228]:


detect_outlier(pom_tab['KWOTA_ROZLICZONA_LOG'])


# In[ ]:





# In[229]:


# pokazanie unikatowych wartości
np.unique(X.GRUPA_WIEKOWA )


# In[230]:


# tworzy tabele ze zmienną oraz zmienną- klucz ID_PACJENTA
pom_tab = tabela_pomocnicza('GRUPA_WIEKOWA','ID_PACJENTA')
pom_tab


# In[231]:


# ze względdu na niską liczność grupo łączy trzy ostatnie poziomy w jeden
def laczenie_3_ostatnich_grup(x):
    if x['GRUPA_WIEKOWA'] in ('(88,93]','(93,98]','(98,103]'):
        return '(88,103]'
    else:
        return x['GRUPA_WIEKOWA']
pom_tab['GRUPA_WIEKOWA'] = pom_tab.apply(laczenie_3_ostatnich_grup, axis=1)


# In[ ]:





# In[232]:


(ggplot(pom_tab, aes('GRUPA_WIEKOWA',fill='GRUPA_WIEKOWA', color = 'GRUPA_WIEKOWA'))
 + geom_bar()
 + geom_text(
     aes(label='stat(prop)*100', group=1, size=8),
     stat='count',
     nudge_y=0.125,
     va='bottom',
     format_string='{:.1f}%'
 )
 +theme(axis_text_x=element_text(rotation=90, hjust=1))
 +ggtitle('Histogram zmiennej GRUPA_WIEKOWA')
)


# In[233]:


X['GRUPA_WIEKOWA'] = X.apply(laczenie_3_ostatnich_grup, axis=1)


# In[234]:


# na testowym
X_t['GRUPA_WIEKOWA'] = X_t.apply(laczenie_3_ostatnich_grup, axis=1)


# In[235]:


# zmienna PLEC


# In[236]:


#pokazanie unikatowych wartości zmiennej
np.unique(X.PLEC)


# In[237]:


# globalna zmiana typu zmiennej na obiektowy
X.PLEC=X.PLEC.astype('object') 


# In[238]:


# na testowym
X_t.PLEC=X_t.PLEC.astype('object') 


# In[239]:


# tworzy tabele ze zmienną oraz zmienną- klucz ID_PACJENTA
pom_tab = tabela_pomocnicza('PLEC','ID_PACJENTA')
pom_tab


# In[240]:


wykres_czestosci('PLEC')


# In[241]:


# zmienna TERYT_POWIATU


# In[ ]:





# In[242]:


# unikatowe wartości zmiennej
np.unique(X.TERYT_POWIATU)


# In[243]:


# globalna zmiana typu zmiennej na object
X.TERYT_POWIATU=X.TERYT_POWIATU.astype('object') 


# In[244]:


# na testowym
X_t.TERYT_POWIATU=X_t.TERYT_POWIATU.astype('object') 


# In[245]:


# tworzy tabele ze zmienną oraz zmienną- klucz ID_PACJENTA
pom_tab = tabela_pomocnicza('TERYT_POWIATU','ID_PACJENTA')


# In[246]:


(ggplot(pom_tab, aes('TERYT_POWIATU',fill='TERYT_POWIATU', color = 'TERYT_POWIATU'))
 + geom_bar()
 + geom_text(
     aes(label='stat(prop)*100', group=1, size=8),
     stat='count',
     nudge_y=0.125,
     #va='bottom',
     format_string='{:.1f}%'
 )
 +theme(axis_text_x=element_blank())
 +ggtitle('Wykres częstości zmiennej TERYT_POWIATU')
)


# In[247]:


# grupowanie kodów powiatu w podregiony
def grupowanie(x):
    if x['TERYT_POWIATU'] in [1215,1211,1217]: #nowotarski
        return 1
    elif x['TERYT_POWIATU'] in [1218,1213,1203,1212]: #oświęcimski
        return 2
    elif x['TERYT_POWIATU'] in [1208,1214,1206,1209,1219,1201]: #krakowski
        return 3
    elif x['TERYT_POWIATU'] in [1202,1204,1263,1216]:#tarnowski
        return 4
    elif x['TERYT_POWIATU'] in [1205,1210,1262,1207]: #nowosądecki
        return 5
    elif x['TERYT_POWIATU'] in [1261]:#m.stołeczne kraków
        return 6
pom_tab['TERYT_POWIATU_PODREGION'] =pom_tab.apply(grupowanie , axis=1)


# In[248]:


# zmiana typu na object
pom_tab['TERYT_POWIATU_PODREGION']= pom_tab['TERYT_POWIATU_PODREGION'].astype('object') 


# In[249]:


#wykres
(ggplot(pom_tab, aes('TERYT_POWIATU_PODREGION',fill='TERYT_POWIATU_PODREGION', color = 'TERYT_POWIATU_PODREGION'))
 + geom_bar()
 + geom_text(
     aes(label='stat(prop)*100', group=1, size=8),
     stat='count',
     nudge_y=0.125,
     #va='bottom',
     format_string='{:.1f}%'
 )
 +theme(axis_text_x=element_blank())
 +ggtitle('Wykres częstości zmiennej TERYT_POWIATU_PODREGION')
)


# In[250]:


X['TERYT_POWIATU_PODREGION'] =  X.apply(grupowanie , axis=1)

X['TERYT_POWIATU_PODREGION']= X['TERYT_POWIATU_PODREGION'].astype('object') 

X = X.drop(['TERYT_POWIATU'], axis=1)


# In[251]:


# na testowym
X_t['TERYT_POWIATU_PODREGION'] =  X_t.apply(grupowanie , axis=1)

X_t['TERYT_POWIATU_PODREGION']= X_t['TERYT_POWIATU_PODREGION'].astype('object') 

X_t = X_t.drop(['TERYT_POWIATU'], axis=1) #w razie jakby była ta zmienna to usunąć


# In[ ]:





# In[ ]:





# In[ ]:





# In[252]:


# ID_RECEPTY tylko identyfikator - nic nie wnosi do analizy


# In[253]:


# dla zmiennych pochodzących z tabeli recepty tworzę tabele pomocznice, z badaną zmienną oraz zmienną ID_RECEPTY
# jest ona unikalna, a ID_PACJENTA 


# In[254]:


#zmienna ROK_REALIZACJI_RECEPTY


# In[255]:


# zmiana typu zmiennej na object
X.ROK_REALIZACJI_RECEPTY= X.ROK_REALIZACJI_RECEPTY.astype('object') 


# In[256]:


# zmiana typu zmiennej na object
X_t.ROK_REALIZACJI_RECEPTY = X_t.ROK_REALIZACJI_RECEPTY.astype('object') 


# In[257]:


# tworzy tabele ze zmienną oraz zmienną- klucz ID_RECEPTY
pom_tab = tabela_pomocnicza('ROK_REALIZACJI_RECEPTY','ID_RECEPTY')


# In[258]:


wykres_czestosci('ROK_REALIZACJI_RECEPTY')


# In[259]:


# zmienna TYDZIEN_POCZATKU_REALIZACJI


# In[260]:


# wartości unikatowe
np.unique(X.TYDZIEN_POCZATKU_REALIZACJI)


# In[261]:


# tworzy tabele ze zmienną oraz zmienną- klucz ID_RECEPTY
pom_tab = tabela_pomocnicza('TYDZIEN_POCZATKU_REALIZACJI','ID_RECEPTY')


# In[262]:


# wiersze, w których unikatowa wartość TYDZIEN_POCZATKU_REALIZACJI pojawia się więcej niż 133 razy
pom_tab[pom_tab['count_opposit']>133]


# In[263]:


# histogram
histogram('TYDZIEN_POCZATKU_REALIZACJI')


# In[264]:


plt.boxplot(pom_tab['TYDZIEN_POCZATKU_REALIZACJI'],labels=['TYDZIEN_POCZATKU_REALIZACJI'])
plt.show()


# In[265]:


#ZMIENNA KOD_ATC


# In[266]:


#ilość unikatowych wartości ( aż 49 )
len(np.unique(X.KOD_ATC))


# In[267]:


# wartości unikatowe zmiennej
np.unique(X.KOD_ATC)


# In[28]:


# tworzy tabele ze zmienną oraz zmienną- klucz ID_RECEPTY
pom_tab = tabela_pomocnicza('KOD_ATC','ID_RECEPTY')


# In[32]:


pom_tab['GLOWNA_GRUPA_ATC'] = pom_tab['KOD_ATC'].str[:1]
wykres_czestosci('GLOWNA_GRUPA_ATC')


# In[33]:


pom_tab = tabela_pomocnicza('KOD_ATC','ID_RECEPTY')


# In[34]:


#10 najliczniejszych kategorii
list(pom_tab[['KOD_ATC','count_opposit']].drop_duplicates().nlargest(14, 'count_opposit')['KOD_ATC'])


# In[35]:


pom_tab.groupby(['KOD_ATC'], as_index=False)['count_opposit']#.agg('nunique')#()#.nlargest(10, 'count_opposit')


# # Czemu odpowiadają poszczególne litery na pierwszym miejscu kodu ATC
# A – Przewód pokarmowy i metabolizm
# B – Krew i układ krwiotwórczy
# C – Układ sercowo-naczyniowy
# D – Dermatologia
# G – Układ moczowo-płciowy i hormony płciowe
# H – Leki hormonalne do stosowania wewnętrznego (bez hormonów płciowych)
# J – Leki stosowane w zakażeniach (przeciwinfekcyjne)
# L – Leki przeciwnowotworowe i immunomodulujące
# M – Układ mięśniowo-szkieletowy
# N – Ośrodkowy układ nerwowy
# P – Leki przeciwpasożytnicze, owadobójcze i repelenty
# R – Układ oddechowy
# S – Narządy wzroku i słuchu
# V – Różne (varia)

# In[36]:


def pogrupowanie(x):
    if x['KOD_ATC'] in ['C09', 'C10', 'C03', 'A10', 'A02', 'C08', 'M01', 'C07', 'J01', 'R03']:
        return x['KOD_ATC']
    else:
        return 'OTHER'
pom_tab['KOD_ATC']= pom_tab.apply(pogrupowanie, axis=1)


# In[37]:


wykres_czestosci('KOD_ATC')


# In[273]:


X['KOD_ATC']= X.apply(pogrupowanie, axis=1)


# In[274]:


# na testowym
X_t['KOD_ATC']= X_t.apply(pogrupowanie, axis=1)


# LICZBA_OPAKOWAN               object 
#  22  KOD_PROCEDURY                 float64
#  23  KOD_ROZPOZNANIA               int64  
#  24  CZY_GLOWNA                    object 
#  25  count_id                      int64  
#  26  enumrate_id                   object 
#  27  enumrate_pacjent_id           object 
#  28  enumrate_pacjent_id_kontaktu  int16  

# In[41]:


# ZMIENNA LICZBA_OPAKOWAN


# In[42]:


# zmiana separatora dziesiętnego na kropkę
X['LICZBA_OPAKOWAN'] = X['LICZBA_OPAKOWAN'].apply(przecinek)


# In[43]:


# na testowym
X_t['LICZBA_OPAKOWAN'] = X_t['LICZBA_OPAKOWAN'].apply(przecinek)


# In[44]:


# tworzy tabele ze zmienną oraz zmienną- klucz ID_RECEPTY
pom_tab = tabela_pomocnicza('LICZBA_OPAKOWAN','ID_RECEPTY')


# In[45]:


# the histogram of the data
nbins=20
plt.hist(pom_tab['LICZBA_OPAKOWAN'],nbins, facecolor='#79edde', alpha=0.75,ec="k")
plt.xlabel('LICZBA_OPAKOWAN')
plt.ylabel('Czestotliwość')
plt.title('Histogram zmiennej LICZBA_OPAKOWAN')
#plt.text(round(sr,2)+0.2, 4000, 'średia = \n'+ str(round(sr,2)), color='r')
plt.axis([0, 20, 0, 4000])
plt.grid(True)
#plt.plot([sr,sr],[0,5000], 'k-', lw=2, ls = '--', color='r')

plt.savefig('LICZBA_OPAKOWAN histogram.png', dpi=250, optimize=False,pad_inches=0.1, bbox_inches = "tight")
plt.show()


# In[46]:


plt.boxplot(pom_tab['LICZBA_OPAKOWAN'],labels=['LICZBA_OPAKOWAN'],whis=2)
plt.title('Wykres pudełkowy zmiennej LICZBA_OPAKOWAN')
plt.show()


# In[280]:


#winorizing


# In[281]:


min_10 = pom_tab['LICZBA_OPAKOWAN'].quantile(0.05)
print( min_10)
max_10 = pom_tab['LICZBA_OPAKOWAN'].quantile(0.95)
print( max_10)


# In[282]:


print(pom_tab['LICZBA_OPAKOWAN'].skew())


# In[283]:


np.unique(pom_tab['LICZBA_OPAKOWAN'])


# In[284]:


pom_tab['LICZBA_OPAKOWAN'] = np.where(pom_tab['LICZBA_OPAKOWAN'] <min_10, min_10,pom_tab['LICZBA_OPAKOWAN'])
pom_tab['LICZBA_OPAKOWAN']= np.where(pom_tab['LICZBA_OPAKOWAN'] >max_10, max_10,pom_tab['LICZBA_OPAKOWAN'])
print(pom_tab['LICZBA_OPAKOWAN'].skew())


# In[285]:


# kwadrat albo  pierwiastek
pom_tab['LICZBA_OPAKOWAN']= pom_tab['LICZBA_OPAKOWAN']**(2)


# In[286]:


# logarytm
pom_tab['LICZBA_OPAKOWAN']= pom_tab['LICZBA_OPAKOWAN'].apply(lambda x: np.log(x+1))


# In[287]:


# maksymalna wartość zmiennej 
pom_tab['LICZBA_OPAKOWAN'].max()


# In[288]:


pom_tab['LICZBA_OPAKOWAN'].min()


# In[289]:


# the histogram of the data
nbins=3
plt.hist(pom_tab['LICZBA_OPAKOWAN'],nbins, facecolor='#79edde', alpha=0.75,ec="k")
plt.xlabel('LICZBA_OPAKOWAN')
plt.ylabel('Czestotliwość')
plt.title('Histogram zmiennej LICZBA_OPAKOWAN')
#plt.text(round(sr,2)+0.2, 4000, 'średia = \n'+ str(round(sr,2)), color='r')
plt.axis([0, 3, 0, 4000])
plt.grid(True)
#plt.plot([sr,sr],[0,5000], 'k-', lw=2, ls = '--', color='r')

plt.savefig('LICZBA_OPAKOWAN histogram.png', dpi=250, optimize=False,pad_inches=0.1, bbox_inches = "tight")
plt.show()


# In[290]:


plt.boxplot(pom_tab['LICZBA_OPAKOWAN'],labels=['LICZBA_OPAKOWAN'],whis=2)
plt.title('Wykres pudełkowy zmiennej LICZBA_OPAKOWAN')
plt.show()


# In[291]:


X['LICZBA_OPAKOWAN'] = np.where(X['LICZBA_OPAKOWAN'] <min_10, min_10,X['LICZBA_OPAKOWAN'])
X['LICZBA_OPAKOWAN']= np.where(X['LICZBA_OPAKOWAN'] >max_10, max_10,X['LICZBA_OPAKOWAN'])


# In[292]:


# zbior testowy
X_t['LICZBA_OPAKOWAN'] = np.where(X_t['LICZBA_OPAKOWAN'] <min_10, min_10,X_t['LICZBA_OPAKOWAN'])
X_t['LICZBA_OPAKOWAN']= np.where(X_t['LICZBA_OPAKOWAN'] >max_10, max_10,X_t['LICZBA_OPAKOWAN'])


# In[293]:


pom_tab.quantile(q=0.25)


# In[294]:


pom_tab.quantile(q=0.5)


# In[295]:


pom_tab.quantile(q=0.75)


# In[296]:


# ZMIENNA KOD_PROCEDURY


# In[297]:


# zmiana typu zmiennej na string ponieważ będę chciał pogrupować wartości po pierwszej części tj, do kropki
X.KOD_PROCEDURY = X.KOD_PROCEDURY.astype('str')


# In[298]:


# zmiana na testowym
X_t.KOD_PROCEDURY = X_t.KOD_PROCEDURY.astype('str')


# In[299]:


# stworzeenie pogrupowanej zmiennej
def procedury(x):
    return x['KOD_PROCEDURY'][:x['KOD_PROCEDURY'].index('.')]


# In[300]:


X['KOD_PROCEDURY_GLOWNY'] = X.apply(procedury, axis=1)


# In[301]:


#test
X_t['KOD_PROCEDURY_GLOWNY'] = X_t.apply(procedury, axis=1)


# In[302]:


# ilość unikatowych wartości
len(np.unique(X.KOD_PROCEDURY_GLOWNY))


# In[303]:


# unikatowe wartości
np.unique(X.KOD_PROCEDURY_GLOWNY)


# In[304]:


# tworzy tabele ze zmienną oraz zmienną- klucz ID_EPIZODU
pom_tab = tabela_pomocnicza('KOD_PROCEDURY_GLOWNY','ID_EPIZODU')


# In[305]:


#10 najliczniejszych kategorii
l_largest= list(pom_tab[['KOD_PROCEDURY_GLOWNY','count_opposit']].drop_duplicates().nlargest(10, 'count_opposit')['KOD_PROCEDURY_GLOWNY'])


# In[306]:


# wydzielenie 10 nalicczniejszych głównych grup
def top_10_kat(x):
    if x['KOD_PROCEDURY_GLOWNY'] in l_largest:
        return x['KOD_PROCEDURY_GLOWNY']
    else:
        return 'OTHER'
    
pom_tab['TOP_10_KOD_PROCEDURY_GLOWNY'] = pom_tab.apply(top_10_kat, axis=1)


# In[307]:


pom_tab['KOD_PROCEDURY'] = pom_tab['TOP_10_KOD_PROCEDURY_GLOWNY']


# In[308]:


wykres_czestosci('KOD_PROCEDURY')


# In[309]:


X['TOP_10_KOD_PROCEDURY_GLOWNY'] = X.apply(top_10_kat, axis=1)

X['KOD_PROCEDURY'] = X['TOP_10_KOD_PROCEDURY_GLOWNY']

X = X.drop(['TOP_10_KOD_PROCEDURY_GLOWNY'], axis=1)
X = X.drop(['KOD_PROCEDURY_GLOWNY'], axis=1)


# In[310]:


# testowy

X_t['TOP_10_KOD_PROCEDURY_GLOWNY'] = X_t.apply(top_10_kat, axis=1)

X_t['KOD_PROCEDURY'] = X_t['TOP_10_KOD_PROCEDURY_GLOWNY']

X_t = X_t.drop(['TOP_10_KOD_PROCEDURY_GLOWNY'], axis=1)
X_t = X_t.drop(['KOD_PROCEDURY_GLOWNY'], axis=1)


# In[311]:


# ZMIENNA KOD_ROZPOZNANIA


# In[312]:


# ilość unikatowych wartości
len(np.unique(X.KOD_ROZPOZNANIA))


# In[313]:


# tworzy tabele ze zmienną oraz zmienną- klucz ID_KONTAKTU
pom_tab = tabela_pomocnicza('KOD_ROZPOZNANIA','ID_KONTAKTU')


# In[314]:


len(np.unique(pom_tab[pom_tab['count_opposit']<20]['KOD_ROZPOZNANIA']))


# In[315]:


len(np.unique(pom_tab.KOD_ROZPOZNANIA.astype('str').str[:5]))


# In[316]:


len(np.unique(pom_tab.KOD_ROZPOZNANIA.astype('str').str[:2]))


# In[317]:


# wydzielenie pierwszych znaków kodu do utworzenia grup
pom_tab['KOD_ROZPOZNANIA_GLOWNA_GRUPA'] = pom_tab.KOD_ROZPOZNANIA.astype('str').str[:2]


# In[318]:


# liczy ile razy występuje dana grupa
pom_tab['count_opposit_2znaki'] = pom_tab.groupby(['KOD_ROZPOZNANIA_GLOWNA_GRUPA'])['ID_KONTAKTU'].transform('count')


# In[319]:


#10 najmniej licznych kategorii
l_lowest= list(pom_tab[['KOD_ROZPOZNANIA_GLOWNA_GRUPA','count_opposit_2znaki']].drop_duplicates().nsmallest(6, 'count_opposit_2znaki')['KOD_ROZPOZNANIA_GLOWNA_GRUPA'])


# In[320]:


def top_10_kat(x):
    if x['KOD_ROZPOZNANIA_GLOWNA_GRUPA'] in l_lowest:
        return 'OTHER'
    else:
        return x['KOD_ROZPOZNANIA_GLOWNA_GRUPA']
        
pom_tab['TOP_13_KOD_ROZPOZNANIA_GLOWNY'] = pom_tab.apply(top_10_kat, axis=1)


# In[321]:


pom_tab['KOD_ROZPOZNANIA']= pom_tab['TOP_13_KOD_ROZPOZNANIA_GLOWNY']


# In[322]:


l_lowest


# In[323]:


(ggplot(pom_tab, aes('KOD_ROZPOZNANIA_GLOWNA_GRUPA',fill='KOD_ROZPOZNANIA_GLOWNA_GRUPA', color = 'KOD_ROZPOZNANIA_GLOWNA_GRUPA'))
 + geom_bar()
 + geom_text(
     aes(label='stat(prop)*100', group=1, size=8),
     stat='count',
     nudge_y=0.125,
     va='bottom',
     format_string='{:.1f}%'
 )
 +theme(axis_text_x=element_text(rotation=90, hjust=1))
 +ggtitle('Wykres częstości zmiennej\n KOD_ROZPOZNANIA_GLOWNA_GRUPA')
)


# In[324]:


(ggplot(pom_tab, aes('KOD_ROZPOZNANIA',fill='KOD_ROZPOZNANIA', color = 'KOD_ROZPOZNANIA'))
 + geom_bar()
 + geom_text(
     aes(label='stat(prop)*100', group=1, size=8),
     stat='count',
     nudge_y=0.125,
     va='bottom',
     format_string='{:.1f}%'
 )
 +theme(axis_text_x=element_text(rotation=90, hjust=1))
 +ggtitle('Wykres częstości zmiennej KOD_ROZPOZNANIA')
)


# In[325]:


X['KOD_ROZPOZNANIA_GLOWNA_GRUPA'] = X.KOD_ROZPOZNANIA.astype('str').str[:2]
X['TOP_13_KOD_ROZPOZNANIA_GLOWNY'] = X.apply(top_10_kat, axis=1)

X['KOD_ROZPOZNANIA'] = X['TOP_13_KOD_ROZPOZNANIA_GLOWNY']

X = X.drop(['TOP_13_KOD_ROZPOZNANIA_GLOWNY','KOD_ROZPOZNANIA_GLOWNA_GRUPA'], axis=1)


# In[326]:


#testowy
X_t['KOD_ROZPOZNANIA_GLOWNA_GRUPA'] = X_t.KOD_ROZPOZNANIA.astype('str').str[:2]
X_t['TOP_13_KOD_ROZPOZNANIA_GLOWNY'] = X_t.apply(top_10_kat, axis=1)

X_t['KOD_ROZPOZNANIA'] = X_t['TOP_13_KOD_ROZPOZNANIA_GLOWNY']

X_t = X_t.drop(['TOP_13_KOD_ROZPOZNANIA_GLOWNY','KOD_ROZPOZNANIA_GLOWNA_GRUPA'], axis=1)


# In[327]:


np.unique(X_t['KOD_ROZPOZNANIA'])


# In[328]:


np.unique(X['KOD_ROZPOZNANIA'])


# In[329]:


# ZMIENNA CZY_GLOWNA


# In[330]:


# tworzy tabele ze zmienną oraz zmienną- klucz ID_KONTAKTU
pom_tab = tabela_pomocnicza('CZY_GLOWNA','ID_KONTAKTU')


# In[331]:


wykres_czestosci('CZY_GLOWNA')


# In[332]:


# nie wiem czy to ok z tym kodem_rozpoznania


# In[333]:


pom_tab = pd.concat([X['CZY_GLOWNA'],X['KOD_ROZPOZNANIA'], X['ID_KONTAKTU']], axis=1)
pom_tab = pom_tab.drop_duplicates()
pom_tab['count'] = pom_tab.groupby(['ID_KONTAKTU'])['CZY_GLOWNA'].transform('count')
pom_tab['count_opposit'] = pom_tab.groupby(['CZY_GLOWNA'])['ID_KONTAKTU'].transform('count')
pom_tab[pom_tab['count']>1]


# In[334]:


(ggplot(pom_tab, aes('CZY_GLOWNA', fill='CZY_GLOWNA', color='CZY_GLOWNA'))
 + geom_bar()
 + geom_text(
     aes(label='stat(prop)*100', group=1),
     stat='count',
     nudge_y=0.125,
     va='bottom',
     format_string='{:.1f}%'
 )
 +ggtitle('Wykres częstości zmiennej CZY_GLOWNA')
)


# In[335]:


X['CZY_GLOWNA'] = X['CZY_GLOWNA'].astype('object')


# In[336]:


# na testowym
X_t['CZY_GLOWNA'] = X_t['CZY_GLOWNA'].astype('object')


# In[337]:


X = X.drop(['ID_PACJENTA', 'ID_EPIZODU', 'ID_KONTAKTU', 'ID_KSIEGI_GLOWNEJ' ], axis=1)


# In[338]:


# na testowym
X_t = X_t.drop(['ID_PACJENTA', 'ID_EPIZODU', 'ID_KONTAKTU', 'ID_KSIEGI_GLOWNEJ' ], axis=1)


# In[339]:


#---- zmiany na zmiennych na całym zbiorze treningowym 


# In[340]:


X = X.drop(['ROK_WZGL','TYDZIEN_UDARU', 'ID_RECEPTY'], axis=1)


# In[341]:


#na testowym
X_t = X_t.drop(['ROK_WZGL','TYDZIEN_UDARU', 'ID_RECEPTY'], axis=1)


# In[342]:


len(y[y['CZY_UDAR']==1])/ len(y[y['CZY_UDAR']==0])


# In[343]:


len(y[y['CZY_UDAR']==1])/len(y[y['CZY_UDAR']==0])


# In[344]:


X_t.to_csv('zbior_testowy_X_v3.csv')
y_t.to_csv('zmienna_objasniana_test_v3.csv')


# In[345]:


X.to_csv('zbior_treningowy_X_v3.csv')
y.to_csv('zmienna_objasniana_v3.csv')


# In[346]:


X.columns


# In[347]:


X_t.columns


# In[348]:


# Koniec analizy częstości i histogramów


# In[477]:


import pandas as pd, numpy as np


# In[478]:


X_t = pd.read_csv('zbior_testowy_X_v3.csv')
y_t = pd.read_csv('zmienna_objasniana_test_v3.csv')


# In[479]:


X = pd.read_csv('zbior_treningowy_X_v3.csv')
y = pd.read_csv('zmienna_objasniana_v3.csv')


# In[480]:


X.drop(columns='Unnamed: 0', inplace = True)
y.drop(columns='Unnamed: 0', inplace = True)
X_t.drop(columns='Unnamed: 0', inplace = True)
y_t.drop(columns='Unnamed: 0', inplace = True)


# In[481]:


X.columns


# In[482]:


len(X_t)/(len(X)+len(X_t))


# In[483]:


# staandaryzacjjaaa - mniej podatna na wartośści odstaające s 82


# In[484]:


np.unique(X_t['Enumerate_pacjent_id_kontaktu'])


# In[485]:


y_t[y_t['CZY_UDAR']==1]


# In[ ]:





# In[486]:


from sklearn import preprocessing


# In[487]:


# standaryzacja


# In[488]:


scaler = preprocessing.StandardScaler().fit(X[['KWOTA_ROZLICZONA_LOG',
'TYDZIEN_POCZATKU_KONTAKTU',
'TYDZIEN_KONCA_KONTAKTU',
'TYDZIEN_POCZATKU_REALIZACJI',
'LICZBA_OPAKOWAN']])


# In[489]:


kolumny_stand = scaler.transform(X[['KWOTA_ROZLICZONA_LOG',
'TYDZIEN_POCZATKU_KONTAKTU',
'TYDZIEN_KONCA_KONTAKTU',
'TYDZIEN_POCZATKU_REALIZACJI',
'LICZBA_OPAKOWAN']])


# In[490]:


X['KWOTA_ROZLICZONA_LOG']=kolumny_stand[:,0]
X['TYDZIEN_POCZATKU_KONTAKTU']=kolumny_stand[:,1]
X['TYDZIEN_KONCA_KONTAKTU']=kolumny_stand[:,2]
X['TYDZIEN_POCZATKU_REALIZACJI']=kolumny_stand[:,3]
X['LICZBA_OPAKOWAN']=kolumny_stand[:,4]


# In[491]:


#użycie tej samej średniej i odchylenia co w treningowym
kolumny_stand_test = scaler.transform(X_t[['KWOTA_ROZLICZONA_LOG',
'TYDZIEN_POCZATKU_KONTAKTU',
'TYDZIEN_KONCA_KONTAKTU',
'TYDZIEN_POCZATKU_REALIZACJI',
'LICZBA_OPAKOWAN']])

X_t['KWOTA_ROZLICZONA_LOG']=kolumny_stand_test[:,0]
X_t['TYDZIEN_POCZATKU_KONTAKTU']=kolumny_stand_test[:,1]
X_t['TYDZIEN_KONCA_KONTAKTU']=kolumny_stand_test[:,2]
X_t['TYDZIEN_POCZATKU_REALIZACJI']=kolumny_stand_test[:,3]
X_t['LICZBA_OPAKOWAN']=kolumny_stand_test[:,4]


# In[492]:


X.columns


# In[493]:


X.KOD_ZAKRESU = X.KOD_ZAKRESU.astype('str')
X_t.KOD_ZAKRESU = X_t.KOD_ZAKRESU.astype('str')
X.TYP_KOMORKI = X.TYP_KOMORKI.astype('str')
X_t.TYP_KOMORKI = X_t.TYP_KOMORKI.astype('str')
X.KOD_PROCEDURY = X.KOD_PROCEDURY.astype('str')
X_t.KOD_PROCEDURY = X_t.KOD_PROCEDURY.astype('str')
X_t.KOD_ROZPOZNANIA = X_t.KOD_ROZPOZNANIA.astype('str')
X.KOD_ROZPOZNANIA = X.KOD_ROZPOZNANIA.astype('str')


# In[ ]:





# In[494]:


# korelacja 


# In[495]:


np.unique(X.KOD_ZAKRESU.astype('str'))


# In[431]:


X.corr()


# In[432]:


from dython import nominal


# In[433]:


for i in X.columns:
    print(i)
    print(np.unique(X[i]))


# In[434]:


cor_tab = nominal.compute_associations(X,nominal_columns = ['ROK_SWIADCZENIA', 'KOD_ZAKRESU', 'KOD_PRODUKTU_JEDNOSTKOWEGO',
      'RODZAJ_SWIADCZEN', 'GRUPA_WIEKOWA', 'PLEC', 
       'ROK_REALIZACJI_RECEPTY',  'KOD_ATC', 'TYP_KOMORKI',
        'KOD_PROCEDURY', 'KOD_ROZPOZNANIA', 'CZY_GLOWNA',
       'Enumerate_pacjent_id_kontaktu',
       'TERYT_POWIATU_PODREGION'])


# In[435]:


# usuwa z wierszy
cor_tab = cor_tab[['KWOTA_ROZLICZONA_LOG','TYDZIEN_POCZATKU_KONTAKTU',
'TYDZIEN_KONCA_KONTAKTU','TYDZIEN_POCZATKU_REALIZACJI','LICZBA_OPAKOWAN']]


# In[436]:


# usuwa z kolumn
cor_tab = cor_tab.drop(['KWOTA_ROZLICZONA_LOG','TYDZIEN_POCZATKU_KONTAKTU',
'TYDZIEN_KONCA_KONTAKTU','TYDZIEN_POCZATKU_REALIZACJI','LICZBA_OPAKOWAN'])


# In[437]:


import seaborn as sns


# In[438]:


import matplotlib.pyplot as plt 


# In[439]:


# korelacja eta
fig, ax = plt.subplots(figsize = (12,7))
title = 'Korelacja eta'
plt.title(title,fontsize = 18)
ttl = ax.title
ttl.set_position([0.5,1.05])
ax.set_xticks([])
ax.set_yticks([])
#ax.axis('off')
sns.heatmap(cor_tab.T,cmap='RdYlGn',ax=ax)
plt.savefig('seabornPandas.png', dpi=100)
plt.show()


# In[440]:


# V Cramera


# In[441]:


v_cramer = nominal.compute_associations(X[['ROK_SWIADCZENIA', 'KOD_ZAKRESU', 'KOD_PRODUKTU_JEDNOSTKOWEGO',
      'RODZAJ_SWIADCZEN', 'GRUPA_WIEKOWA', 'PLEC', 
       'ROK_REALIZACJI_RECEPTY',  'KOD_ATC',
        'KOD_PROCEDURY', 'KOD_ROZPOZNANIA', 'CZY_GLOWNA',
       'Enumerate_pacjent_id_kontaktu',
       'TERYT_POWIATU_PODREGION']], nominal_columns=['ROK_SWIADCZENIA', 'KOD_ZAKRESU', 'KOD_PRODUKTU_JEDNOSTKOWEGO',
      'RODZAJ_SWIADCZEN', 'GRUPA_WIEKOWA', 'PLEC', 
       'ROK_REALIZACJI_RECEPTY',  'KOD_ATC', 'TYP_KOMORKI',
        'KOD_PROCEDURY', 'KOD_ROZPOZNANIA', 'CZY_GLOWNA',
       'Enumerate_pacjent_id_kontaktu',
       'TERYT_POWIATU_PODREGION'])


# In[442]:


fig, ax = plt.subplots(figsize = (12,7))
title = 'Korelacja V Cramera'
plt.title(title,fontsize = 18)
ttl = ax.title
ttl.set_position([0.5,1.05])
ax.set_xticks([])
ax.set_yticks([])
#ax.axis('off')
sns.heatmap(v_cramer.T,cmap='RdYlGn',ax=ax)
plt.savefig('vCramera.png', dpi=100)
plt.show()


# In[443]:


# spearman


# In[444]:


cor_spearman = X[['KWOTA_ROZLICZONA_LOG','TYDZIEN_POCZATKU_KONTAKTU',
'TYDZIEN_KONCA_KONTAKTU','TYDZIEN_POCZATKU_REALIZACJI','LICZBA_OPAKOWAN']].corr('spearman')


# In[445]:


cor_spearman


# In[446]:


fig, ax = plt.subplots(figsize = (12,7))
title = 'Korelacja Spearmana'
plt.title(title,fontsize = 18)
ttl = ax.title
ttl.set_position([0.5,1.05])
ax.set_xticks([])
ax.set_yticks([])
#ax.axis('off')
sns.heatmap(cor_spearman.T,cmap='RdYlGn',ax=ax)
plt.savefig('vCramera.png', dpi=100)
plt.show()


# In[447]:


# drop Tydzien_poczatku kontaktu, kod_produktu jednostkowego kod zakresu


# In[448]:


len(y)


# In[449]:


# przypisuje nazwy kolumn 
data_final_vars=X.columns.values.tolist()
y=y


# In[450]:


np.unique(X.KOD_ZAKRESU.astype('str'))


# In[451]:


np.unique(X_t.KOD_ZAKRESU.astype('str'))


# In[452]:


from sklearn.preprocessing import LabelEncoder


# In[453]:


col_object = list(X.select_dtypes(include=['object']).columns)
col_object


# In[454]:


X_pom = X


# In[455]:


for j in col_object:
    X_pom[j] = X_pom[j].astype('str')


# In[456]:


#KOD_ZAKRESU
#KOD_PRODUKTU_JEDNOSTKOWEGO
#GRUPA_WIEKOWA
#KOD_ATC
#KOD_PROCEDURY
#KOD_ROZPOZNANIA
#CZY_GLOWNA
#Enumerate_pacjent_id_kontaktu
#KOD_PROCEDURY_GLOWNY


# In[457]:


np.unique(X.KOD_ZAKRESU.astype('str'))


# In[458]:


lb_make = LabelEncoder()
d1={}
for i in col_object:
    print(i)
    lb_make.fit(X_pom[i])
    d1[i] = dict(zip(lb_make.classes_, lb_make.transform(lb_make.classes_)))
    X_pom[i] = lb_make.transform(X_pom[i])


# In[ ]:





# In[459]:


#importances = X_pom.drop('CZY_UDAR', axis=1).apply(lambda x: x.corr(X_pom.CZY_UDAR))


# In[460]:


#indices = np.argsort(importances)
#print(importances[indices])


# In[461]:


#zmienne_kor = []
#for i in range(0, len(indices)):
#    if np.abs(importances[i])>0.03:
#        zmienne_kor.append(X_pom.columns[i])


# In[462]:


#X_pom = X_pom[zmienne_kor]


# In[ ]:





# In[463]:


from statsmodels.stats.outliers_influence import variance_inflation_factor

def calc_vif(X):

    # Calculating VIF
    vif = pd.DataFrame()
    vif["variables"] = X.columns
    vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

    return(vif)


# In[464]:


vif_tab = calc_vif(X_pom)
vif_tab


# In[465]:


#X_pom = X_pom.drop(labels = ['TYDZIEN_POCZATKU_KONTAKTU'], axis=1 )


# In[466]:


X_pom = X_pom.drop(labels = ['TYDZIEN_KONCA_KONTAKTU'], axis=1 )


# In[467]:


vif_tab = calc_vif(X_pom)
vif_tab


# In[468]:


X_pom = X_pom.drop(labels = ['KOD_ZAKRESU'], axis=1 )
vif_tab = calc_vif(X_pom)
vif_tab


# In[ ]:





# In[469]:


X_pom = X_pom.drop(labels = ['GRUPA_WIEKOWA'], axis=1 )
vif_tab = calc_vif(X_pom)
vif_tab


# In[470]:


X_pom = X_pom.drop(labels = ['KOD_PRODUKTU_JEDNOSTKOWEGO'], axis=1 )
vif_tab = calc_vif(X_pom)
vif_tab


# In[471]:


X_pom = X_pom.drop(labels = ['PLEC'], axis=1 )
vif_tab = calc_vif(X_pom)
vif_tab


# In[472]:


zmienne_po_metodach_filter = list(vif_tab['variables'])


# In[473]:


print(zmienne_po_metodach_filter)


# In[474]:


vif_tab.to_csv('vif_tab_po_usunieciu_3.csv')


# In[496]:


zmienne_po_metodach_filter  = ['ROK_SWIADCZENIA', 'TYDZIEN_POCZATKU_KONTAKTU', 'TYP_KOMORKI',
                               'RODZAJ_SWIADCZEN', 'ROK_REALIZACJI_RECEPTY', 'TYDZIEN_POCZATKU_REALIZACJI', 'KOD_ATC',
                               'LICZBA_OPAKOWAN', 'KOD_PROCEDURY', 'KOD_ROZPOZNANIA', 'CZY_GLOWNA', 
                               'Enumerate_pacjent_id_kontaktu', 'KWOTA_ROZLICZONA_LOG', 'TERYT_POWIATU_PODREGION']


# In[497]:


X = X[zmienne_po_metodach_filter]


# In[498]:


X.columns


# In[499]:


X_t.columns


# In[500]:


X_t = X_t[zmienne_po_metodach_filter]


# In[501]:


from sklearn.preprocessing import OneHotEncoder 
from sklearn.compose import ColumnTransformer 


# In[502]:


cat_list = ['ROK_SWIADCZENIA', 
    'TYP_KOMORKI', 'RODZAJ_SWIADCZEN',
       'ROK_REALIZACJI_RECEPTY',  'KOD_ATC',
        'KOD_PROCEDURY',  'KOD_ROZPOZNANIA',
       'Enumerate_pacjent_id_kontaktu','CZY_GLOWNA',
       'TERYT_POWIATU_PODREGION']


# In[503]:


len(X)


# In[ ]:


len(y)


# In[463]:


# uzupełnienie kolumn które są w treningowym ale nie w testowym
#( testowy po prostu nie złapał wartości niektórych zmiennyh a przy tworzeniu 
#zmiennych binarnych powstały różnice w wymiarowości)


# In[505]:


for i in cat_list:
    print(i)
    rated_dummies = pd.get_dummies(X[i], prefix=i, prefix_sep='_',)
    X = pd.concat([X, rated_dummies], axis=1)
    X = X.drop([i], axis=1)


# In[506]:


for i in cat_list:

    rated_dummies = pd.get_dummies(X_t[i], prefix=i, prefix_sep='_',)
    X_t = pd.concat([X_t, rated_dummies], axis=1)
    X_t = X_t.drop([i], axis=1)


# In[507]:


for i in list(X.columns)[4:]:
    if i not in X_t.columns:
        X_t[i] = 0
        print(i)


# In[508]:


for i in list(X_t.columns)[4:]:
    if i not in X.columns:
        print(i)


# In[510]:


len(X.columns)


# In[511]:


len(X_t.columns)


# In[514]:


X_t.to_csv('Zbior_testowy_dummies_3.csv')


# In[ ]:


X.to_csv('Zbior_uczacy_dummies_3.csv')


# In[ ]:





# In[ ]:





# In[ ]:





# In[515]:





# In[464]:


import pandas as pd, numpy as np


# In[465]:


X = pd.read_csv('Zbior_uczacy_dummies_3.csv')


# In[466]:


X_t = pd.read_csv('Zbior_testowy_dummies_3.csv')


# In[467]:


y = pd.read_csv('zmienna_objasniana_v3.csv')
y_t = pd.read_csv('zmienna_objasniana_test_v3.csv')


# In[468]:


from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import seaborn as sn
import matplotlib.pyplot as plt


# In[469]:


from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression


# In[470]:


y.drop(columns = 'Unnamed: 0', inplace = True)
y_t.drop(columns = 'Unnamed: 0', inplace = True)


# In[471]:


X.drop(columns = 'Unnamed: 0', inplace = True)
X_t.drop(columns = 'Unnamed: 0', inplace = True)


# In[472]:


len(y[y['CZY_UDAR']==1])/len(y)


# In[473]:


from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.linear_model import LogisticRegression


# In[474]:


# przed połączeniem z innymi tabelami dzieli na zbiór testowy i treningowy aby móc wykonać undersampling na treningowym
from sklearn.model_selection import train_test_split
X_v, X_t_t, y_v, y_t_t = train_test_split(X_t, y_t,
                                                    stratify=y_t, 
                                                    test_size=0.5,random_state=105)


# In[475]:


from sklearn.metrics import confusion_matrix
import statsmodels.api as sm
from matplotlib.legend_handler import HandlerLine2D
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from matplotlib import pyplot
from sklearn import tree
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn import linear_model
from sklearn.metrics import auc
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_score


# In[ ]:





# In[ ]:





# In[476]:


# Regresja Logistyczna


# In[477]:


constant_filter = VarianceThreshold(threshold=0.1)
constant_filter.fit(X)
X.columns[constant_filter.get_support()]


# In[478]:


constant_columns = X.columns[constant_filter.get_support()]


# In[479]:


X_proba_2 = X[constant_columns]


# In[480]:


logreg = LogisticRegression(max_iter=1000, dual=False)
rfe = RFE(logreg, 15)
rfe = rfe.fit(X_proba_2, np.ravel(y))
print(rfe.support_)
print(rfe.ranking_)


# In[481]:


lista_zm_log = []
for i in range(len(rfe.support_)):
    if rfe.support_[i] == True:
          lista_zm_log.append(X.columns[i])


# In[482]:


lista_zm_log


# In[483]:


X_log = X[lista_zm_log]


# In[484]:


X_log =X_log.drop(columns='RODZAJ_SWIADCZEN_6')
X_t = X_t.drop(columns='RODZAJ_SWIADCZEN_6')


# In[485]:


result=logit_model.fit()
print(result.summary2())


# In[486]:


logreg = LogisticRegression(class_weight="balanced")
logreg.fit(X_log, np.ravel(y))


# In[487]:


lista_zm_log_proba_2 = list(X_log.columns)


# In[488]:


y_pred = logreg.predict(X_v[lista_zm_log_proba_2 ])
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_v[lista_zm_log_proba_2], y_pred)))


# In[489]:


#valid


# In[490]:


predicted_log_v = logreg.predict_proba(X_v[lista_zm_log_proba_2 ])[:,1]


# In[491]:


for j in range(1,10):
    thresh_log_v = []
    for i in predicted_log_v:
        if i >= j/10:
            thresh_log_v.append(1)
        else:
            thresh_log_v.append(0)
    print(j)
    print('F1 score {}'.format(f1_score(y_v, np.round(thresh_log_v), average='macro')))       


# In[492]:


thresh_log_v = []
for i in predicted_log_v:
    if i >= 0.8:
        thresh_log_v.append(1)
    else:
        thresh_log_v.append(0)


# In[493]:


print('Recall score {}'.format(recall_score(y_v, np.round(thresh_log_v), average='macro')))
print('F1 score {}'.format(f1_score(y_v, np.round(thresh_log_v), average='macro')))
print('Precision score {}'.format(precision_score(y_v, np.round(thresh_log_v), average='macro')))
print('Accuracy score {}'.format(accuracy_score(y_v, np.round(thresh_log_v))))


# In[494]:


confusion_matrix_v_log = confusion_matrix(y_v, thresh_log_v)
pd.DataFrame(confusion_matrix_v_log)


# In[495]:


#test


# In[496]:


predicted_log_t = logreg.predict_proba(X_t_t[lista_zm_log_proba_2 ])[:,1]
thresh_log_t = []
for i in predicted_log_t:
    if i >= 0.8:
        thresh_log_t.append(1)
    else:
        thresh_log_t.append(0)
print('F1 score {}'.format(f1_score(y_t_t, np.round(thresh_log_t), average='macro')))


# In[497]:


print('Recall score {}'.format(recall_score(y_t_t, np.round(thresh_log_t), average='macro')))
print('Precision score {}'.format(precision_score(y_t_t, np.round(thresh_log_t), average='macro')))
print('Accuracy score {}'.format(accuracy_score(y_t_t, np.round(thresh_log_t))))


# In[498]:


from sklearn.metrics import confusion_matrix
confusion_matrix_t_log = confusion_matrix(y_t_t, thresh_log_t)
pd.DataFrame(confusion_matrix_t_log)


# In[ ]:





# In[499]:



logit_roc_auc_t = roc_auc_score(y_t_t, thresh_log_t)
fpr_t, tpr_t, thresholds_t = roc_curve(y_t_t,thresh_log_t)

logit_roc_auc_v = roc_auc_score(y_v, thresh_log_v)
fpr_v, tpr_v, thresholds_v = roc_curve(y_v,thresh_log_v)


plt.figure()

plt.plot(fpr_t, tpr_t, label='Logistic Regression test (area = %0.2f)' % logit_roc_auc_t)

plt.plot(fpr_v, tpr_v, label='Logistic Regression valid (area = %0.2f)' % logit_roc_auc_v, color='green')

plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()


# In[500]:


# drzewo decyzyjne


# In[123]:


dtree = DecisionTreeClassifier(random_state = 105)
dtree.fit(X, y)  


# In[124]:


pred = dtree.predict(X_v)  


# In[126]:


max_depths = np.linspace(1, 32, 32, endpoint=True)
train_results = []
test_results = []


# In[21]:


for max_depth in max_depths:
    print(max_depth)
    dt = DecisionTreeClassifier(random_state=105,max_depth=max_depth)
    dt.fit(X, y)
    train_pred = dt.predict(X)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y, train_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    train_results.append(roc_auc)
    y_pred = dt.predict(X_v)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_v, y_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    test_results.append(roc_auc)

line1, = plt.plot(max_depths, train_results, 'b', label='Train AUC')
line2, = plt.plot(max_depths, test_results, 'r', label='Test AUC')
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('AUC score')
plt.xlabel('Tree depth')
plt.show()


# In[23]:


min_samples_splits = np.linspace(0.1, 1.0, 10, endpoint=True)
train_results = []
test_results = []


# In[24]:


for min_samples_split in min_samples_splits:
    print(min_samples_split)
    dt = DecisionTreeClassifier(random_state = 105, min_samples_split=min_samples_split, max_depth=10)
    dt.fit(X, y)
    train_pred = dt.predict(X)
    false_positive_rate, true_positive_rate, thresholds =    roc_curve(y, train_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    train_results.append(roc_auc)
    y_pred = dt.predict(X_v)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_v, y_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    test_results.append(roc_auc)
line1, = plt.plot(np.array(min_samples_splits), np.array(train_results), 'b', label='Train AUC')
line2, = plt.plot(min_samples_splits, test_results, 'r', label='Test AUC')
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('AUC score')
plt.xlabel('min samples split')


# In[25]:


plt.show()


# In[127]:


min_samples_leafs = np.linspace(0.1, 0.5, 5, endpoint=True)
train_results = []
test_results = []


# In[28]:


for min_samples_leaf in min_samples_leafs:
    print(min_samples_leaf)
    dt = DecisionTreeClassifier(random_state=105,min_samples_leaf=min_samples_leaf, max_depth=10, min_samples_split=0.2)
    dt.fit(X, y)
    train_pred = dt.predict(X)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y, train_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    train_results.append(roc_auc)
    y_pred = dt.predict(X_v)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_v, y_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    test_results.append(roc_auc)

line1, = plt.plot(min_samples_leafs, train_results, 'b', label='Train AUC')
line2, = plt.plot(min_samples_leafs, test_results, 'r', label='Test AUC')
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('AUC score')
plt.xlabel('min samples leaf')
plt.show()


# In[29]:


max_features = list(range(1,X.shape[1]))
train_results = []
test_results = []


# In[30]:


for max_feature in max_features:
    print(max_feature)
    dt = DecisionTreeClassifier(random_state=105, max_features=max_feature,min_samples_leaf =0.2, max_depth=10, min_samples_split=0.2)
    dt.fit(X, y)
    train_pred = dt.predict(X)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y, train_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    train_results.append(roc_auc)
    y_pred = dt.predict(X_v)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_v, y_pred)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    test_results.append(roc_auc)

line1, = plt.plot(max_features, train_results, 'b', label='Train AUC')
line2, = plt.plot(max_features, test_results, 'r', label='Test AUC')
plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
plt.ylabel('AUC score')
plt.xlabel('max features')
plt.show()


# In[501]:


# drzewo decyzyjne z wybranymi najlepszymi hiperparametrami
dtree = DecisionTreeClassifier(random_state=105, max_features=36,min_samples_leaf =0.2, max_depth=10, min_samples_split=0.2)
dtree.fit(X, y)  


# In[503]:


pred = dtree.predict(X_v)  


# In[ ]:





# In[505]:


print('F1 score {}'.format(f1_score(y_v, np.round(pred), average='micro')))


# In[506]:


# valid


# In[507]:


predicted_tree_v = dtree.predict_proba(X_v)[:,1]


# In[509]:


predicted_tree_v = dtree.predict_proba(X_v)[:,1]
for j in range(10,100,5):
    print((j/100))
    thresh_tree_v = []
    for i in predicted_tree_v:
        if i >= (j/100):
            thresh_tree_v.append(1)
        else:
            thresh_tree_v.append(0)
    print( roc_auc_score(y_v, thresh_tree_v))
    print('F1 score {}'.format(f1_score(y_v, np.round(thresh_tree_v), average='macro')))
    print('Accuracy score {}'.format(accuracy_score(y_v, np.round(thresh_tree_v))))


# In[511]:


predicted_tree_v = dtree.predict_proba(X_v)[:,1]


# In[513]:


thresh_tree_v = []
for i in predicted_tree_v:
    if i >= 0.55:
        thresh_tree_v.append(1)
    else:
        thresh_tree_v.append(0)

print('AUC ' + str(roc_auc_score(y_v, thresh_tree_v)))
print('F1 score {}'.format(f1_score(y_v, np.round(thresh_tree_v), average='macro')))


# In[514]:


print('Recall score {}'.format(recall_score(y_v, np.round(thresh_tree_v), average='macro')))
print('Precision score {}'.format(precision_score(y_v, np.round(thresh_tree_v), average='macro')))
print('Accuracy score {}'.format(accuracy_score(y_v, np.round(thresh_tree_v)))) 


# In[515]:


from sklearn.metrics import confusion_matrix
confusion_matrix_tree_v = confusion_matrix(y_v, thresh_tree_v)
pd.DataFrame(confusion_matrix_tree_v)


# In[ ]:





# In[516]:


#test


# In[517]:


predicted_tree_t = dtree.predict_proba(X_t_t)[:,1]

thresh_tree_t = []
for i in predicted_tree_t:
    if i >= 0.55:
        thresh_tree_t.append(1)
    else:
        thresh_tree_t.append(0)
print('F1 score {}'.format(f1_score(y_t_t, np.round(thresh_tree_t), average='macro')))
print('AUC ' + str(roc_auc_score(y_t_t, thresh_tree_t)))


# In[518]:


print('Recall score {}'.format(recall_score(y_t_t, np.round(thresh_tree_t), average='macro')))
print('Precision score {}'.format(precision_score(y_t_t, np.round(thresh_tree_t), average='macro')))
print('Accuracy score {}'.format(accuracy_score(y_t_t, np.round(thresh_tree_t))))


# In[519]:


from sklearn.metrics import confusion_matrix
confusion_matrix_tree_t = confusion_matrix(y_t_t, thresh_tree_t)
pd.DataFrame(confusion_matrix_tree_t)


# In[520]:


fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=300)
tree.plot_tree(dtree,
               feature_names = list(X.columns), 
               class_names=['0','1'],
               filled = True);


# In[521]:


roc_auc_v = roc_auc_score(y_v, thresh_tree_v)
fpr_v, tpr_v, thresholds_v = roc_curve(y_v, thresh_tree_v)

roc_auc_t = roc_auc_score(y_t_t, thresh_tree_t)
fpr_t, tpr_t, thresholds = roc_curve(y_t_t, thresh_tree_t)

plt.figure()
plt.plot(fpr_v, tpr_v, label='Drzewo decyzyjne valid(area = %0.4f)' % roc_auc_v)

plt.plot(fpr_t, tpr_t, label='Drzewo decyzyjne test (area = %0.4f)' % roc_auc_t, color = 'green')

plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[147]:


# las losowy


# In[522]:


# random forest model creation
rfc = RandomForestClassifier(random_state=105)
rfc.fit(X,np.ravel(y))
# predictions
rfc_predict = rfc.predict(X_v)


# In[524]:


print('F1 score {}'.format(f1_score(y_v, np.round(rfc_predict), average='macro')))


# In[528]:


false_positive_rate, true_positive_rate, thresholds = roc_curve(y_v, rfc_predict)
roc_auc = auc(false_positive_rate, true_positive_rate)
roc_auc


# In[527]:


confusion_matrix = confusion_matrix(y_v, rfc_predict)
print(confusion_matrix)


# In[14]:



# Number of trees in random forest


# In[223]:


n_estimators = [int(x) for x in np.linspace(start = 10, stop = 200, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [1,3,5,10,15,20]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [1, 2, 5, 10,15,20]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 5,10,15,20]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               "bootstrap": [True, False],
               "criterion": ["gini", "entropy"]
              }


# In[54]:


print(random_grid)


# In[224]:


rf = RandomForestClassifier(random_state = 105)


# In[226]:


rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 50, cv = 3, scoring='roc_auc' , verbose=60, n_jobs=1, random_state=105)

print('fit')
rf_random.fit(X, np.ravel(y))


# In[ ]:


pd.DataFrame(rf_random.cv_results_).to_csv('random_forest_random_search.csv')


# In[229]:


print('Best Score: ', rf_random.best_score_) 
print('Best Params: ', rf_random.best_params_) 


# In[ ]:


{'n_estimators': 178, 'min_samples_split': 15,
 'min_samples_leaf': 5, 'max_features': 'sqrt',
 'max_depth': 15, 'criterion': 'gini',
 'bootstrap': False}


# In[ ]:





# In[ ]:


'n_estimators': 178, 'min_samples_split': 15, 'min_samples_leaf': 5, 'max_features': 'sqrt', 'max_depth': 15, 'criterion': 'gini', 'bootstrap': False


# In[529]:



rfc = RandomForestClassifier(random_state=105,n_estimators= 178, min_samples_split= 15, min_samples_leaf = 5,
                             max_features= 'sqrt', max_depth= 15, criterion= 'gini', bootstrap= False
                            )
rfc.fit(X,np.ravel(y))

rfc_predict = rfc.predict(X_v)


# In[530]:


f_score = f1_score(y_v, np.round(rfc_predict), average='macro')

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_v, rfc_predict)
roc_auc = auc(false_positive_rate, true_positive_rate)


# In[531]:


rfc_predict_test = rfc.predict(X_t_t)


# In[532]:


f_score


# In[533]:


roc_auc


# In[534]:


print('F1 score ' +str(f1_score(y_t_t, rfc_predict_test, average='macro')))


# In[535]:


false_positive_rate, true_positive_rate, thresholds = roc_curve(y_t_t, rfc_predict_test)
print('AUC :'+str(auc(false_positive_rate, true_positive_rate)))


# In[536]:


print('Recall score {}'.format(recall_score(y_t_t, rfc_predict_test, average='macro')))
print('Precision score {}'.format(precision_score(y_t_t, rfc_predict_test, average='macro')))
print('Accuracy score {}'.format(accuracy_score(y_t_t, rfc_predict_test)))


# In[537]:


from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(rfc_predict_test, y_t_t)
pd.DataFrame(confusion_matrix)


# In[538]:


print('Recall score {}'.format(recall_score(y_v, rfc_predict, average='macro')))
print('Precision score {}'.format(precision_score(y_v, rfc_predict, average='macro')))
print('Accuracy score {}'.format(accuracy_score(y_v, rfc_predict)))


# In[539]:


from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(rfc_predict, y_v)
pd.DataFrame(confusion_matrix)


# In[540]:


roc_auc_v = roc_auc_score(y_v, rfc_predict)
fpr_v, tpr_v, thresholds_v = roc_curve(y_v, rfc_predict)

roc_auc_t = roc_auc_score(y_t_t, rfc_predict_test)
fpr_t, tpr_t, thresholds = roc_curve(y_t_t, rfc_predict_test)

plt.figure()
plt.plot(fpr_v, tpr_v, label='Las losowy valid(area = %0.4f)' % roc_auc_v)

plt.plot(fpr_t, tpr_t, label='Las losowy test (area = %0.4f)' % roc_auc_t, color = 'green')

plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()


# In[ ]:





# In[ ]:





# In[130]:


#svm


# In[541]:


clf = linear_model.SGDClassifier(loss="hinge", penalty="l2", max_iter=100, n_jobs=-1, random_state = 105)


# In[542]:


clf.fit(X, np.ravel(y))


# In[543]:


sdg_svm_pred = clf.predict(X_v)


# In[544]:



f_score = f1_score(y_v, np.round(sdg_svm_pred), average='macro')

false_positive_rate, true_positive_rate, thresholds = roc_curve(y_v, sdg_svm_pred)
roc_auc = auc(false_positive_rate, true_positive_rate)


# In[545]:


roc_auc 


# In[546]:


f_score


# In[359]:





# In[362]:


penalty = ['l1', 'l2', 'elasticnet'] 
alpha = [ 0.001, 0.01, 0.1, 1, 10] 
learning_rate = ['constant', 'optimal', 'invscaling', 'adaptive'] 
eta0 = [0.1, 0.5 ,1, 10] 


# In[ ]:


grid_search = pd.DataFrame(columns=['penalty','alpha','learning_rate','eta0', 'threshold','AUC','f1'])


# In[380]:


grid_search  = pd.DataFrame({'penalty':0,'alpha': 0,'learning_rate':0,'eta0':0,
                                  'threshold':0,'AUC':0,'f1':0}, index=[0])


# In[382]:


for a in  penalty:
    for b in alpha:
        for c in learning_rate:
            for d in eta0:
                clf = linear_model.SGDClassifier(penalty= a, loss= 'hinge', learning_rate= c,
                                                 eta0 = d, class_weight =  'balanced', alpha= b, random_state = 105)
                clf.fit(X, np.ravel(y))

                from sklearn.calibration import CalibratedClassifierCV
                calibrator = CalibratedClassifierCV(clf, cv='prefit')
                model=calibrator.fit(X, np.ravel(y))


                y_test_pred = model.predict_proba(X_v)

                predicted7 = y_test_pred[:,1] 
                for j in range(10,100,5):
                    print((j/100))
                    cos7 = []
                    for i in predicted7:
                        if i >= (j/100):
                            cos7.append(1)
                        else:
                            cos7.append(0)
                    auc_score = roc_auc_score(y_v, cos7)
                    f1_scores = f1_score(y_v, np.round(cos7), average='macro')
                    #print('Accuracy score {}'.format(accuracy_score(y_v, np.round(cos7))))
                    grid_search  = grid_search.append(pd.DataFrame({'penalty':a,'alpha': b,'learning_rate':c,'eta0':d,
                                  'threshold':j/100,'AUC':auc_score,'f1':f1_scores}, index=[0]), ignore_index=True)


# In[383]:


grid_search.to_csv('SGD_svm_grid_moj.csv')


# In[387]:


grid_search[grid_search['AUC']==max(grid_search['AUC'])]


# In[ ]:





# In[547]:


clf = linear_model.SGDClassifier(penalty= 'l2', loss= 'hinge', learning_rate= 'invscaling',
                                                 eta0 = 10, class_weight =  'balanced', alpha= 0.001, random_state = 105)
clf.fit(X, np.ravel(y))


# In[548]:


from sklearn.calibration import CalibratedClassifierCV
calibrator = CalibratedClassifierCV(clf, cv='prefit')
model=calibrator.fit(X, np.ravel(y))


# In[550]:


y_pred_v = model.predict_proba(X_v)

predicted7 = y_pred_v[:,1]
thresh_svm_sgd_v = []
for i in predicted7:
    if i >= 0.75:
        thresh_svm_sgd_v.append(1)
    else:
        thresh_svm_sgd_v.append(0)


# In[551]:


y_pred_t = model.predict_proba(X_t_t)
predicted8 = y_pred_t[:,1] 

thresh_svm_sgd_t = []
for i in predicted8:
    if i >= 0.75:
        thresh_svm_sgd_t.append(1)
    else:
        thresh_svm_sgd_t.append(0)


# In[552]:


#test


# In[553]:


print('Recall score '+str(metrics.recall_score(y_t_t, np.round(thresh_svm_sgd_t))))
print('Precision score {}'.format(precision_score(y_t_t, np.round(thresh_svm_sgd_t), average='macro')))
print('Accuracy score {}'.format(accuracy_score(y_t_t, np.round(thresh_svm_sgd_t))))
print('f1 score '+str(f1_score(y_t_t, np.round(thresh_svm_sgd_t), average='macro')))
print('AUC score '+str(metrics.roc_auc_score(y_t_t, np.round(thresh_svm_sgd_t))))


# In[554]:


from sklearn.metrics import confusion_matrix
confusion_matrix_svm_v = confusion_matrix(y_t_t, np.round(thresh_svm_sgd_t))
pd.DataFrame(confusion_matrix_svm_v)


# In[555]:


# valid


# In[556]:


print('Recall score '+str(metrics.recall_score(y_v, thresh_svm_sgd_v)))
print('Precision score {}'.format(precision_score(y_v, thresh_svm_sgd_v, average='macro')))
print('Accuracy score {}'.format(accuracy_score(y_v, thresh_svm_sgd_v)))
print('f1 score '+str(f1_score(y_v, np.round(thresh_svm_sgd_v), average='macro')))
print('AUC score '+str(metrics.roc_auc_score(y_v, thresh_svm_sgd_v)))


# In[557]:


from sklearn.metrics import confusion_matrix
confusion_matrix_svm_t = confusion_matrix(y_v, np.round(thresh_svm_sgd_v))
pd.DataFrame(confusion_matrix_svm_t)


# In[558]:


# roc


# In[559]:


roc_auc_v = roc_auc_score(y_v, thresh_svm_sgd_v)
fpr_v, tpr_v, thresholds_v = roc_curve(y_v, thresh_svm_sgd_v)

roc_auc_t = roc_auc_score(y_t_t, thresh_svm_sgd_t)
fpr_t, tpr_t, thresholds = roc_curve(y_t_t, thresh_svm_sgd_t)

plt.figure()
plt.plot(fpr_v, tpr_v, label='Svm z wykorystaniem SGD valid(area = %0.2f)' % roc_auc_v)

plt.plot(fpr_t, tpr_t, label='Svm z wykorystaniem SGD test (area = %0.2f)' % roc_auc_t, color = 'green')

plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()      


# In[ ]:





# In[ ]:





# In[ ]:





# Sieć neuronowa

# In[560]:


clf_neur = MLPClassifier(solver='adam', alpha=1e-5,
                     hidden_layer_sizes=(2,5), random_state=105)


# In[561]:


clf_neur.fit(X, np.ravel(y))


# In[562]:


pred_neur = clf_neur.predict(X_v)


# In[563]:


metrics.roc_auc_score(y_v, pred_neur)


# In[564]:


confusion_matrix = confusion_matrix(y_v, pred_neur)
print(confusion_matrix)


# In[ ]:





# In[ ]:


# tuning hiperparametrów i thresholdu


# In[432]:


hidden_layer_sizes= [(10,30,10),(20,),(2,5,2), (5,10,10,5)]
activation= ['identity', 'logistic', 'tanh', 'relu']
solver=[ 'adam']
alpha= [0.0001, 0.05,0.005]
learning_rate= ['invscaling','adaptive']


# In[ ]:


grid_search_neuron = pd.DataFrame(columns=['hidden_layer_sizes','activation','alpha','learning_rate', 'threshold','AUC','f1'])


# In[435]:


grid_search_neuron  = pd.DataFrame({'hidden_layer_sizes':0,'activation': 0,'alpha':0,'learning_rate':0,
                                  'threshold':0,'AUC':0,'f1':0}, index=[0])


# In[439]:


for a in  hidden_layer_sizes:
    for b in activation:
        for c in alpha:
            for d in learning_rate:
                clf_neur = MLPClassifier(hidden_layer_sizes = a, activation=b, alpha=c ,learning_rate= d, random_state=105)
                clf_neur.fit(X, np.ravel(y))

                y_test_pred = clf_neur.predict_proba(X_v)[:,1]

                predicted_neur = y_test_pred  
                for j in range(10,100,5):
                    print((j/100))
                    cos_neur = []
                    for i in predicted_neur:
                        if i >= (j/100):
                            cos_neur.append(1)
                        else:
                            cos_neur.append(0)
                    auc_score = roc_auc_score(y_v, cos_neur)
                    f1_scores = f1_score(y_v, np.round(cos_neur), average='macro')
                    #print(pd.DataFrame({'hidden_layer_sizes':a,'activation': b,'alpha':c,'learning_rate':d,
                    #              'threshold':j/100,'AUC':auc_score,'f1':f1_scores}))
                    #print('Accuracy score {}'.format(accuracy_score(y_v, np.round(cos7))))
                    grid_search_neuron  = grid_search_neuron.append(pd.DataFrame({'hidden_layer_sizes':str(a),'activation': b,'alpha':c,'learning_rate':d,
                                  'threshold':j/100,'AUC':auc_score,'f1':f1_scores}, index=[0]), ignore_index=True)


# In[440]:


grid_search_neuron.to_csv('grid_search_neuron.csv')


# In[442]:


grid_search_neuron[grid_search_neuron['AUC']==max(grid_search_neuron['AUC'])]


# In[443]:


grid_search_neuron[grid_search_neuron['f1']==max(grid_search_neuron['f1'])]


# In[ ]:





# In[565]:


clf_neur = MLPClassifier(hidden_layer_sizes = (10, 30, 10), activation='identity', alpha=0.05 ,learning_rate= 'invscaling', random_state=105)
clf_neur.fit(X, np.ravel(y))


# In[566]:


# valid


# In[567]:


y_pred_neur_v = clf_neur.predict_proba(X_v)[:,1]

predicted_neur = y_pred_neur_v

thresh_neur_v = []
for i in predicted_neur:
    if i >= 0.9:
        thresh_neur_v.append(1)
    else:
        thresh_neur_v.append(0)


# In[568]:


print('Recall score {}'.format(recall_score(y_v, np.round(thresh_neur_v), average='macro')))
print('Precision score {}'.format(precision_score(y_v, np.round(thresh_neur_v), average='macro')))
print('Accuracy score {}'.format(accuracy_score(y_v, np.round(thresh_neur_v))))
print('F1 score {}'.format(f1_score(y_v, np.round(thresh_neur_v), average='macro')))
print('AUC score {}'.format(roc_auc_score(y_v, np.round(thresh_neur_v))))


# In[569]:


from sklearn.metrics import confusion_matrix
confusion_matrix_neur_v = confusion_matrix(y_v, np.round(thresh_neur_v))
pd.DataFrame(confusion_matrix_neur_v)


# In[ ]:


# test 


# In[570]:


y_pred_t = clf_neur.predict_proba(X_t_t)[:,1]

predicted_neur_t = y_pred_t

thresh_neur_t = []
for i in predicted_neur_t:
    if i >= 0.9:
        thresh_neur_t.append(1)
    else:
        thresh_neur_t.append(0)


# In[571]:


print('Recall score {}'.format(recall_score(y_t_t, np.round(thresh_neur_t), average='macro')))
print('Precision score {}'.format(precision_score(y_t_t, np.round(thresh_neur_t), average='macro')))
print('Accuracy score {}'.format(accuracy_score(y_t_t, np.round(thresh_neur_t))))
print('F1 score {}'.format(f1_score(y_t_t, np.round(thresh_neur_t), average='macro')))
print('AUC score {}'.format(roc_auc_score(y_t_t, np.round(thresh_neur_t))))


# In[572]:


from sklearn.metrics import confusion_matrix
confusion_matrix_neur_t = confusion_matrix(y_t_t, np.round(thresh_neur_t))
pd.DataFrame(confusion_matrix_neur_t)


# In[573]:


roc_auc_v = roc_auc_score( y_v, np.round(thresh_neur_v))

fpr_v, tpr_v, thresholds_v = roc_curve(y_v, np.round(thresh_neur_v))

roc_auc_t = roc_auc_score( y_t_t, np.round(thresh_neur_t))
fpr_t, tpr_t, thresholds = roc_curve( y_t_t, np.round(thresh_neur_t))

plt.figure()
plt.plot(fpr_v, tpr_v, label='Sieć neuronowa valid(area = %0.5f)' % roc_auc_v)

plt.plot(fpr_t, tpr_t, label='Sieć neuronowa test (area = %0.5f)' % roc_auc_t, color = 'green')

plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()  


# In[ ]:





# In[574]:


# ROC wszystkie algorytmy


# In[ ]:


# valid


# In[576]:


roc_auc_log = roc_auc_score( y_v, np.round(thresh_log_v))
fpr_log, tpr_log, thresholds_log = roc_curve(y_v, np.round(thresh_log_v))

roc_auc_tree = roc_auc_score( y_v, np.round(thresh_tree_v))
fpr_tree, tpr_tree, thresholds_tree = roc_curve(y_v, np.round(thresh_tree_v))

roc_auc_rf = roc_auc_score( y_v, np.round(rfc_predict))
fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_v, np.round(rfc_predict))

roc_auc_svm_sgd = roc_auc_score( y_v, np.round(thresh_svm_sgd_v))
fpr_svm_sgd, tpr_svm_sgd, thresholds_svm_sgd = roc_curve(y_v, np.round(thresh_svm_sgd_v))

roc_auc_neur = roc_auc_score( y_v, np.round(thresh_neur_v))
fpr_neur, tpr_neur, thresholds_neur = roc_curve(y_v, np.round(thresh_neur_v))




plt.figure()
plt.plot(fpr_log, tpr_log, label='Regresja Logistyczna valid (area = %0.5f)' % roc_auc_log)

plt.plot(fpr_tree, tpr_tree, label='Drzewo decyzyjne valid (area = %0.5f)' % roc_auc_tree, color = 'green')

plt.plot(fpr_rf, tpr_rf, label='Las losowy valid (area = %0.5f)' % roc_auc_rf, color = 'magenta')

plt.plot(fpr_svm_sgd, tpr_svm_sgd, label='Svm z użyciem SGD valid (area = %0.5f)' % roc_auc_svm_sgd, color = 'black')

plt.plot(fpr_neur, tpr_neur, label='Sieć neuronowa valid (area = %0.5f)' % roc_auc_neur, color = 'purple')



plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()  


# In[59]:


# test


# In[579]:


roc_auc_log = roc_auc_score( y_t_t, np.round(thresh_log_t))
fpr_log, tpr_log, thresholds_log = roc_curve(y_t_t, np.round(thresh_log_t))

roc_auc_tree = roc_auc_score( y_t_t, np.round(thresh_tree_t))
fpr_tree, tpr_tree, thresholds_tree = roc_curve(y_t_t, np.round(thresh_tree_t))

roc_auc_rf = roc_auc_score( y_t_t, np.round(rfc_predict_test))
fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_t_t, np.round(rfc_predict_test))

roc_auc_svm_sgd = roc_auc_score( y_t_t, np.round(thresh_svm_sgd_t))
fpr_svm_sgd, tpr_svm_sgd, thresholds_svm_sgd = roc_curve(y_t_t, np.round(thresh_svm_sgd_t))

roc_auc_neur = roc_auc_score( y_t_t, np.round(thresh_neur_t))
fpr_neur, tpr_neur, thresholds_neur = roc_curve(y_t_t, np.round(thresh_neur_t))




plt.figure()
plt.plot(fpr_log, tpr_log, label='Regresja Logistyczna test (area = %0.5f)' % roc_auc_log)

plt.plot(fpr_tree, tpr_tree, label='Drzewo decyzyjne test (area = %0.5f)' % roc_auc_tree, color = 'green')

plt.plot(fpr_rf, tpr_rf, label='Las losowy test (area = %0.5f)' % roc_auc_rf, color = 'magenta')

plt.plot(fpr_svm_sgd, tpr_svm_sgd, label='Svm z użyciem SGD test (area = %0.5f)' % roc_auc_svm_sgd, color = 'black')

plt.plot(fpr_neur, tpr_neur, label='Sieć neuronowa test (area = %0.5f)' % roc_auc_neur, color = 'purple')



plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()  


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




