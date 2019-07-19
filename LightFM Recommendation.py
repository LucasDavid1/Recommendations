import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from sklearn import preprocessing
from lightfm import LightFM
from scipy.sparse import csr_matrix 
from scipy.sparse import coo_matrix 
from sklearn.metrics import roc_auc_score
import time
from lightfm.evaluation import auc_score
import pickle
import re
import seaborn as sns
books = pd.read_csv('/home/jovyan/work/Data/BX-Books.csv', sep=';', error_bad_lines=False, encoding="latin-1")
books.columns = ['ISBN', 'bookTitle', 'bookAuthor', 'yearOfPublication', 'publisher', 'imageUrlS', 'imageUrlM', 'imageUrlL']
users = pd.read_csv('/home/jovyan/work/Data/BX-Users.csv', sep=';', error_bad_lines=False, encoding="latin-1")
users.columns = ['userID', 'Location', 'Age']
ratings = pd.read_csv('/home/jovyan/work/Data/BX-Book-Ratings.csv', sep=';', error_bad_lines=False, encoding="latin-1")
ratings.columns = ['userID', 'ISBN', 'bookRating']

# Data Cleaning

books.drop(['imageUrlS', 'imageUrlM', 'imageUrlL'],axis=1,inplace=True)
# Se reasignan los valores de la fila ISBN == '0789466953'
books.loc[books.ISBN == '0789466953','yearOfPublication'] = 2000
books.loc[books.ISBN == '0789466953','bookAuthor'] = "James Buckley"
books.loc[books.ISBN == '0789466953','publisher'] = "DK Publishing Inc"
books.loc[books.ISBN == '0789466953','bookTitle'] = "DK Readers: Creating the X-Men, How Comic Books Come to Life (Level 4: Proficient Readers)"
# Se reasignan los valores de la fila ISBN == '078946697X'
books.loc[books.ISBN == '078946697X','yearOfPublication'] = 2000
books.loc[books.ISBN == '078946697X','bookAuthor'] = "Michael Teitelbaum"
books.loc[books.ISBN == '078946697X','publisher'] = "DK Publishing Inc"
books.loc[books.ISBN == '078946697X','bookTitle'] = "DK Readers: Creating the X-Men, How It All Began (Level 4: Proficient Readers)"
# Se reasignan los valores de la fila ISBN == '2070426769'
books.loc[books.ISBN == '2070426769','yearOfPublication'] = 2003
books.loc[books.ISBN == '2070426769','bookAuthor'] = "Jean-Marie Gustave Le ClÃ?Â©zio"
books.loc[books.ISBN == '2070426769','publisher'] = "Gallimard"
books.loc[books.ISBN == '2070426769','bookTitle'] = "Peuple du ciel, suivi de 'Les Bergers"
# Ahora que están bine los años, se pasa a valores numéricos
books.yearOfPublication=pd.to_numeric(books.yearOfPublication, errors='coerce')
# Se deja en NULL los años 0 o mayores a 2006
books.loc[(books.yearOfPublication > 2006) | (books.yearOfPublication == 0),'yearOfPublication'] = np.NAN
# Estos NULL se rellenan con el promedio
books.yearOfPublication.fillna(round(books.yearOfPublication.mean()), inplace=True)
# Los publisher con nulos se rellenan con 'other'
books.loc[(books.ISBN == '193169656X'),'publisher'] = 'other'
books.loc[(books.ISBN == '1931696993'),'publisher'] = 'other'
# Las edades menores a 5 y mayores a 90 se reemplazan por el promedio y se pasan a integer
users.loc[(users.Age > 90) | (users.Age < 5), 'Age'] = np.nan
users.Age = users.Age.fillna(users.Age.mean())
users.Age = users.Age.astype(np.int32)
#Se calzan las tablas para que tengan los mismos id de usuarios e ítems
ratings_new = ratings[ratings.ISBN.isin(books.ISBN)]
ratings = ratings[ratings.userID.isin(users.userID)]
ratings_explicit = ratings_new[ratings_new.bookRating != 0]

# Usuarios que hayan rankeado al menos 20 libros
# Y libros que hayan sido rankeados por lo menos por 20 usuarios

counts1 = ratings_explicit['userID'].value_counts()
ratings_explicit = ratings_explicit[ratings_explicit['userID'].isin(counts1[counts1 >= 20].index)]
counts = ratings_explicit['bookRating'].value_counts()
ratings_explicit = ratings_explicit[ratings_explicit['bookRating'].isin(counts[counts >= 20].index)]
ratings_explicit.shape

######## Training ########
# Divide el DataFrame en train y test trasnformándolos en una matriz de esparción.
# También devuelve raw_train_df que es train en forma de dataframe.
def informed_train_test(rating_df, train_ratio):
    split_cut = np.int(np.round(rating_df.shape[0] * train_ratio))
    train_df = rating_df.iloc[0:split_cut]
    test_df = rating_df.iloc[split_cut::]
    # Deja a los usuarios y libros que estén en train y en test
    test_df = test_df[(test_df['userID'].isin(train_df['userID'])) & (test_df['ISBN'].isin(train_df['ISBN']))]
    id_cols = ['userID', 'ISBN']
    trans_cat_train = dict()
    trans_cat_test = dict()
    for k in id_cols:
        cate_enc = preprocessing.LabelEncoder()
        # Transforma todas las columnas a categorías numéricas
        trans_cat_train[k] = cate_enc.fit_transform(train_df[k].values)
        trans_cat_test[k] = cate_enc.transform(test_df[k].values)

    # --- Encode ratings:
    cate_enc = preprocessing.LabelEncoder()
    ratings = dict()
    # Transforma la columna bookRating a categórica numérica
    ratings['train'] = cate_enc.fit_transform(train_df.bookRating)
    # La transforma sin hacerle fit (si se repite un valor, se deja, a diferencia de fit_transform)
    ratings['test'] = cate_enc.transform(test_df.bookRating)

    n_users = len(np.unique(trans_cat_train['userID']))
    n_items = len(np.unique(trans_cat_train['ISBN']))


    # class scipy.sparse.coo_matrix(arg1, shape=None, dtype=None, copy=False)
    # arg1: (data,(row, col))
    # coo_matrix toma los valores que hay en data y la ubica en una matriz con la 
    # ubicación correspondiente de (row, col)
    train = coo_matrix((ratings['train'], (trans_cat_train['userID'], \
                                                          trans_cat_train['ISBN'])) \
                                      , shape=(n_users, n_items))
    test = coo_matrix((ratings['test'], (trans_cat_test['userID'], \
                                                        trans_cat_test['ISBN'])) \
                                     , shape=(n_users, n_items))
    # Retorna una matriz de esparción, como la matriz de interacción, pero con los parámetros entregados.
    return train, test, train_df

# Se divide la data 80% train y 20% test
train, test, raw_train_df = informed_train_test(ratings_explicit, .8)

# Se calcula el área bajo la curva (AUC)
start_time = time.time()
# Se crea el modelo con la función loss='Weighted Approximated-Rank Pairwise (warp)'
# WARP: Maximises the rank of positive examples by repeatedly sampling negative examples until rank violating one is found.
model=LightFM(no_components=115,learning_rate=0.027,loss='warp')
# Aquí se le entrega una matriz de esparción (la mayoría de sus elementos son 0) y se entrena el modelo.
model.fit(train,epochs=12,num_threads=4)

auc_train = auc_score(model, train).mean()
auc_test = auc_score(model, test).mean()

print("--- Run time:  {} mins ---".format((time.time() - start_time)/60))
print("Train AUC Score: {}".format(auc_train))
print("Test AUC Score: {}".format(auc_test))



###### Interaction Matrix

user_item_matrix = raw_train_df.pivot(index='userID', columns='ISBN', values='bookRating')
user_item_matrix.fillna(0, inplace = True)
user_item_matrix = user_item_matrix.astype(np.int32)
print(user_item_matrix.shape)
user_item_matrix.head()

# Diccionario de usuarios e ítems
def user_item_dikts(interaction_matrix, items_df):
    user_ids = list(interaction_matrix.index)
    user_dikt = {}
    counter = 0 
    for i in user_ids:
        user_dikt[i] = counter
        counter += 1

    item_dikt ={}
    for i in range(items_df.shape[0]):
        item_dikt[(items_df.loc[i,'ISBN'])] = items_df.loc[i,'bookTitle']
    
    return user_dikt, item_dikt   

# User Recommendations
def similar_recommendation(model, interaction_matrix, user_id, user_dikt, 
                               item_dikt,threshold = 0,number_rec_items = 15):


    n_users, n_items = interaction_matrix.shape
    user_x = user_dikt[user_id]
    scores = pd.Series(model.predict(user_x,np.arange(n_items))) # A .predict se le pasan los usuarios a quien 
    # recomendar y los items para calcular la afinidad entre usuario-item
    scores.index = interaction_matrix.columns #Se obtiene el score dela matriz de interacción y se ordena
    scores = list(pd.Series(scores.sort_values(ascending=False).index))
    
    known_items = list(pd.Series(interaction_matrix.loc[user_id,:][interaction_matrix.loc[user_id,:] > threshold].index).sort_values(ascending=False))
    # Se obtienen los items con los que ya interactuó el usuario y valoró por sobre el threshold
    scores = [x for x in scores if x not in known_items]
    score_list = scores[0:number_rec_items]
    known_items = list(pd.Series(known_items).apply(lambda x: item_dikt[x])) # Se aplica el diccionario para que aparezcan los nombres en lugar de los ids
    scores = list(pd.Series(score_list).apply(lambda x: item_dikt[x]))

    print("Items that were liked by the User:")
    counter = 1
    for i in known_items[:25]:
        print(str(counter) + '- ' + i)
        counter+=1

    print("\n Recommended Items:")
    counter = 1
    for i in scores:
        print(str(counter) + '- ' + i)
        counter+=1

# Recomendación de un item para una lista de usuarios
def users_for_item(model,interaction_matrix,ISBN,number_of_user):

    n_users, n_items = interaction_matrix.shape
    # x es una lista de los book_id
    x = np.array(interaction_matrix.columns)
    # A .predict se le pasan todos los usuarios y se calcula la afinidad con un item en específico
                                    # searchsorted busca en la lista x la posición de ISBN
                                    # np.repeat repite n_users veces el valor que resulte de searchsorted
    scores = pd.Series(model.predict(np.arange(n_users), np.repeat(x.searchsorted(ISBN),n_users)))
    # Se crea una lista con los usuarios con mayor score
    user_list = list(interaction_matrix.index[scores.sort_values(ascending=False).head(number_of_user).index])
    return user_list 

# Recomendación de ítems similares a un item en particular
from sklearn.metrics.pairwise import cosine_similarity

def item_emdedding_distance_matrix(model,interaction_matrix):

    # Se crea una matriz de similitud de los ítems y se toman los valores en vectores
    df_item_norm_sparse = csr_matrix(model.item_embeddings)
    similarities = cosine_similarity(df_item_norm_sparse)
    item_emdedding_distance_matrix = pd.DataFrame(similarities)
    item_emdedding_distance_matrix.columns = interaction_matrix.columns
    item_emdedding_distance_matrix.index = interaction_matrix.columns
    return item_emdedding_distance_matrix

def also_bought_recommendation(item_emdedding_distance_matrix, item_id, 
                             item_dikt, n_items = 4):

    # Se toma la matriz de afinidad de los ítems y se ordena en base a la similitud con un item en específico
    recommended_items = list(pd.Series(item_emdedding_distance_matrix.loc[item_id,:]. \
                                  sort_values(ascending = False).head(n_items+1). \
                                  index[1:n_items+1]))
    
    print("Item of interest :{}".format(item_dikt[item_id]))
    print("Items that are frequently bought together:")
    counter = 1
    for i in recommended_items:
        print(str(counter) + '- ' +  item_dikt[i])
        counter+=1
    return recommended_items

# Se crean los diccionarios de usuario e item
user_dikt, item_dikt = user_item_dikts(user_item_matrix, books)

##### Recomendación a Usuario

              # modelo, matriz de interacción, id usuario, diccionario usuario y threshold
similar_recommendation(model, user_item_matrix, 254, user_dikt, item_dikt,threshold = 7)



##### Recomendación de un item a una lista de Usuarios
users_for_item(model, user_item_matrix, '0195153448', 10)


##### Items parecidos a uno en particular
item_embedings = item_emdedding_distance_matrix(model,user_item_matrix)
also_bought_recommendation(item_embedings,'B0000T6KHI' ,item_dikt)


