import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
from sklearn.svm import LinearSVC
import pickle
from rb.core.document import Document
from rb.complexity.complexity_index import ComplexityIndex, compute_indices
from rb.cna.cna_graph import CnaGraph
from rb.core.lang import Lang
from rb.similarity.vector_model import (CorporaEnum, VectorModel,
                                        VectorModelType)
from rb.similarity.vector_model_factory import create_vector_model

A1train = pd.read_csv('/home/user/level text/A1/input/stats.csv')
A1=[]
for i in range(A1train.shape[0]):
    A1.append("A1")
A1train['level'] = A1
print(A1train.head())

A2train = pd.read_csv('/home/user/level text/A2/input/stats.csv')
A2=[]
for i in range(A2train.shape[0]):
    A2.append("A2")
A2train['level'] = A2
print(A2train.head())

A2plustrain = pd.read_csv('/home/user/level text/A2+/input/stats.csv')
A2plus=[]
for i in range(A2plustrain.shape[0]):
    A2plus.append("A2+")
A2plustrain['level'] = A2plus
print(A2plustrain.head())

B1train = pd.read_csv('/home/user/level text/B1/input/stats.csv')
B1=[]
for i in range(B1train.shape[0]):
    B1.append("B1")
B1train['level'] = B1
print(B1train.head())

B1plustrain = pd.read_csv('/home/user/level text/B1+/input/stats.csv')
B1plus=[]
for i in range(B1plustrain.shape[0]):
    B1plus.append("B1+")
B1plustrain['level'] = B1plus
print(B1plustrain.head())

B2train = pd.read_csv('/home/user/level text/B2/input/stats.csv')
B2=[]
for i in range(B2train.shape[0]):
    B2.append("B2")
B2train['level'] = B2
print(B2train.head())

B2plustrain = pd.read_csv('/home/user/level text/B2+/input/stats.csv')
B2plus=[]
for i in range(B2plustrain.shape[0]):
    B2plus.append("B2+")
B2plustrain['level'] = B2plus
print(B2plustrain.head())

C1train = pd.read_csv('/home/user/level text/C1/input/stats.csv')
C1=[]
for i in range(C1train.shape[0]):
    C1.append("C1")
C1train['level'] = C1
print(C1train.head())

C2train = pd.read_csv('/home/user/level text/C2/input/stats.csv')
C2=[]
for i in range(C2train.shape[0]):
    C2.append("C2")
C2train['level'] = C2
print(A1train.head())


train = pd.concat([A1train, A2train, A2plustrain, B1train, B1plustrain, B2train, B2plustrain, C1train, C2train], ignore_index=True)

train

xtrain=train.drop(columns=['filename','level'])#.astype(float)
ytrain=train['level']
#print("***** Train_Set *****")
#print(train.describe())
#print(train.columns.values)
#train.isna().head()
#train.fillna(train.mean(), inplace=True)
#train.info()
#X = np.array(train.drop(['filename'], 1).astype(float))
lsvc = LinearSVC()
lsvc.fit(xtrain, ytrain)
score = lsvc.score(xtrain, ytrain)
# save the model to disk

print("Score: ", score)
filename = 'lsvc.sav'
pickle.dump(lsvc, open(filename, 'wb'))

model = create_vector_model(Lang.DE, VectorModelType.from_str("word2vec"), "wikibooks")
text = "Aktuelle Entwicklungen in der Hochschulbildung den letzten drei Jahren lassen sich eine „Revolution“ darstellen, die nie zuvor in Geschichte der Hochschulbildung war. Diese Ereignisse vor allem beziehen sich auf das Internet und den Bildungsmarkt. Das Phänomen lässt sich als die „Demokratie der Hochschulbildung“ bezeichnen."
document = Document(lang=Lang.DE, text=text)
cna_graph = CnaGraph(docs=document, models=[model])
compute_indices(doc=document, cna_graph=cna_graph)
data = {}
for key, v in document.indices.items():
    data[repr(key)] = [v]


item = pd.DataFrame.from_dict(data)
print(item.head())
yitem = lsvc.predict(item)

print(yitem)

