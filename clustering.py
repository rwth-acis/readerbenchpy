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
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.metrics import accuracy_score

A1level = pd.read_csv('/home/user/textData/A1/stats.csv')
A1=[]
for i in range(A1level.shape[0]):
    A1.append("A")
A1level['level'] = A1
print(A1level.head())

A2level = pd.read_csv('/home/user/textData/A2/stats.csv')
A2=[]
for i in range(A2level.shape[0]):
    A2.append("A")
A2level['level'] = A2
print(A2level.head())



B1level = pd.read_csv('/home/user/textData/B1/stats.csv')
B1=[]
for i in range(B1level.shape[0]):
    B1.append("B")
B1level['level'] = B1
print(B1level.head())



B2level = pd.read_csv('/home/user/textData/B2/stats.csv')
B2=[]
for i in range(B2level.shape[0]):
    B2.append("B")
B2level['level'] = B2
print(B2level.head())



Clevel = pd.read_csv('/home/user/textData/C/stats.csv')
C=[]
for i in range(Clevel.shape[0]):
    C.append("C")
Clevel['level'] = C
print(Clevel.head())




data = pd.concat([A1level, A2level, B1level, B2level, Clevel], ignore_index=True)

data

x=data.drop(columns=['filename','level'])#.astype(float)
y=data['level']
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.15)
print (X_train.shape, Y_train.shape)
print (X_test.shape, Y_test.shape)
#print("***** Train_Set *****")
#print(train.describe())
#print(train.columns.values)
#train.isna().head()
#train.fillna(train.mean(), inplace=True)
#train.info()
#X = np.array(train.drop(['filename'], 1).astype(float))
lsvc = LinearSVC()
#lsvc = svm.SVC(kernel='rbf')
lsvc.fit(X_train, Y_train)
score = lsvc.score(X_test, Y_test)
# save the model to disk

print("Score: ", score)
predictions = lsvc.predict(X_test)
...
# make predictions
yhat = lsvc.predict(X_test)
# evaluate predictions
acc = accuracy_score(Y_test, yhat)
print('Accuracy: %.3f' % acc)
filename = 'lsvc.sav'
pickle.dump(lsvc, open(filename, 'wb'))

model = create_vector_model(Lang.DE, VectorModelType.from_str("word2vec"), "wiki")
text = "Aktuelle Entwicklungen in der Hochschulbildung den letzten drei Jahren lassen sich eine „Revolution“ darstellen, die nie zuvor in Geschichte der Hochschulbildung war. Diese Ereignisse vor allem beziehen sich auf das Internet und den Bildungsmarkt. Das Phänomen lässt sich als die „Demokratie der Hochschulbildung“ bezeichnen."
document = Document(lang=Lang.DE, text=text)
cna_graph = CnaGraph(docs=document, models=[model])
compute_indices(doc=document, cna_graph=cna_graph)
data = {}
for key, v in document.indices.items():
    data[repr(key)] = [v]


#item = pd.DataFrame.from_dict(data)
#print(item.head())
#yitem = lsvc.predict(item)

#print(yitem)

