from typing import Union
from pickle import dump
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.metrics import accuracy_score
from fastapi.middleware.cors import CORSMiddleware

draws = pd.read_csv("dataset.csv")

X = draws.iloc[:, [0, 1, 2, 3,4,5,6,7,8]].values
y = draws.iloc[:, [9]].values.ravel()

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, stratify=y)

## MLP
clf = MLPClassifier(solver='adam', hidden_layer_sizes=(40,), learning_rate_init=0.1, momentum=0.5)
clf.fit(X_train, y_train)

y_predict_mlp = clf.predict(X_test)

print("Acuracia MLP: ", accuracy_score(y_test, y_predict_mlp))
## Decision tree
clf2 = tree.DecisionTreeClassifier()
clf2 = clf2.fit(X_train, y_train)

y_predict_dt = clf2.predict(X_test)
print("Acuracia DT: ", accuracy_score(y_test, y_predict_dt))
## KNN
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, y_train)

y_predict_knn = neigh.predict(X_test)

print("Acuracia DT: ", accuracy_score(y_test, y_predict_knn))

## SVM
clf3 = svm.SVC()
clf3.fit(X, y)

y_predict_svm = clf3.predict(X_test)
print("Acuracia SVM: ", accuracy_score(y_test, y_predict_svm))

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permitir todas as origens. Alterar isso para uma lista específica é mais seguro.
    allow_credentials=True,
    allow_methods=["*"],  # Permitir todos os métodos HTTP (GET, POST, PUT, DELETE, etc.).
    allow_headers=["*"],  # Permitir todos os headers.
)

class Game(BaseModel):
    positions: list[str]
    ia: str

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/game")
def analyze_game(game: Game):
    list_to_predict = list(map(lambda x: x.replace('x', "1"), game.positions))
    list_to_predict = list(map(lambda x: x.replace('o', "-1"), list_to_predict))
    list_to_predict = list(map(lambda x: x.replace('b', "0"), list_to_predict)) 
    list_to_predict = [pd.to_numeric(x, errors='coerce') for x in list_to_predict]
    
    if(game.ia == 'svm'):
        predict = clf3.predict([list_to_predict]) 
    elif(game.ia == 'knn'):
        predict = neigh.predict([list_to_predict])
    elif(game.ia == 'dt'):
        predict = clf2.predict([list_to_predict])
    else:
        predict = clf.predict([list_to_predict])

    return {"status_game": predict[0]}
