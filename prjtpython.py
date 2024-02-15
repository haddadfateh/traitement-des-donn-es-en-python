import pandas as pd
import numpy as np
import matplotlib.pyplot as plt




# Chargement des donnÃ©es pour train_data et x_test
data_train = pd.read_csv('mobile_train.csv')
X_test = pd.read_csv('mobile_test_data.csv')


# # SÃ©paration des donnÃ©es
# Donnees_train = data_train.sample(frac=0.8, random_state=96)
# Donnees_valid = data_train.drop(Donnees_train.index)
#
# X_train = Donnees_train[Donnees_train.columns[:-1]]
# Y_train = Donnees_train[Donnees_train.columns[-1]]
#
# X_valid = Donnees_valid[Donnees_valid.columns[:-1]]
# Y_valid = Donnees_valid[Donnees_valid.columns[-1]]


# SÃ©paration des donnÃ©es

df_train = data_train.sample(frac=0.8, random_state=64)
df_valid = data_train.drop(df_train.index)


X_train = df_train.iloc[:, :-1].values
Y_train = df_train.iloc[:, -1].values
X_valid = df_valid.iloc[:, :-1].values
Y_valid = df_valid.iloc[:, -1].values


X_test = X_test.values



def euclidean_distance(v1, v2):
    return np.sqrt(np.sum(np.square(v1 - v2)))

# def euclidean_distance(v1, v2):
#     distance = 0
#     for i in range(v1.shape[0]):
#         distance += (v1[i] - v2[i])**2
#     return np.sqrt(distance)


# Recherche des k plus proches voisins
def neighbors(X_train, y_label, x_test, k):
    list_distances = []
    for i in range(X_train.shape[0]):
        la_distence = euclidean_distance(X_train[i], x_test)
        list_distances.append(la_distence)
    df = pd.DataFrame()
    df["label"] = y_label
    df["distance"] = list_distances
    df = df.sort_values(by="distance")
    return df.iloc[:k, :]
# def neighbors(X_train, y_label, x_test, k):
#     list_distances = []
#     for i in range(len(X_train)):
#         dist = euclidean_distance(X_train.iloc[i].values, x_test)
#         list_distances.append(dist)
#     df = pd.DataFrame()
#     df["label"] = y_label
#     df["distance"] = list_distances
#     df = df.sort_values(by="distance")
#     return df.iloc[:k,:]

#PrÃ©diction du label de l'Ã©chantillon test
def prediction(neighbors):
    counts = neighbors['label'].value_counts()
    return counts.index[0]
# # PrÃ©diction du label de l'Ã©chantillon test
# def prediction(neighbors):
#     counts = neighbors['label'].value_counts()
#     return counts.index[0]


## Evaluation de la performance du modÃ¨le
def evaluation(X_train, Y_train, X_valid, Y_valid, k, verbose=True):
    vrai = 0  # resultat correct (juste)
    faux = 0  # resultat       faux
    liste_preds = []
    compteur = 0
    for i in range(X_valid.shape[0]):
        nearest_neighbors = neighbors(X_train, Y_train, X_valid[i], k)
        mon_predict = prediction(nearest_neighbors)
        liste_preds.append(mon_predict)
        if mon_predict == Y_valid[i]:
            vrai += 1
        else:
            faux += 1
            if verbose:
                print("prediction fausse: ")
                print("->> la Prediction : ->>", prediction(nearest_neighbors))
                print("le resultat que normalement on doit trouver cest  : ", Y_valid[i])
        compteur += 1
    accuracy = vrai / compteur
    if verbose:
        print("-->accuracy:-->", accuracy)
    return accuracy
# def evaluation(X_train, Y_train, X_valid, Y_valid, k, verbose=True):
#     TP = 0 # Vrai
#     FP = 0 # Faux
#     list_pred = []
#     total = 0
#     for i in range(X_valid.shape[0]):
#         nearest_neighbors = neighbors(X_train, Y_train, X_valid.iloc[i].values, k)
#         pred = prediction(nearest_neighbors)
#         list_pred.append(pred)
#         if pred == Y_valid.iloc[i]:
#             TP += 1
#         else:
#             FP += 1
#             if verbose:
#                 print("Erreur de prediction : ")
#                 print("Prediction : ", prediction(nearest_neighbors))
#                 print("Resultat attendu : ", Y_valid.iloc[i])
#         total += 1
#     accuracy = TP / total
#     if verbose:
#         print("Accuracy:", accuracy)
#     return accuracy

## Test de classification sur les donnÃ©es de test
def test(X_train, Y_train, X_test, k):
    liste_predictions = []
    for i in range(X_test.shape[0]):
        nearest_neighbors = neighbors(X_train, Y_train, X_test[i], k)
        pred = prediction(nearest_neighbors)
        liste_predictions.append(pred)
        #print("-->Prediction -->: ", pred, "<--")
    return liste_predictions
# def test(X_train, Y_train, X_test, k):
#     list_pred = []
#     for i in range(X_test.shape[0]):
#         nearest_neighbors = neighbors(X_train, Y_train, X_test.iloc[i].values, k)
#         pred = prediction(nearest_neighbors)
#         list_pred.append(pred)
#         print("Prediction : ", pred)
#     return list_pred

# EntraÃ®nement du modÃ¨le
liste_predictions = test(X_train, Y_train, X_test, 10)
# Sauvegarde des prÃ©dictions dans un fichier CSV
A = np.array(liste_predictions)
np.savetxt('mobile_test_predictions.csv', A)

# Recherche du meilleur k par validation croisÃ©e
k_values = range(1, 20, 2)
accuracy_values = []

for k in k_values:
    accuracy = evaluation(X_train, Y_train, X_valid, Y_valid, k, verbose=False)
    accuracy_values.append(accuracy)

plt.plot(k_values, accuracy_values)
plt.xlabel('Valeur de k')
plt.ylabel('PrÃ©cision')
plt.title('Performance du modÃ¨le en fonction de k')
plt.show()
#partie 2 reseau neurone
import pandas as pd
# # Chargement des donnÃ©es pour train_data et x_test
data_train = pd.read_csv('mobile_train.csv')
X_test = pd.read_csv('mobile_test_data.csv')
#
# SÃ©paration des donnÃ©es
df_train = data_train.sample(frac=0.8, random_state=64)
df_valid = data_train.drop(df_train.index)

X_train = df_train.iloc[:, :-1].values
Y_train = df_train.iloc[:, -1].values
X_valid = df_valid.iloc[:, :-1].values
Y_valid = df_valid.iloc[:, -1].values

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Fonction d'Ã©valuation pour le modÃ¨le K plus proches voisins
def evaluate_knn_model(X_train, Y_train, X_valid, Y_valid, n_neighbors):
    accuracies = []  # Liste pour stocker les prÃ©cisions pour chaque nombre de voisins

    for n in n_neighbors:
        clf = KNeighborsClassifier(n_neighbors=n)
        clf.fit(X_train, Y_train)
        Y_pred = clf.predict(X_valid)
        accuracy = accuracy_score(Y_valid, Y_pred)
        accuracies.append(accuracy)

    return accuracies
#
# ParamÃ¨tres pour le modÃ¨le K plus proches voisins
n_neighbors = np.arange(1, 21)  # Nombre de voisins Ã  tester

# Appeler la fonction d'Ã©valuation pour obtenir les prÃ©cisions pour chaque nombre de voisins
accuracies = evaluate_knn_model(X_train, Y_train, X_valid, Y_valid, n_neighbors)


# Tracer le graphe de la performance en fonction du nombre de voisins
plt.plot(n_neighbors, accuracies)
plt.xlabel('Nombre de voisins')
plt.ylabel('PrÃ©cision')
plt.title('Performance du modÃ¨le K plus proches voisins')
plt.grid(True)
plt.show()




from sklearn.neural_network import MLPClassifier

# Initialize the MLPClassifier with appropriate parameters
clf = MLPClassifier(solver='adam', hidden_layer_sizes=(350,), alpha=1e-04, learning_rate='adaptive')

# Fit the classifier to the training data
clf.fit(X_train, Y_train)

def evaluation(X_train, Y_train, X_valid, Y_valid, k, nn, verbose=True):
    TP = 0 # True Positive
    FP = 0 # False Positive
    total = 0

    if nn == True:
        pred = clf.predict(X_valid)
        for i in range(X_valid.shape[0]):
            if pred[i] == Y_valid[i]:
                TP += 1
            else:
                print("Prediction : ", pred[i])
                print("Resultat attendu : ", Y_valid[i])
            total += 1
    else:
        for i in range(X_valid.shape[0]):
            # Define the prediction method for nn
            prediction = nn.predict(X_valid[i].reshape(1, -1))
            if prediction == Y_valid[i]:
                TP += 1
            else:
                FP += 1
                print("Prediction : ", prediction)
                print("Resultat attendu : ", Y_valid[i])
            total += 1
    accuracy = (TP) / total
    if verbose:
        print("Accuracy:" + str(accuracy))
    return accuracy
K=10
# Call the evaluation function
evaluation(X_train, Y_train, X_valid, Y_valid, K, clf, verbose=True)

# # EntraÃ®nement du modÃ¨le
# # Load the test data
# mobile_test_data = pd.read_csv("mobile_test_data.csv")
#
# # Make predictions on the test data
# pred = clf.predict(mobile_test_data)
#
# # Save the predictions to a CSV file
# np.savetxt('mobile_test_predictions.csv', pred, delimiter=',')