from nltk.corpus import wordnet
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from sklearn import tree

# Paso 1: Recopilación y preparación de los datos
print ("Cargando datos...")
dataset = pd.read_csv("../Dataset/BERT_sentiment_IMDB_Dataset.csv")
print("Matriz de dataset:")
print(dataset.head())
documents = dataset["review"]
labels = dataset["sentiment"]

# Paso 2: Construcción del diccionario de palabras emocionales utilizando WordNet
print ("Construyendo diccionario de palabras emocionales...")
emotional_words = set()
for synset in wordnet.all_synsets():
    for lemma in synset.lemmas():
        if lemma.antonyms():
            emotional_words.add(lemma.name())

# Paso 3: Construcción de una matriz de documento de palabras utilizando TF-IDF
print ("Construyendo matriz de documento de palabras...")
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(documents)
print("Matriz de documento de palabras:")
print(pd.DataFrame(X[:3, :].toarray()))

# Paso 4: Función de peso (no se implementa en este código)

# Paso 5 y 6: SVD (Descomposición de valores singulares) y Reducción utilizando LSA
print("Realizando reducción de dimensionalidad...")
lsa = TruncatedSVD(n_components=100)  # Aumenta el número de componentes principales
X_lsa = lsa.fit_transform(X)

# Muestra las primeras tres filas de las matrices U, Σ y V^T
print("Matriz U:")
print(pd.DataFrame(lsa.components_[:3, :]))
print("\nMatriz Σ:")
print(pd.DataFrame(lsa.singular_values_[:3]))
print("\nMatriz V^T:")
print(pd.DataFrame(lsa.transform(X).T[:3, :]))

# Muestra las primeras tres filas de la matriz X_lsa
print("Matriz X_lsa:")
print(pd.DataFrame(X_lsa[:3, :]))

# Paso 7: División de los datos en conjuntos de entrenamiento y prueba
print("Dividiendo los datos en conjuntos de entrenamiento y prueba...")
X_train, X_test, y_train, y_test = train_test_split(X_lsa, labels, test_size=0.2, random_state=42)

# Paso adicional: Conversión de etiquetas de clase a valores numéricos
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)

# Paso 8: Ajuste de hiperparámetros del Random Forest
print ("Ajustando hiperparámetros del Random Forest...")
random_forest = RandomForestClassifier(max_depth=10, min_samples_split=5)
random_forest.fit(X_train, y_train)
# Obtén el primer árbol de decisión del modelo de Random Forest
primer_arbol = random_forest.estimators_[0]

# Visualiza el árbol de decisión
fig, ax = plt.subplots(figsize=(12, 12))
tree.plot_tree(primer_arbol, ax=ax)
plt.show()

# Paso 9: Predicción en el conjunto de entrenamiento y cálculo de la pérdida y precisión
y_train_pred = random_forest.predict(X_train)
train_loss = log_loss(y_train, y_train_pred)
train_accuracy = accuracy_score(y_train, y_train_pred)

# Paso 10: Predicción en el conjunto de prueba y cálculo de la pérdida y precisión
y_test_pred = random_forest.predict(X_test)
test_loss = log_loss(y_test, y_test_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

# Paso 11: Interpretación y análisis de los resultados
print("\nPérdida en el conjunto de entrenamiento:", train_loss)
print("Precisión en el conjunto de entrenamiento:", train_accuracy)
print("\nPérdida en el conjunto de prueba:", test_loss)
print("Precisión en el conjunto de prueba:", test_accuracy)


def classifySentiment(text):
    # Paso 1: Preprocesamiento del texto de entrada
    processed_text = vectorizer.transform([text])

    # Paso 2: Reducción de dimensionalidad utilizando SVD (LSA)
    text_lsa = lsa.transform(processed_text)

    # Paso 3: Clasificación de polaridad utilizando el modelo de Random Forest
    sentiment_label = random_forest.predict(text_lsa)
    sentiment = label_encoder.inverse_transform(sentiment_label)[0]

    # Paso 4: Devolución de la clasificación de polaridad
    return sentiment


review_text = "Avengers: Infinity War at least had the good taste to abstain from Jeremy Renner. No such luck in Endgame."
sentiment = classifySentiment(review_text)
print("\nTexto:", review_text)
print("Sentiment:", sentiment)
