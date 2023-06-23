import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.corpus import wordnet

# Paso 1: Recopilación y preparación de los datos
dataset = pd.read_csv("../Dataset/BERT_sentiment_IMDB_Dataset.csv")
documents = dataset["review"]
labels = dataset["sentiment"]

# Paso 2: Construcción del diccionario de palabras emocionales utilizando WordNet
emotional_words = set()
for synset in wordnet.all_synsets():
    for lemma in synset.lemmas():
        if lemma.antonyms():
            emotional_words.add(lemma.name())

# Paso 3: Construcción de una matriz de documento de palabras utilizando TF-IDF
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(documents)

# Paso 4: Función de peso (no se implementa en este código)

# Paso 5 y 6: SVD (Descomposición de valores singulares) y Reducción utilizando LSA
lsa = TruncatedSVD(n_components=2)
X_lsa = lsa.fit_transform(X)

# Paso 7: Cálculo de la puntuación emocional del texto utilizando VADER
analyzer = SentimentIntensityAnalyzer()


def classifySentiment(text):
    sentiment = analyzer.polarity_scores(text)["compound"]
    if sentiment >= 0:
        label = "positivo"
    else:
        label = "negativo"
    return text, label


# Paso 8: Interpretación y análisis de los resultados
review_text = ("Avengers: Infinity War at least had the good taste to abstain from Jeremy Renner. No such luck in Endgame.")
text_representation, predicted_label = classifySentiment(review_text)

print("\nMatriz término-documento reducida con LSA:")
print(X_lsa)
print("\n Texto: ",review_text)
print("\nSentimiento del texto:", predicted_label)
