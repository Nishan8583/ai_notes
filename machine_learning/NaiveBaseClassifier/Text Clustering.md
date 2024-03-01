# Text Clustering
- why? end up with a model that can classify documents and texts.
## Tokenization
- Convert words into numbers that usable by ML.
- One technique is TF-IDF (Term frequency - Inverse document frequency)
- term frequency ff(w) = (number of times words appread in doc / Total number of words in doc)
    - inverse document frequency idf(w) = log(number of documents/number of documents that contain the word )
    - In python: TfidVectorizer() // do id-idf, and CountVectorizer()  // count tokens
- Occurence of certain words map to certain cluster.
- The code attached is complete copy of https://actalent.udemy.com/course/introduction-to-machine-learning-in-python/learn/lecture/29408122#overview 