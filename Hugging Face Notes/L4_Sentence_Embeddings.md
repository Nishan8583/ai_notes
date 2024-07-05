# Lesson 4: Sentence Embeddings

- In the classroom, the libraries are already installed for you.
- If you would like to run this code on your own machine, you can install the following:
``` 
    !pip install sentence-transformers
```

- Here is some code that suppresses warning messages.


```python
from transformers.utils import logging
logging.set_verbosity_error()
```

### Build the `sentence embedding` pipeline using 🤗 Transformers Library


```python
from sentence_transformers import SentenceTransformer
```


```python
model = SentenceTransformer("all-MiniLM-L6-v2")
```

More info on [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).


```python
sentences1 = ['The cat sits outside',
              'A man is playing guitar',
              'The movies are awesome']
```


```python
embeddings1 = model.encode(sentences1, convert_to_tensor=True)
```


```python
embeddings1
```


```python
sentences2 = ['The dog plays in the garden',
              'A woman watches TV',
              'The new movie is so great']
```


```python
embeddings2 = model.encode(sentences2, 
                           convert_to_tensor=True)
```


```python
print(embeddings2)
```

* Calculate the cosine similarity between two sentences as a measure of how similar they are to each other.


```python
from sentence_transformers import util
```


```python
cosine_scores = util.cos_sim(embeddings1,embeddings2)
```


```python
print(cosine_scores)
```


```python
for i in range(len(sentences1)):
    print("{} \t\t {} \t\t Score: {:.4f}".format(sentences1[i],
                                                 sentences2[i],
                                                 cosine_scores[i][i]))
```

### Try it yourself! 
- Try this model with your own sentences!
