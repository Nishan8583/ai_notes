# LangChain: Evaluation
- Evaluation the model and QA if we are correct or not.
- `langchain.debug` = True to see all the steps being carried out.
## Create our QandA application


```python
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.vectorstores import DocArrayInMemorySearch
```


```python
file = 'OutdoorClothingCatalog_1000.csv'
loader = CSVLoader(file_path=file)
data = loader.load()
```


```python
index = VectorstoreIndexCreator(
    vectorstore_cls=DocArrayInMemorySearch
).from_loaders([loader])
```


```python
llm = ChatOpenAI(temperature = 0.0, model=llm_model)
qa = RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=index.vectorstore.as_retriever(), 
    verbose=True,
    chain_type_kwargs = {
        "document_separator": "<<<<>>>>>"
    }
)
```

### Coming up with test datapoints


```python
data[10]
```


```python
data[11]
```

### Hard-coded examples


```python
examples = [
    {
        "query": "Do the Cozy Comfort Pullover Set\
        have side pockets?",
        "answer": "Yes"
    },
    {
        "query": "What collection is the Ultra-Lofty \
        850 Stretch Down Hooded Jacket from?",
        "answer": "The DownTek collection"
    }
]
```

### LLM-Generated examples


```python
from langchain.evaluation.qa import QAGenerateChain

```


```python
example_gen_chain = QAGenerateChain.from_llm(ChatOpenAI(model=llm_model))
```


```python
# the warning below can be safely ignored
```


```python
new_examples = example_gen_chain.apply_and_parse(
    [{"doc": t} for t in data[:5]]
)
```


```python
new_examples[0]
```


```python
data[0]
```

### Combine examples


```python
examples += new_examples
```


```python
qa.run(examples[0]["query"])
```

## Manual Evaluation


```python
import langchain
langchain.debug = True
```


```python
qa.run(examples[0]["query"])
```


```python
# Turn off the debug mode
langchain.debug = False
```

## LLM assisted evaluation


```python
predictions = qa.apply(examples)
```


```python
from langchain.evaluation.qa import QAEvalChain
```


```python
llm = ChatOpenAI(temperature=0, model=llm_model)
eval_chain = QAEvalChain.from_llm(llm)
```


```python
graded_outputs = eval_chain.evaluate(examples, predictions)
```


```python
for i, eg in enumerate(examples):
    print(f"Example {i}:")
    print("Question: " + predictions[i]['query'])
    print("Real Answer: " + predictions[i]['answer'])
    print("Predicted Answer: " + predictions[i]['result'])
    print("Predicted Grade: " + graded_outputs[i]['text'])
    print()
```


```python
graded_outputs[0]
```

## LangChain evaluation platform

The LangChain evaluation platform, LangChain Plus, can be accessed here https://www.langchain.plus/.  
Use the invite code `lang_learners_2023`

