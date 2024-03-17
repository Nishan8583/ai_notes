# RAG
- I did not know this was called Retreival Augmented Generateion (RAG).
```
One of the most powerful applications enabled by LLMs is sophisticated question-answering (Q\&A) chatbots. These are applications that can answer questions about specific source information. These applications use a technique known as Retrieval Augmented Generation, or RAG.

```
- Idea is `use LLM like metas llama, and make it answers questions via a embeddings generated form a document.`
- Reference:
    - This github [code](https://github.com/AIAnytime/Llama2-Medical-Chatbot/blob/main/model.py) is helpful.
    - Youtube [video](https://youtu.be/kXuHxI5ZcG0?si=4euSa1K9dd8ZL6-b).
    - About embeddings [article](https://datasciencedojo.com/blog/embeddings-and-llm/#).
- About RAG application [information](https://python.langchain.com/docs/use_cases/question_answering/). This is such a good article. I love it.
- RAG architecture high-level overview:
**Indexing**: a pipeline for ingesting data from a source and indexing it. *This usually happens offline.*

**Retrieval and generation**: the actual RAG chain, which takes the user query at run time and retrieves the relevant data from the index, then passes that to the model.
- Looki into the [link](https://python.langchain.com/docs/use_cases/question_answering/) for more information.

# For cuda
1. check list of supported devices [here](https://developer.nvidia.com/cuda-gpus), under CUDA-Enabled Geforce and TITAN products.
2. Install [CUDA sdk](https://developer.nvidia.com/cuda-downloads).
3. Install [cuda for torch](https://pytorch.org/get-started/locally/).
4. To check if it worked i did this
```python
import torch
torch.cuda.device_count()
```