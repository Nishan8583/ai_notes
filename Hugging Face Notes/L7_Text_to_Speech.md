# Lesson 7: Text to Speech

- In the classroom, the libraries are already installed for you.
- If you would like to run this code on your own machine, you can install the following:

```
    !pip install transformers
    !pip install gradio
    !pip install timm
    !pip install timm
    !pip install inflect
    !pip install phonemizer
    
```

**Note:**  `py-espeak-ng` is only available Linux operating systems.

To run locally in a Linux machine, follow these commands:
```
    sudo apt-get update
    sudo apt-get install espeak-ng
    pip install py-espeak-ng
```

### Build the `text-to-speech` pipeline using the 🤗 Transformers Library

- Here is some code that suppresses warning messages.


```python
from transformers.utils import logging

logging.set_verbosity_error()
```


```python
from transformers import pipeline

narrator = pipeline("text-to-speech",
                    model="./models/kakao-enterprise/vits-ljs")
```

Info about [kakao-enterprise/vits-ljs](https://huggingface.co/kakao-enterprise/vits-ljs)


```python
text = """
Researchers at the Allen Institute for AI, \
HuggingFace, Microsoft, the University of Washington, \
Carnegie Mellon University, and the Hebrew University of \
Jerusalem developed a tool that measures atmospheric \
carbon emitted by cloud servers while training machine \
learning models. After a model’s size, the biggest variables \
were the server’s location and time of day it was active.
"""
```


```python
narrated_text = narrator(text)
```


```python
from IPython.display import Audio as IPythonAudio

IPythonAudio(narrated_text["audio"][0],
             rate=narrated_text["sampling_rate"])
```

### Try it yourself! 
- Try this model with your own text to speech examples!
