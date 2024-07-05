# Lesson 3: Translation and Summarization

- In the classroom, the libraries are already installed for you.
- If you would like to run this code on your own machine, you can install the following:

```
    !pip install transformers 
    !pip install torch
```

- Here is some code that suppresses warning messages.


```python
from transformers.utils import logging
logging.set_verbosity_error()
```

### Build the `translation` pipeline using 🤗 Transformers Library


```python
from transformers import pipeline 
import torch
```


```python
translator = pipeline(task="translation",
                      model="./models/facebook/nllb-200-distilled-600M",
                      torch_dtype=torch.bfloat16) 
```

NLLB: No Language Left Behind: ['nllb-200-distilled-600M'](https://huggingface.co/facebook/nllb-200-distilled-600M).




```python
text = """\
My puppy is adorable, \
Your kitten is cute.
Her panda is friendly.
His llama is thoughtful. \
We all have nice pets!"""
```


```python
text_translated = translator(text,
                             src_lang="eng_Latn",
                             tgt_lang="fra_Latn")
```

To choose other languages, you can find the other language codes on the page: [Languages in FLORES-200](https://github.com/facebookresearch/flores/blob/main/flores200/README.md#languages-in-flores-200)

For example:
- Afrikaans: afr_Latn
- Chinese: zho_Hans
- Egyptian Arabic: arz_Arab
- French: fra_Latn
- German: deu_Latn
- Greek: ell_Grek
- Hindi: hin_Deva
- Indonesian: ind_Latn
- Italian: ita_Latn
- Japanese: jpn_Jpan
- Korean: kor_Hang
- Persian: pes_Arab
- Portuguese: por_Latn
- Russian: rus_Cyrl
- Spanish: spa_Latn
- Swahili: swh_Latn
- Thai: tha_Thai
- Turkish: tur_Latn
- Vietnamese: vie_Latn
- Zulu: zul_Latn


```python
text_translated
```

## Free up some memory before continuing
- In order to have enough free memory to run the rest of the code, please run the following to free up memory on the machine.


```python
import gc
```


```python
del translator
```


```python
gc.collect()
```

### Build the `summarization` pipeline using 🤗 Transformers Library


```python
summarizer = pipeline(task="summarization",
                      model="./models/facebook/bart-large-cnn",
                      torch_dtype=torch.bfloat16)
```

Model info: ['bart-large-cnn'](https://huggingface.co/facebook/bart-large-cnn)


```python
text = """Paris is the capital and most populous city of France, with
          an estimated population of 2,175,601 residents as of 2018,
          in an area of more than 105 square kilometres (41 square
          miles). The City of Paris is the centre and seat of
          government of the region and province of Île-de-France, or
          Paris Region, which has an estimated population of
          12,174,880, or about 18 percent of the population of France
          as of 2017."""
```


```python
summary = summarizer(text,
                     min_length=10,
                     max_length=100)
```


```python
summary
```

### Try it yourself! 
- Try this model with your own texts!


```python

```
