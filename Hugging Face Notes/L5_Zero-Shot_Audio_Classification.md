# Lesson 5: Zero-Shot Audio Classification

- In the classroom, the libraries have already been installed for you.
- If you are running this code on your own machine, please install the following:
``` 
    !pip install transformers
    !pip install datasets
    !pip install soundfile
    !pip install librosa
```

The `librosa` library may need to have [ffmpeg](https://www.ffmpeg.org/download.html) installed. 
- This page on [librosa](https://pypi.org/project/librosa/) provides installation instructions for ffmpeg.

- Here is some code that suppresses warning messages.


```python
from transformers.utils import logging
logging.set_verbosity_error()
```

### Prepare the dataset of audio recordings


```python
from datasets import load_dataset, load_from_disk

# This dataset is a collection of different sounds of 5 seconds
# dataset = load_dataset("ashraq/esc50",
#                       split="train[0:10]")
dataset = load_from_disk("./models/ashraq/esc50/train")
```


```python
audio_sample = dataset[0]
```


```python
audio_sample
```


```python
from IPython.display import Audio as IPythonAudio
IPythonAudio(audio_sample["audio"]["array"],
             rate=audio_sample["audio"]["sampling_rate"])
```

### Build the `audio classification` pipeline using 🤗 Transformers Library


```python
from transformers import pipeline
```


```python
zero_shot_classifier = pipeline(
    task="zero-shot-audio-classification",
    model="./models/laion/clap-htsat-unfused")
```

More info on [laion/clap-htsat-unfused](https://huggingface.co/laion/clap-htsat-unfused).

### Sampling Rate for Transformer Models
- How long does 1 second of high resolution audio (192,000 Hz) appear to the Whisper model (which is trained to expect audio files at 16,000 Hz)? 


```python
(1 * 192000) / 16000
```




    12.0



- The 1 second of high resolution audio appears to the model as if it is 12 seconds of audio.

- How about 5 seconds of audio?


```python
(5 * 192000) / 16000
```




    60.0



- 5 seconds of high resolution audio appears to the model as if it is 60 seconds of audio.


```python
zero_shot_classifier.feature_extractor.sampling_rate
```


```python
audio_sample["audio"]["sampling_rate"]
```

* Set the correct sampling rate for the input and the model.


```python
from datasets import Audio
```


```python
dataset = dataset.cast_column(
    "audio",
     Audio(sampling_rate=48_000))
```


```python
audio_sample = dataset[0]
```


```python
audio_sample
```


```python
candidate_labels = ["Sound of a dog",
                    "Sound of vacuum cleaner"]
```


```python
zero_shot_classifier(audio_sample["audio"]["array"],
                     candidate_labels=candidate_labels)
```


```python
candidate_labels = ["Sound of a child crying",
                    "Sound of vacuum cleaner",
                    "Sound of a bird singing",
                    "Sound of an airplane"]
```


```python
zero_shot_classifier(audio_sample["audio"]["array"],
                     candidate_labels=candidate_labels)
```

### Try it yourself! 
- Try this model with some other labels and audio files!
