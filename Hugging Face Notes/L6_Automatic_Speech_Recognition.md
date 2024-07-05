# Lesson 6: Automatic Speech Recognition

- In the classroom, the libraries are already installed for you.
- If you would like to run this code on your own machine, you can install the following:
``` 
    !pip install transformers
    !pip install -U datasets
    !pip install soundfile
    !pip install librosa
    !pip install gradio
```

The `librosa` library may need to have [ffmpeg](https://www.ffmpeg.org/download.html) installed. 
- This page on [librosa](https://pypi.org/project/librosa/) provides installation instructions for ffmpeg.

- Here is some code that suppresses warning messages.


```python
from transformers.utils import logging
logging.set_verbosity_error()
```

### Data preparation


```python
from datasets import load_dataset
```


```python
dataset = load_dataset("librispeech_asr",
                       split="train.clean.100",
                       streaming=True,
                       trust_remote_code=True)
```


```python
example = next(iter(dataset))
```


```python
dataset_head = dataset.take(5)
list(dataset_head)
```


```python
list(dataset_head)[2]
```


```python
example
```


```python
from IPython.display import Audio as IPythonAudio

IPythonAudio(example["audio"]["array"],
             rate=example["audio"]["sampling_rate"])
```

### Build the pipeline


```python
from transformers import pipeline
```


```python
asr = pipeline(task="automatic-speech-recognition",
               model="./models/distil-whisper/distil-small.en")
```

Info about [distil-whisper/distil-small.en](https://huggingface.co/distil-whisper)


```python
asr.feature_extractor.sampling_rate
```


```python
example['audio']['sampling_rate']
```


```python
asr(example["audio"]["array"])
```


```python
example["text"]
```

### Build a shareable app with Gradio

### Troubleshooting Tip
- Note, in the classroom, you may see the code for creating the Gradio app run indefinitely.
  - This is specific to this classroom environment when it's serving many learners at once, and you won't wouldn't experience this issue if you run this code on your own machine.
- To fix this, please restart the kernel (Menu Kernel->Restart Kernel) and re-run the code in the lab from the beginning of the lesson.


```python
import os
import gradio as gr
```


```python
demo = gr.Blocks()
```


```python
def transcribe_speech(filepath):
    if filepath is None:
        gr.Warning("No audio found, please retry.")
        return ""
    output = asr(filepath)
    return output["text"]
```


```python
mic_transcribe = gr.Interface(
    fn=transcribe_speech,
    inputs=gr.Audio(sources="microphone",
                    type="filepath"),
    outputs=gr.Textbox(label="Transcription",
                       lines=3),
    allow_flagging="never")
```

To learn more about building apps with Gradio, you can check out the short course: [Building Generative AI Applications with Gradio](https://www.deeplearning.ai/short-courses/building-generative-ai-applications-with-gradio/), also taught by Hugging Face.


```python
file_transcribe = gr.Interface(
    fn=transcribe_speech,
    inputs=gr.Audio(sources="upload",
                    type="filepath"),
    outputs=gr.Textbox(label="Transcription",
                       lines=3),
    allow_flagging="never",
)
```


```python
with demo:
    gr.TabbedInterface(
        [mic_transcribe,
         file_transcribe],
        ["Transcribe Microphone",
         "Transcribe Audio File"],
    )

demo.launch(share=True, 
            server_port=int(os.environ['PORT1']))
```


```python
demo.close()
```

## Note: Please stop the demo before continuing with the rest of the lab.
- The app will continue running unless you run
  ```Python
  demo.close()
  ```
- If you run another gradio app (later in this lesson) without first closing this appp, you'll see an error message:
  ```Python
  OSError: Cannot find empty port in range
  ```

* Testing with a longer audio file


```python
import soundfile as sf
import io
```


```python
audio, sampling_rate = sf.read('narration_example.wav')
```


```python
sampling_rate
```


```python
asr.feature_extractor.sampling_rate
```


```python
asr(audio)
```

_Note:_ Running the cell above will return:
```
ValueError: We expect a single channel audio input for AutomaticSpeechRecognitionPipeline
```


* Convert the audio from stereo to mono (Using librosa)


```python
audio.shape
```


```python
import numpy as np

audio_transposed = np.transpose(audio)
```


```python
audio_transposed.shape
```


```python
import librosa
```


```python
audio_mono = librosa.to_mono(audio_transposed)
```


```python
IPythonAudio(audio_mono,
             rate=sampling_rate)
```


```python
asr(audio_mono)
```

_Warning:_ The cell above might throw a warning because the sample rate of the audio sample is not the same of the sample rate of the model.

Let's check and fix this!


```python
sampling_rate
```


```python
asr.feature_extractor.sampling_rate
```


```python
audio_16KHz = librosa.resample(audio_mono,
                               orig_sr=sampling_rate,
                               target_sr=16000)
```


```python
asr(
    audio_16KHz,
    chunk_length_s=30, # 30 seconds
    batch_size=4,
    return_timestamps=True,
)["chunks"]
```

* Build the Gradio interface.


```python
import gradio as gr
demo = gr.Blocks()
```


```python
def transcribe_long_form(filepath):
    if filepath is None:
        gr.Warning("No audio found, please retry.")
        return ""
    output = asr(
      filepath,
      max_new_tokens=256,
      chunk_length_s=30,
      batch_size=8,
    )
    return output["text"]
```


```python
mic_transcribe = gr.Interface(
    fn=transcribe_long_form,
    inputs=gr.Audio(sources="microphone",
                    type="filepath"),
    outputs=gr.Textbox(label="Transcription",
                       lines=3),
    allow_flagging="never")

file_transcribe = gr.Interface(
    fn=transcribe_long_form,
    inputs=gr.Audio(sources="upload",
                    type="filepath"),
    outputs=gr.Textbox(label="Transcription",
                       lines=3),
    allow_flagging="never",
)
```


```python
with demo:
    gr.TabbedInterface(
        [mic_transcribe,
         file_transcribe],
        ["Transcribe Microphone",
         "Transcribe Audio File"],
    )
demo.launch(share=True, 
            server_port=int(os.environ['PORT1']))
```


```python
demo.close()
```

## Note: Please stop the demo before continuing with the rest of the lab.
- The app will continue running unless you run
  ```Python
  demo.close()
  ```
- If you run another gradio app (later in this lesson) without first closing this appp, you'll see an error message:
  ```Python
  OSError: Cannot find empty port in range
  ```

### Try it yourself!
- Try this model with your own audio files!


```python
import soundfile as sf
import io

audio, sampling_rate = sf.read('narration_example.wav')
```


```python
sampling_rate
```


```python
asr.feature_extractor.sampling_rate
```
