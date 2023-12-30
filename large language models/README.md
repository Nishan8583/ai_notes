# Large Language Models
- LLM inference, use trained langauge model, parameters file.
- facebook llama. https://github.com/karpathy/llama2.c/tree/master
- Trained using a lot of internet data.
- Training requires lot of data.
- PDF downloaded from https://drive.google.com/file/d/1pxx_ZI7O-Nwl7ZLNk5hI3WzAsTLwvNU7/view
- Youtube tutorial https://youtu.be/zjkBMFhNj_g?si=80kfy9FbEhnDIhEM .
- Neural network, next word prediction, predicting what the next word might be.
- The network (LLAMA) "dreams" internet documents, because trained on internet documents.
- Some of info may be memorized, some not.

# How does it work?
- Billions of parameters, we just adjust it, so that the network as a whole outputs what we want.
- `first` feed large data from internet, large data sets (pretrain). `quantity`. obtain `base model`.
- Continue training with the parameters obtained.`second`, training the assistant, swap data sets with made from people(like Q/A). `quality over quantity`. (fine tuning/alignment). obtain `assistant model`, `cheaper to do`.
- Base model of lama atleast, does not answer much (more questions it will ask), but meta has released assistant model which is good at answering.
![first_summary.png](./first_summary.png)
- optional stage 3, reinforcement learning from human feedbackL(RLHF), compare answers.
- More and more automated.

# LLM Scaling laws
- Performance of LLMs can be mostly predicted on the basis of N (number of parameters on network), D (amount of text we train on).
- DEMO, in the demo, it seems new model of GPT can use external tools, like browser, matplotlib code in python and get graphs and so on.