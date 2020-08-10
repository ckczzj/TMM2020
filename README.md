# Temporal Textual Localization in Video via Adversarial Bi-Directional Interaction Networks
code for ***Temporal Textual Localization in Video via Adversarial Bi-Directional Interaction Networks***，which is accepted by IEEE Transactions on Multimedia 2020.

[paper](https://google.com/ncr)



## Prerequisites

- Python3
- PyTorch 1.5+
- gensim
- nltk



## Run

1. place [glove.bin](http://nlp.stanford.edu/data/glove.840B.300d.zip) in ./data folder

2. place ActivityNet video features in ./data/ActivityNet/feature folder, which is h5 file

3. place TACoS video features in ./data/TACoS/feature folder, which is npy file

4. (optional) modify json file in config folder for experimental setting

5. `python main.py`