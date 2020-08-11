# Temporal Textual Localization in Video via Adversarial Bi-Directional Interaction Networks
code for ***Temporal Textual Localization in Video via Adversarial Bi-Directional Interaction Networks***，which is accepted by IEEE Transactions on Multimedia 2020.

[paper](https://google.com/ncr)



## Prerequisites

- Python3
- PyTorch 1.5+
- gensim
- nltk
- fairseq



## Run

1. download and unzip [glove.840B.300d.zip](http://nlp.stanford.edu/data/glove.840B.300d.zip) and place it at ./data/glove.bin

3. download and unzip [TACoS_feature.zip](https://drive.google.com/file/d/1B0blGPXmmgyDNtmdxeSNoT9Gcw7t6tGl/view?usp=sharing) and place it at ./data/TACoS/feature/

4. (optional) modify json file in config folder for experimental setting

5. `python main.py --config-path ./config/TACoS.json`



Because of huge size of ActivityNet video features, we can't place it on Google Drive. You can download the videos [here](http://activity-net.org/download.html) and extract their c3d feature using ./data/c3d.py