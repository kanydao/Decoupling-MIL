# Decoupling MIL Transformer-based Network for Weakly Supervised Polyp Detection

Contributed by Hantao Zhang, Risheng Xie, Shouhong wan, and peiquan jin

![](paper_images/framework.png)

## Installation

To train and test our model on your computer, please ensure that you have PyTorch version 1.7.0 installed, and your Python version should be 3.7.11.

Installl the dependencies.

```
pip install -r requirements.txt
```



## Data preparation

### Dataset

The dataset consists of 517 videos. Specifically, we have gathered 72 normal videos without polyps and 368 abnormal videos with polyps for training purposes. Additionally, 17 normal videos and 60 abnormal videos are reserved for testing. The detailed configuration of the dataset will be made publicly available upon acceptance of the paper.



## Training

### Step 1: Train the decoupling model

```shell
python main_transformer.py
```



### Step 2: Fine-tune the  convolutional adapters

```shell
python main_adapter.py
```

## License

HOD is released under the MIT license. See [LICENSE](./LICENSE) for additional details.

