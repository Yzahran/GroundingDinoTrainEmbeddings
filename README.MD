# Fine-Tuning Grounding DINO with Token Embeddings

Welcome to the **GDDemoTokens** repository! This project builds on the [GROUNDING DINO](https://github.com/open-mmlab/mmdetection/blob/main/configs/grounding_dino/README.md) model from the MMDetection framework. Here, we fine-tune Grounding DINO by adjusting only the class embeddings relevant to your dataset, rather than the weights of the entire model.

## Overview of Modifications

### Modified Files
To achieve this fine-tuning, we have made changes to the following files:

- **`GroundingDinoTrainEmbeddings/mmdetection/mmdet/models/detectors/grounding_dino.py`**:  
  This file has been modified to allow fine-tuning of only the class embeddings.
  
- **`GroundingDinoTrainEmbeddings/mmdetection/mmdet/models/language_models/bert.py`**:  
  Adjustments in this file enable the use of `[unused*]` tokens in the BERT model, which are employed for new class embeddings. This ensures that the original embeddings of the tokens are preserved and can be reused whenever needed.

- **`GroundingDinoTrainEmbeddings/mmdetection/mmdet/evaluation/metrics/nms_coco_metric.py`**:  
  This new file has been added to support evaluation with  NMS mAP during model evaluation.

### New Configuration Files
We've also added new configuration files to support custom fine-tuning:

- **`GroundingDinoTrainEmbeddings/configs/custom_gd.py`**:  
  Here, the classes of interest in your dataset are defined and mapped to corresponding new tokens. The new tokens are allocated using the `[unused*]` tokens of the BERT model, allowing for custom class embeddings while keeping the original token embeddings intact and available for reuse.

- **`GroundingDinoTrainEmbeddings/configs/train.py`** and **`GroundingDinoTrainEmbeddings/configs/test.py`**:  
  These scripts are provided for training and testing the model, respectively. Before running them, ensure that all paths are correctly set according to your specific dataset, which should be annotated in COCO format.

  > **Note**: To enable evaluation with NMS mAP, uncomment the relevant line in both `train.py` and `test.py`.

## Getting Started

### 1. Clone the Repository
First, clone this repository to your local machine:

```bash
git clone https://github.com/Yzahran/GroundingDinoTrainEmbeddings.git
cd GroundingDinoTrainEmbeddings
```
### (Optional) Download the Dataset
If you want to use our dataset generated using the CLEVR framework, you can download it from the following link:

[Download CLEVR Dataset](https://drive.google.com/file/d/1AyUxpOMcFGdTewBFigb4rVUgBifb7MfC/view?usp=sharing)


### 2. Set Up the Docker Environment
To make things easier, we've included a Docker setup:

1. Navigate to the Docker directory:
    ```bash
    cd docker
    ```

2. Modify the `docker-compose.yml` file to reflect the correct paths for your setup.

3. Build and start the Docker containers:
    ```bash
    make image
    make up
    ```

4. To check the running Docker processes, use:
    ```bash
    docker ps
    ```

5. Access the Docker container:
    ```bash
    docker exec -it <CONTAINER ID> /bin/bash
    cd /GroundingDinoTrainEmbeddings
    ```
>**Note**: Replace `<CONTAINER ID>` with the actual CONTAINER ID obtained from the `docker ps aux` command.

### 3. Training the Model
To train the model on a single GPU and ensure reproducibility by setting the random seed to 42, run:

```bash
bash mmdetection/tools/dist_train.sh configs/train.py 1 --cfg-options randomness.seed=42 randomness.deterministic=True
```

>  **Note**: If you want to perform validation with NMS mAP during training, make sure to uncomment the relevant line in `configs/train.py`.

### 4. Testing the Model
To test the model using the weights obtained during training and save the predictions to a `.pkl` file, execute:

```bash
bash mmdetection/tools/dist_test.sh configs/test.py path/to/weights 1 --out path/to/save/predictions
```

> **Note**: If you want to evaluate the model using NMS mAP during testing, make sure to uncomment the relevant line in `configs/test.py`.

## Important

- **Path Configurations**: Before running the training or testing scripts, ensure that all paths in `configs/train.py` and `configs/test.py` are appropriately configured for your dataset.
- **Annotation Format**: The dataset annotations must be in COCO format.
