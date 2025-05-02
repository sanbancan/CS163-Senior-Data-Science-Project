# CS163-Senior-Data-Science-Project


# Image Captioning with Attention

This repository implements an end-to-end image captioning pipeline using the "Show, Attend, and Tell" architecture. The system generates natural language captions for images by combining deep convolutional neural networks, an attention mechanism, and recurrent neural networks.

Website: [Project Demo & Documentation](https://dsprojectwebsite.el.r.appspot.com/)

---
SAMPLES

![image](https://github.com/user-attachments/assets/9f199180-6601-4c28-ae67-d35867d003de)
![image](https://github.com/user-attachments/assets/42aca738-1bb0-4d99-a641-d252983135c7)
![image](https://github.com/user-attachments/assets/52063acd-9d79-4489-9d5e-858ba1e17997)
![image](https://github.com/user-attachments/assets/25b325c7-4b29-4a16-a8cd-eb688f86a31f)
![image](https://github.com/user-attachments/assets/5a25eb19-6168-47ec-817b-f3775e32cf67)
![image](https://github.com/user-attachments/assets/96fa72c4-c24c-49f1-8651-035a4c0fa0b6)
![image](https://github.com/user-attachments/assets/d9949528-5682-417b-be11-83af65a8fbd6)











## Summary

This project provides a complete workflow for generating descriptive captions for images using deep learning. It covers data preprocessing, model training, evaluation, and visualization of attention maps, enabling both research and practical applications in computer vision and natural language processing.

---

## Setup Instructions

### 1. Clone the Repository

git clone https://github.com/sanbancan/CS163-Senior-Data-Science-Project.git
cd CS163-Senior-Data-Science-Project


### 2. Create and Activate a Python Environment

We recommend using Python 3.8+ and [virtualenv](https://virtualenv.pypa.io/) or [conda](https://docs.conda.io/).

python -m venv venv
source venv/bin/activate # On Windows: venv\Scripts\activate


### 3. Install Required Packages

Install all dependencies with:

pip install -r requirements.txt


**Key Packages:**
- torch, torchvision
- numpy, h5py, pillow, imageio
- tqdm, nltk

### 4. Download Datasets

- Download the COCO dataset (or Flickr8k/Flickr30k) as described in the [data section](#data).
- Place images and annotation files in the appropriate directories as referenced in the code.

---

## Pipeline Overview

The project pipeline consists of the following stages:

1. **Data Preparation**
    - Use `create_input_files.py` to preprocess images and captions.
    - Generates HDF5 files for images, JSON files for encoded captions, and a word map.

2. **Model Training**
    - Train the encoder (CNN) and decoder (RNN with attention) using `train.py`.
    - Supports checkpointing, learning rate scheduling, and early stopping.
    - Model parameters, optimizer states, and BLEU-4 scores are saved for reproducibility.

3. **Evaluation & Inference**
    - Evaluate the trained model on validation/test sets.
    - Generate captions for new images using beam search (`caption_image_beam_search`).
    - Visualize attention overlays to interpret which image regions influenced each word.

4. **Analysis & Website Publication**
    - Analyze model performance using BLEU and other metrics.
    - Publish results and interactive demos on the project website: [https://dsprojectwebsite.el.r.appspot.com/](https://dsprojectwebsite.el.r.appspot.com/)

---

## Repository Structure

| Directory / File         | Purpose                                                                                      |
|------------------------- |---------------------------------------------------------------------------------------------|
| `create_input_files.py`  | Preprocesses raw images and captions, creates HDF5 and JSON files for model training.       |
| `train.py`               | Contains the training loop, validation, and checkpointing for the encoder-decoder model.    |
| `models.py`              | Defines the `Encoder` (CNN) and `DecoderWithAttention` (RNN with attention) architectures.  |
| `utils.py`               | Utility functions for data loading, metrics, embeddings, and checkpoint management.         |
| `datasets.py`            | Custom PyTorch dataset for loading image-caption pairs efficiently during training.         |
| `caption.py`             | Inference and visualization: generates captions and attention heatmaps for images.          |
| `requirements.txt`       | List of all required Python packages.                                                       |
| `README.md`              | Project documentation and setup instructions (this file).                                   |
| `data/`                  | Directory for processed data files (HDF5/JSON) and word maps.                              |
| `checkpoints/`           | Directory to store model checkpoints during training.                                       |

---

## Key Processing Code Locations

- **Data Preparation:** [`create_input_files.py`](create_input_files.py)
- **Model Definition:** [`models.py`](models.py)
- **Training & Validation:** [`train.py`](train.py)
- **Utilities:** [`utils.py`](utils.py)
- **Inference & Visualization:** [`caption.py`](caption.py)
- **Dataset Loader:** [`datasets.py`](datasets.py)


---

## References

- [Show, Attend and Tell: Neural Image Caption Generation with Visual Attention (Xu et al., 2015)](https://arxiv.org/abs/1502.03044)
- [COCO 2017 Dataset](https://cocodataset.org/#download)
