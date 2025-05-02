# CS163-Senior-Data-Science-Project


# Image Captioning with Attention

This repository implements an end-to-end image captioning pipeline using the "Show, Attend, and Tell" architecture. The system generates natural language captions for images by combining deep convolutional neural networks, an attention mechanism, and recurrent neural networks.

Website: [Project Demo & Documentation](https://dsprojectwebsite.el.r.appspot.com/)

---
SAMPLES

![image](https://github.com/user-attachments/assets/9f199180-6601-4c28-ae67-d35867d003de)
![image](https://github.com/user-attachments/assets/42aca738-1bb0-4d99-a641-d252983135c7)
![image](https://github.com/user-attachments/assets/52063acd-9d79-4489-9d5e-858ba1e17997)
![image](https://github.com/user-attachments/assets/5f848e7e-f699-4300-bcab-8fcfed709226)
![image](https://github.com/user-attachments/assets/09922a56-091a-4859-ac9d-5d883c67922d)
![image](https://github.com/user-attachments/assets/af036e27-f580-4ad0-b30c-ed2a7740c55a)

smooth_fn = SmoothingFunction().method1

# Build dicts for COCOEvalCap
gts = {}   # idx -> list of GT strings
res = {}   # idx -> [pred string]
best_bleus = []

for idx, img_id in enumerate(selected):
    # 1) gather refs & hyp
    ann_ids = coco.getAnnIds(imgIds=img_id)
    anns    = coco.loadAnns(ann_ids)
    refs    = [ann['caption'] for ann in anns]
    hyp_words = batch[idx][1]
    hyp_str   = ' '.join(hyp_words)

    # 2) fill for evaluators
    gts[idx] = refs
    res[idx] = [hyp_str]

    # 3) best-BLEU among all references
    per_ref = [
        sentence_bleu([r.split()], hyp_str.split(), smoothing_function=smooth_fn)
        for r in refs
    ]
    best_bleus.append(max(per_ref))

# 4) compute CIDEr & SPICE
cider_scorer = Cider()
spice_scorer = Spice()

_, cider_scores = cider_scorer.compute_score(gts, res)


# 5) display each image + captions + metrics
for idx, img_id in enumerate(selected):
    pil_img, pred_words = batch[idx]
    refs    = gts[idx]
    hyp_str = res[idx][0]
    bleu    = best_bleus[idx]
    cider   = cider_scores[idx]
    

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(pil_img)
    ax.axis('off')

    # Multi‐line title: generated, GTs, then metrics
    gt_display = ' | '.join(refs[:3])
    title = (
        f"Generated: {hyp_str}\n"
        f"GTs: {gt_display}\n"
        f"BLEU: {bleu:.4f}   CIDEr: {cider:.4f}"
    )
    ax.set_title(title, wrap=True, fontsize=10)
    plt.tight_layout()
    plt.show()














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
