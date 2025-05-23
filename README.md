# 🌱 Soil Image Classification - Part 1
This repository contains the complete implementation for the **Soil Image Classification Challenge (Part 1)** organized by **Annam.ai** at **IIT Ropar**. The objective is to classify each soil image into one of four categories: **Alluvial**, **Black**, **Clay**, or **Red** soil types using deep learning models.

## Team Members
- **Yashodip More** – Electrical Engineering, RCPIT, Maharashtra – yashodipmore2004@gmail.com  
- **S.M. Sakthivel** – AI & Data Science, Achariya College, Puducherry – s.m.sakthivelofficial@gmail.com  
- **Komal Kumavat** – Electrical Engineering, RCPIT, Maharashtra – komalkumavat025@gmail.com

---

##  Overview

Soil plays a crucial role in agriculture and the environment. In this challenge, machine learning and computer vision techniques are used to build a model that can classify images into one of the four soil types:

- Alluvial soil
- Black soil
- Clay soil
- Red soil

The final model is evaluated using the **minimum F1-score** across all soil types to encourage balanced model performance.

---

##  Problem Statement

Given a dataset of soil images with labels and a test set with unlabeled images, the task is to train a model that can classify new soil images into one of the four types. The images differ in resolution and size, and preprocessing is essential for consistent performance.

---

##  Repository Structure

```
soil-image-classification/
│
├── dataset/
│   ├── train/
│   │   ├── 0001.jpg
│   │   ├── 0002.jpg
│   │   └── ...
│   ├── test/
│   │   ├── 1001.jpg
│   │   ├── 1002.jpg
│   │   └── ...
│── train_labels.csv
│── test_ids.csv
│──submission.csv
│
├── notebook/
│   └── Soil Prediction Model Final.ipynb
│
├── requirements.txt
└── README.md
```

---

##  Setup Instructions

###  1. Clone the Repository

```bash
git clone https://github.com/<your-username>/soil-image-classification.git
cd soil-image-classification
```

###  2. Install Dependencies

You can install all the required Python libraries using:

```bash
pip install -r requirements.txt
```

#### Requirements
```txt
numpy
pandas
matplotlib
scikit-learn
opencv-python
seaborn
tqdm
torch
torchvision
Pillow
```

###  3. Dataset Setup

- Download the official dataset from the competition page.
- Create a `dataset/` directory and place the following inside:

```
dataset/
├── train/
├── test/
├── train_labels.csv
├── test_ids.csv
└── sample_submission.csv
```

---

##  Running the Model

### On Local GPU (e.g. NVIDIA RTX)

1. Ensure you have `CUDA` installed with PyTorch.
2. Open the notebook in Jupyter or VSCode:

```bash
jupyter notebook notebook/Soil\ Prediction\ Model\ Final.ipynb
```

3. Run all cells in order.

---

### On Kaggle

1. Upload this repository folder as a dataset.
2. Open a new Kaggle Notebook and **attach the uploaded dataset**.
3. Copy the notebook content into your Kaggle Notebook.
4. Run all cells.

---

##  Model Logic & Approach

###  Data Preprocessing

- Images resized to `224x224` pixels.
- Normalization using ImageNet mean and std.
- Data Augmentation:
  - Random horizontal flip
  - Random rotation
  - Color jitter

###  Model Architecture

- A pre-trained CNN model (e.g., ResNet18 or EfficientNet) is fine-tuned on our dataset.
- The final layer is adapted to classify into 4 classes.
- Loss: CrossEntropyLoss
- Optimizer: Adam or SGD
- Scheduler: StepLR for learning rate decay

###  Training & Validation

- Dataset split into 80% train and 20% validation.
- Model is trained for multiple epochs with checkpointing and early stopping.
- Performance is monitored via F1-score and accuracy.

###  Evaluation Metric

The primary evaluation metric is:

**Minimum per-class F1-score**:
```python
final_score = min(f1_class1, f1_class2, f1_class3, f1_class4)
```

This encourages performance consistency across all soil types.

---

##  Submission Format

Prepare a CSV with predictions:

```csv
image_id,soil_type
1001.jpg,Clay soil
1002.jpg,Red soil
...
```

Ensure column names and formats match the provided `sample_submission.csv`.

---

##  Team

- **Team Name:** Sarthak
- **Members:**
  - Komal Rajesh Kumavat
  - More Yashodip
  - Girase Jaykumar
- **College:** R. C. Patel Institute of Technology, Shirpur

---

##  License

This project is licensed under the **MIT License**.

---

##  Acknowledgements

Thanks to [Annam.ai](https://annam.ai/) and [IIT Ropar](https://www.iitrpr.ac.in/) for organizing this valuable challenge and providing the dataset.
