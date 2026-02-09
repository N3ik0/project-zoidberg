<div align="center">
  <img src="assets/icons/zoidberg_logo.svg" width="100" height="100" alt="Zoidberg Logo">
  <h1>Project Zoidberg: Pneumonia Classification</h1>
</div>

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Academic%20Project-lightgrey)

**Project Zoidberg** is a Deep Learning project developed as part of the **Master 1 in Artificial Intelligence** curriculum.

Its main objective is to assist medical diagnosis by automatically detecting and classifying pneumonia from Chest X-Ray images. Unlike standard binary classifiers, this model distinguishes between **three clinical states**:
1.  **Normal** (Healthy)
2.  **Bacterial Pneumonia**
3.  **Viral Pneumonia**

## Context & Privacy

This project utilizes a dataset of pediatric chest X-rays.
> **⚠️ Data Privacy Notice:**
> Due to the sensitive nature of medical data and privacy regulations, the training dataset **is not included** in this repository. The code is provided for educational and architectural demonstration purposes.

To run this project, you would need to structure your own dataset as described in the [Usage](#-usage) section.

## Architecture & Methodology

The project implements a **Transfer Learning** approach to achieve high accuracy with limited computational resources.

* **Model:** ResNet50V2 (Pre-trained on ImageNet).
* **Strategy:** Fine-tuning. The convolutional base is frozen, and a custom classification head is trained for the 3 specific classes.
* **Preprocessing:**
    * Custom `Loader` class to parse file paths and extract labels from filenames.
    * Pandas DataFrame management for dataset split.
    * Keras `ImageDataGenerator` for rescaling (pixel normalization) and batching.

## Project Structure

The project follows a modular architecture to separate concerns (Data Loading vs. Preprocessing vs. Modeling).

```text
project-zoidberg/
│
├── data/                  # Data folder (Empty in this repo)
│   └── raw/               # Expected location for raw X-Ray images
│
├── models/                # Data folder (for saving models)
│
├── src/                   # Source code
│   ├── __init__.py
│   ├── data_loader.py     # Scans directories and assigns labels (0, 1, 2)
│   ├── preprocessing.py   # Keras Generators & Normalization
│   └── model.py           # ResNet50V2 architecture definition
│
├── main.py                # Orchestrator: Runs the training pipeline
├── requirements.txt       # List of dependencies
└── README.md              # Project documentation