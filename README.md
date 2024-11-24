# Sentiment Analysis on Tweets using BERT

## Overview
This project implements a sentiment analysis model using the BERT transformer architecture to classify tweets as positive or negative. The model is trained on the [Sentiment140 dataset](http://help.sentiment140.com/for-students/) and deployed with a simple API for real-time predictions. An Android app was also developed to use the model's API for real-time sentiment analysis.

## Dataset
- **Name**: Sentiment140
- **Columns Used**: `target`, `text`
- **Target Mapping**: 0 -> Negative, 4 -> Positive

## Project Workflow

### 1. Data Preprocessing
To prepare data for BERT, the following preprocessing steps were applied:
- **Text Cleaning**: Removed URLs, mentions, hashtags, and special characters.
- **Tokenization**: Used BERT's tokenizer to convert text into tokens.
- **Padding and Truncation**: Ensured a uniform sequence length.
- **Train-Test Split**: Split the data into training (80%) and testing (20%) sets.

### 2. Model Architecture
A pre-trained BERT model (`bert-base-uncased`) with a classification head was used for sentiment classification. The model architecture includes:
- **BERT Base Model**: Provides contextualized embeddings.
- **Classification Head**: Maps embeddings to binary sentiment labels (positive/negative).

### 3. Model Training
- **Optimizer**: AdamW with a learning rate of `5e-5`.
- **Batch Size**: 16
- **Epochs**: 3
- **Training Loop**: Performed forward and backward passes, calculated loss, and updated model weights.

### 4. Model Evaluation
Evaluated model performance using:
- **Metrics**: Accuracy, Precision, Recall, F1-score
- **Confusion Matrix**: Visualized model performance across classes
- **Classification Report**: Displayed precision, recall, and F1-score per class

### 5. Model Deployment
Deployed the model with a FastAPI server for real-time predictions.
1. **Endpoint**: `/predict`
2. **Input**: JSON payload containing text.
3. **Output**: Predicted sentiment (Positive or Negative).

#### Example of API usage:
```python
import requests

response = requests.post("http://localhost:8000/predict", json={"text": "I love using this product!"})
print(response.json())
# Output: {"sentiment": "Positive"}
