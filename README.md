# 📊 Twitter Sentiment Analysis with Fine-Tuned RoBERTa

## 📌 Project Overview

This project focuses on performing sentiment analysis on tweets by fine-tuning a transformer-based language model. Specifically, I used the `cardiffnlp/twitter-roberta-base-sentiment` model as a starting point and fine-tuned it on a labeled dataset of tweets to improve its performance for a custom sentiment classification task.

The goal was to classify tweets into sentiment categories based on their content and evaluate the effectiveness of fine-tuning versus using a base pretrained model.

---

## 🚀 Key Features

- ✅ Fine-tuned a state-of-the-art transformer model (`twitter-roberta-base-sentiment`) for sentiment classification.
- ✅ Custom dataset split into training and validation sets using stratified sampling.
- ✅ Training progress tracked through saved checkpoints.
- ✅ Inference performed on unseen test data using the final fine-tuned model.
- ✅ Predictions exported in a submission-ready `.csv` file.

---

## 📝 Dataset

The dataset consisted of tweets labeled with sentiment categories.  
It was split into:

- **Training set**: 90% of the data
- **Validation set**: 10% of the data (using stratified split to maintain label balance)
- **Test set**: Unlabeled tweets for final model predictions

---

## 🛠️ Tools & Libraries

- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)
- [PyTorch](https://pytorch.org/)
- [Scikit-learn](https://scikit-learn.org/)
- [Pandas](https://pandas.pydata.org/)
- [tqdm](https://github.com/tqdm/tqdm) for progress bars

---

## 📊 Model Training

The fine-tuning process involved:

1. **Loading the `cardiffnlp/twitter-roberta-base-sentiment` model and tokenizer**
2. **Tokenizing the tweets and preparing datasets**
3. **Training the model with a custom training loop using the Hugging Face `Trainer`**
4. **Saving model checkpoints at regular intervals**
5. **Using the final checkpoint (`checkpoint-1338`) for inference**

---

## 📈 Inference & Submission

- The fine-tuned model was loaded from the final checkpoint.
- The tokenizer was loaded from the original `cardiffnlp/twitter-roberta-base-sentiment`.
- Inference was performed on the test set in batches to prevent memory overload.
- Final predictions were saved in `Predictions.csv` for evaluation or submission.

---


## 📊 Results

The fine-tuned model showed improved sentiment classification accuracy compared to the base model on the validation set.  
Future improvements could include hyperparameter tuning, experimenting with different model architectures, or data augmentation techniques.

---

