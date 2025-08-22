# 🤖 Neural Network Sentiment Classifier  

This project implements a **transformer-based neural network** for **multi-label text sentiment classification**, developed as part of the **Neural Networks course (Final Project)** at the University of Arizona.  

The model was submitted to a **Codabench competition** as part of the class-wide evaluation.  

---

## 🎯 Objectives  
- Build a neural network for sentiment classification using transformer embeddings.  
- Practice **hyperparameter tuning** (learning rate, batch size, dropout, dense layers).  
- Evaluate model performance with **F1-score** on validation and test datasets.  

---

## 🗂️ Project Structure  
```
graduate-project-sahana-santhosh/
│── nn.py            # Training script with transformer + Keras pipeline  
│── requirements.txt # Dependencies  
│── README.md        # Project overview (this file)
```

---

## ⚙️ Tools & Libraries  
- **Python**  
- **TensorFlow / Keras**  
- **Hugging Face Transformers (DistilRoBERTa)**  
- **scikit-learn** (evaluation metrics)  
- **datasets, pandas, numpy**  

---

## 🔍 Methods  
1. **Data Loading**  
   - Loaded CSV datasets (`train.csv`, `dev.csv`) using Hugging Face `datasets`.  
   - Tokenized input text with **DistilRoBERTa tokenizer**.  

2. **Model Architecture**  
   - Pre-trained **DistilRoBERTa** transformer as embedding layer.  
   - Added **dense layers** with dropout for regularization.  
   - Configurable units (256, 128) with activation functions.  

3. **Training & Tuning**  
   - Hyperparameter tuning: learning rate, batch size, dropout rate, number of epochs.  
   - Optimized with **Adam optimizer**.  

4. **Evaluation**  
   - Used **macro/micro F1-score** for multi-label classification.  
   - Saved best model in `.keras` format for reproducibility.  

---

## 📈 Key Results  
- Successfully fine-tuned DistilRoBERTa for sentiment classification.  
- Achieved strong F1-score performance on validation data (Codabench evaluated).  
- Demonstrated improvements over baseline models by applying hyperparameter tuning and dense layer optimization.  

---

## 📜 Deliverables  
- Model training pipeline (`nn.py`)  
- Fine-tuned model (`model.keras`)  
- Codabench-ready submission files  

---

## 👩‍💻 Contributors  
- [Sahana Santhosh] — Model implementation, hyperparameter tuning, evaluation setup  

---

## 🚀 How to Run  
1. Clone the repository  
   ```bash
   git clone https://github.com/yourusername/nn-sentiment-classifier.git
   cd graduate-project-sahana-santhosh
   ```  
2. Install requirements  
   ```bash
   pip install -r requirements.txt
   ```  
3. Train the model  
   ```bash
   python nn.py --train_path train.csv --dev_path dev.csv
   ```  

---

## 📌 License  
This project is for academic purposes (Neural Networks course, University of Arizona).  
