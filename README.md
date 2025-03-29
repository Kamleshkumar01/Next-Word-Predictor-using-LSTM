# Next Word Predictor using LSTM


## 📌 Overview
Next Word Predictor is an **LSTM-based** deep learning model that predicts the next word based on an input sequence. It utilizes **Natural Language Processing (NLP) techniques**, including tokenization, word embedding, and sequence modeling, to generate accurate predictions.

## 🚀 Features
✅ Uses **LSTM neural networks** for next-word prediction  
✅ Tokenizes and preprocesses text using **TensorFlow/Keras Tokenizer**  
✅ Implements **sequence modeling** to learn word relationships  
✅ Provides **real-time word prediction** based on input text  

---

## 🔧 How to Run?
### 📌 STEPS:
### Step 01: Clone the Repository
```bash
git clone https://github.com/your-username/next-word-predictor-lstm.git
cd next-word-predictor-lstm
```

### Step 02: Create a Conda Environment
```bash
conda create -n lstm_predictor python=3.8 -y
conda activate lstm_predictor
```

### Step 03: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 04: Train the Model
Run the training script to preprocess text and train the LSTM model:
```bash
python lstm_project.py
```

Alternatively, you can use the trained model for prediction:
```python
from lstm_project import tokenizer, model, pad_sequences
import numpy as np

def predict_next_word(text, num_words=1):
    for _ in range(num_words):
        token_text = tokenizer.texts_to_sequences([text])[0]
        padded_token_text = pad_sequences([token_text], maxlen=73, padding='pre')
        pos = np.argmax(model.predict(padded_token_text))
        for word, index in tokenizer.word_index.items():
            if index == pos:
                text += " " + word
                break
    return text

input_text = "Kamlesh Kumar is"
predicted_text = predict_next_word(input_text, num_words=10)
print("Predicted Sentence:", predicted_text)
```

---

## 📂 Dataset
- The dataset consists of textual data used for training the model.  
- Preprocessing includes **tokenization, padding, and vocabulary creation** using TensorFlow/Keras Tokenizer.  

## 🏗 Model Architecture
🔹 **Embedding Layer**: Converts words into dense vector representations  
🔹 **LSTM Layers**: Captures sequential dependencies in the text  
🔹 **Dense Layer**: Outputs probability distribution over the vocabulary  

---

## 🛠 Dependencies
- Python 3.x  
- TensorFlow/Keras  
- NumPy  
- Pandas  
- Matplotlib  

## 📊 Results
✔️ The model achieves **accurate next-word predictions** based on training data  
✔️ Performance improves with **larger datasets and hyperparameter tuning**  

---

## 🔮 Future Improvements
📌 Implement **Bidirectional LSTMs** for enhanced accuracy  
📌 Train on **larger and diverse datasets** for better generalization  
📌 Integrate with a **web-based application** for real-time usage  

---

## 🤝 Contributing
Feel free to **fork** the repository, submit **pull requests**, and report issues!  

## 📜 License
📝 This project is licensed under the **MIT License**.

