
faqs = """About Me
I am Kamlesh Kumar, an M.Tech Artificial Intelligence student at NIT Hamirpur. My expertise lies in Machine Learning, Deep Learning, Large Language Models (LLMs), and AI-driven problem-solving. I have a strong foundation in Python, SQL, and OOPs and specialize in AI model optimization, fine-tuning large-scale AI models, prompt engineering, and AI agent development. My research focuses on AI applications in Structural Health Monitoring (SHM), Medical AI, and Intelligent Automation.

1. Conference Presentations & Certifications
1.1 International Conference on Automation and Machine Learning using Robots and Artificial Intelligence Methods (AMLURAIM-2025) – NIT Rourkela

Received a Certificate of Participation & Presentation from NIT Rourkela.

Presented research on AI-Enhanced SVD Methods for Mode Shape Identification in Structural Health Monitoring (SHM).

Engaged with AI experts, receiving valuable insights and feedback on intelligent SHM solutions.

1.2 International Conference on Electrical, Electronics & Automation (E2ACON 2025) – NIT Jalandhar

Received a Certificate of Participation at E2ACON 2025, co-hosted by Newcastle University in Singapore (NUiS), Singapore.

Presented research on Randomized SVD-Based Mode Shape Identification for Intelligent Structural Health Monitoring of Steel Truss Bridges.

Interacted with global researchers and industry experts, discussing AI-driven SHM innovations.

1.3 Acknowledgment

Special thanks to my supervisors, Dr. Kamlesh Dutta and Dr. Hemant Kumar Vinayak, for their continuous guidance, support, and mentorship throughout my research journey.

2. Research Interests & Expertise
2.1 Large Language Models (LLMs) & NLP

Fine-tuning models such as GPT, Llama-2, and transformer-based architectures for domain-specific applications.

2.2 Computer Vision & Deep Learning

Image and video processing, feature extraction, and AI-powered anomaly detection.

2.3 AI for Structural Health Monitoring (SHM)

Application of Randomized SVD-based methods for bridge mode shape identification.

2.4 Medical AI

Developing AI-driven diagnostic tools and chatbots for intelligent healthcare applications.

2.5 AI Agent Development & Prompt Engineering

Building autonomous AI agents capable of reasoning, contextual understanding, and self-improvement.

3. Key Projects & Achievements
3.1 MediBot – AI-Powered Medical Chatbot

Developed an LLM-based chatbot using Llama-2-7B-Chat for medical query resolution.

Implemented LangChain & Pinecone for efficient knowledge retrieval and response generation.

Focused on privacy-first AI by deploying the chatbot locally, ensuring data security.

3.2 Schizophrenia Detection using Deep Learning

Built a deep learning model for EEG-based schizophrenia detection and classification.

Applied Empirical Wavelet Transform (EWT) for feature extraction and signal enhancement.

Achieved 90.39 percent accuracy, 90 percent precision, 94.03 percent recall, and 91.79 percent F1-score.

3.3 AI-Powered Email Spam Classifier

Developed a machine learning model for spam detection using Naïve Bayes, SVM, and XGBoost.

Achieved 97.87 percent accuracy, optimizing through hyperparameter tuning and cross-validation.

3.4 Structural Health Monitoring (SHM) of Bridges

Applied Randomized SVD-based methods to extract mode shapes from bridge vibration data.

Improved signal denoising, mode shape accuracy, and computational efficiency.

3.5 LLM Fine-Tuning & AI Agent Development

Fine-tuned GPT-4, Llama-2, and Falcon models for domain-specific applications in healthcare, cybersecurity, and customer support.

Developed AI-driven agents capable of contextual reasoning, autonomous decision-making, and self-improvement.

4. Technical Skills
4.1 Programming Languages

Python, C, SQL, OOPs

4.2 Machine Learning & Deep Learning

TensorFlow, PyTorch, Scikit-Learn, OpenCV

4.3 LLM & NLP

LangChain, Hugging Face, Transformer models, Prompt Engineering

4.4 Development Tools

Jupyter Notebook, VS Code, Flask, Docker, Git

4.5 Platforms

Google Cloud, Google Colab, Kaggle, Hugging Face Hub

5. Career Goals & Aspirations
5.1 Research & Innovation

Advance AI applications in healthcare, SHM, and real-time automation.

5.2 LLM Fine-Tuning & Autonomous AI Systems

Work on fine-tuning LLMs, autonomous AI systems, and AI-driven decision-making models.

5.3 Scalable AI Solutions

Develop scalable and impactful AI solutions for real-world applications.

5.4 Collaboration & Industry Engagement

Collaborate with leading AI research groups, startups, and technology companies.

I am actively seeking AI research and industry opportunities to apply my expertise and contribute to cutting-edge AI advancements. Let’s connect.
"""

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer()

tokenizer.fit_on_texts([faqs])

len(tokenizer.word_index)

input_sequences = []
for sentence in faqs.split('\n'):
  tokenized_sentence = tokenizer.texts_to_sequences([sentence])[0]

  for i in range(1,len(tokenized_sentence)):
    input_sequences.append(tokenized_sentence[:i+1])

input_sequences

max_len = max([len(x) for x in input_sequences])

from tensorflow.keras.preprocessing.sequence import pad_sequences
padded_input_sequences = pad_sequences(input_sequences, maxlen = max_len, padding='pre')

padded_input_sequences

X = padded_input_sequences[:,:-1]

y = padded_input_sequences[:,-1]

X.shape

y.shape



tokenizer.word_index

from tensorflow.keras.utils import to_categorical
y = to_categorical(y,num_classes=306)

y.shape

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense
from tensorflow.keras.models import Model

# Get correct input length dynamically
input_length = X.shape[1]

# Define model with corrected input shape

model = Sequential()
model.add(Embedding(input_dim=306, output_dim=100, input_length=input_length))
model.add(LSTM(150, return_sequences=True))  # First LSTM should return sequences
model.add(LSTM(150))  # Second LSTM processes sequences
model.add(Dense(306, activation='softmax'))  # Output layer

# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

# Train model
model.fit(X, y, epochs=100)

print("Original X shape:", X.shape)

import time
import numpy as np

text = "Kamlesh Kumar is a "

for i in range(10):
  # tokenize
  token_text = tokenizer.texts_to_sequences([text])[0]
  # padding
  padded_token_text = pad_sequences([token_text], maxlen=73, padding='pre')
  # predict
  pos = np.argmax(model.predict(padded_token_text))

  for word,index in tokenizer.word_index.items():
    if index == pos:
      text = text + " " + word
      print(text)
      time.sleep(2)

