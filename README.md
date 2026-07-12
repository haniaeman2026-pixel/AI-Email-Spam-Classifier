#  AI Email Spam Classifier

> An AI-powered Email Spam Classifier built with Machine Learning and Natural Language Processing (NLP) to classify emails as **Spam** or **Safe** through a professional Streamlit web application.

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?logo=streamlit)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange?logo=scikitlearn)
![License](https://img.shields.io/badge/License-Educational-green)

---

#  Overview

Email spam is one of the most common cybersecurity challenges. This project uses **Machine Learning** and **Natural Language Processing (NLP)** to automatically detect whether an email is **Spam** or **Safe**.

The application preprocesses email text, converts it into numerical features using **TF-IDF Vectorization**, and predicts the email category using the **Multinomial Naive Bayes** algorithm.

A modern **Streamlit Dashboard** provides real-time predictions, confidence scores, probability visualization, and downloadable prediction reports.

---

#  Features

- 📧 Real-Time Email Spam Detection
- 🤖 Machine Learning Prediction
- 🧠 Natural Language Processing (NLP)
- 📊 Confidence Score
- 📈 Prediction Probability
- 🥧 Pie Chart Visualization
- 📥 Download Prediction Report
- 🎈 Interactive Streamlit Dashboard
- ⚡ Fast & Accurate Prediction
- 💻 Responsive User Interface

---

#  Tech Stack

- Python
- Streamlit
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Joblib
- Hugging Face Datasets

---

#  Machine Learning Workflow

```
Email Text
      │
      ▼
Text Cleaning
      │
      ▼
TF-IDF Vectorization
      │
      ▼
Multinomial Naive Bayes
      │
      ▼
Spam / Safe Prediction
```

---

#  Project Structure

```text
AI-Email-Spam-Classifier/
│
├── app.py
├── train.py
├── spam_model.pkl
├── vectorizer.pkl
├── requirements.txt
├── README.md
├── .gitignore
└── assets/
```

---

#  Installation

### 1️ Clone Repository

```bash
git clone https://github.com/haniaeman2026-pixel/AI-Email-Spam-Classifier.git
```

### 2️ Open Project

```bash
cd AI-Email-Spam-Classifier
```

### 3️ Create Virtual Environment

```bash
python -m venv venv
```

### 4️ Activate Virtual Environment

**Windows**

```bash
.\venv\Scripts\activate
```

### 5️ Install Libraries

```bash
pip install -r requirements.txt
```

---

#  Train the Model

```bash
python train.py
```

This creates:

- spam_model.pkl
- vectorizer.pkl

---

#  Run the Application

```bash
streamlit run app.py
```

---

#  Machine Learning Model

| Component | Details |
|-----------|----------|
| Algorithm | Multinomial Naive Bayes |
| Feature Extraction | TF-IDF Vectorizer |
| Task | Email Spam Classification |
| Deployment | Streamlit |

---

#  Application Features

The application includes:

- 📧 Email Input Box
- 🤖 AI Spam Prediction
- 📊 Confidence Score
- 📈 Probability Visualization
- 🥧 Pie Chart
- 📥 Download Prediction Report
- 🎈 Interactive Dashboard

---

#  Future Improvements

- Deep Learning (LSTM / BERT)
- Dark Mode
- Email Attachment Analysis
- Multi-Language Support
- Cloud Deployment
- Prediction History

---

#  Developer

**Hania Eman**

🎓 AI & Data Science Student

💻 Python Developer

🤖 Machine Learning Enthusiast

📊 Streamlit Dashboard Developer

---

#  License

This project is created for educational purposes and internship learning.

---

#  Support

If you found this project useful, please consider giving it a ⭐ on GitHub.

---

##  Thank You

Thank you for visiting this project.

**Made with ❤️ using Python, Machine Learning, NLP, and Streamlit.**
