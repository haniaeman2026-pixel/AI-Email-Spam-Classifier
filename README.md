# 📧 AI Email Spam Classifier

A professional Machine Learning web application that classifies emails as **Spam** or **Not Spam** using Natural Language Processing (NLP) techniques and a **Multinomial Naive Bayes** model. The application is deployed with **Streamlit** and provides real-time predictions through an interactive dashboard.

---

## 📌 Project Overview

Email spam is one of the most common cybersecurity challenges. This project uses Machine Learning to automatically analyze email text and predict whether an email is legitimate or spam.

The application performs text preprocessing, converts text into numerical features using TF-IDF Vectorization, and classifies emails using a trained Multinomial Naive Bayes model.

---

## ✨ Features

- 📧 Detects Spam and Safe Emails
- 🤖 Machine Learning-Based Prediction
- 📊 Confidence Score Display
- 📈 Probability Visualization
- 🎈 Interactive Streamlit Dashboard
- 📥 Download Prediction Report
- ⚡ Fast Real-Time Prediction
- 💻 User-Friendly Interface

---

## 🛠️ Technologies Used

- Python
- Streamlit
- Scikit-learn
- Pandas
- NumPy
- Matplotlib
- Joblib
- Hugging Face Datasets

---

## 🧠 Machine Learning Workflow

1. Load Dataset
2. Clean and Preprocess Email Text
3. Convert Text using TF-IDF Vectorizer
4. Train Multinomial Naive Bayes Model
5. Save Model and Vectorizer
6. Deploy using Streamlit

---

## 📂 Project Structure

```text
Email-Spam-Classifier/
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

## ⚙️ Installation

### Clone the Repository

```bash
git clone https://github.com/yourusername/Email-Spam-Classifier.git
```

### Move into the Project Folder

```bash
cd Email-Spam-Classifier
```

### Create Virtual Environment

**Windows**

```bash
python -m venv venv
```

### Activate Virtual Environment

```bash
.\venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Train the Model

```bash
python train.py
```

This generates:

- `spam_model.pkl`
- `vectorizer.pkl`

---

## 🚀 Run the Application

```bash
streamlit run app.py
```

---

## 📊 Model Information

| Component | Value |
|-----------|-------|
| Algorithm | Multinomial Naive Bayes |
| Feature Extraction | TF-IDF Vectorizer |
| Task | Email Spam Classification |
| Deployment | Streamlit |

---

## 📸 Application Preview

The application includes:

- Professional Dashboard
- Email Input Area
- Spam Detection
- Confidence Score
- Probability Visualization
- Downloadable Prediction Report

---

## 📚 Future Improvements

- Deep Learning (LSTM/BERT)
- Email Attachment Analysis
- Dark Mode
- Prediction History
- Cloud Deployment
- Multi-Language Support

---

## 👩‍💻 Developer

**Hania Eman**

AI & Data Science Student

Machine Learning Enthusiast

Python Developer

---

## 📄 License

This project is developed for educational and internship purposes.

---

## ⭐ Support

If you found this project helpful, consider giving it a ⭐ on GitHub.

---

**Made with ❤️ using Python, Machine Learning, and Streamlit**