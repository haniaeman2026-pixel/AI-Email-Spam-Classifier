# ==========================================================
# 📧 AI EMAIL SPAM CLASSIFIER
# Developed By: Hania Eman
# ==========================================================

import streamlit as st
import joblib
import re
import string
import matplotlib.pyplot as plt

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="AI Email Spam Classifier",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# LOAD MODEL
# ==========================================================

@st.cache_resource
def load_model():
    model = joblib.load("spam_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model()

# ==========================================================
# TEXT CLEANING
# ==========================================================

def clean_text(text):

    text = text.lower()

    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"www\S+", "", text)
    text = re.sub(r"\d+", "", text)

    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    text = re.sub(r"\s+", " ", text).strip()

    return text

# ==========================================================
# PREMIUM CSS
# ==========================================================

st.markdown("""
<style>

.stApp{
background:#EEF4FF;
}

/* Sidebar */

section[data-testid="stSidebar"]{
background:linear-gradient(180deg,#0F172A,#1D4ED8);
}

section[data-testid="stSidebar"] *{
color:white;
}

/* Hero Banner */

.hero{
background:linear-gradient(90deg,#2563EB,#06B6D4);
padding:40px;
border-radius:22px;
text-align:center;
color:white;
box-shadow:0px 10px 30px rgba(0,0,0,.18);
margin-bottom:25px;
}

/* Glass Card */

.glass{
background:white;
padding:25px;
border-radius:18px;
box-shadow:0px 8px 20px rgba(0,0,0,.08);
margin-bottom:20px;
}

/* Button */

div.stButton > button{

width:100%;
height:60px;
font-size:22px;
font-weight:bold;
background:linear-gradient(90deg,#2563EB,#06B6D4);
color:white;
border:none;
border-radius:12px;
transition:.3s;

}

div.stButton > button:hover{

transform:scale(1.02);

}

/* Metrics */

[data-testid="metric-container"]{

background:white;
padding:15px;
border-radius:15px;
box-shadow:0px 5px 15px rgba(0,0,0,.08);

}

</style>
""", unsafe_allow_html=True)

# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.title("📧 AI Spam Detector")

st.sidebar.markdown("---")

st.sidebar.success("👩‍💻 Developer")

st.sidebar.write("**Hania Eman**")

st.sidebar.markdown("---")

st.sidebar.info("🤖 Model")

st.sidebar.write("Multinomial Naive Bayes")

st.sidebar.info("📊 Vectorizer")

st.sidebar.write("TF-IDF")

st.sidebar.markdown("---")

st.sidebar.success("💻 Tech Stack")

st.sidebar.write("✔ Python")

st.sidebar.write("✔ Streamlit")

st.sidebar.write("✔ Scikit-Learn")

st.sidebar.write("✔ Joblib")

st.sidebar.write("✔ Hugging Face")

st.sidebar.markdown("---")


# ==========================================================
# HERO SECTION
# ==========================================================

st.markdown("""
<div class="hero">

<h1>📧 AI Email Spam Classifier</h1>

<h4>Smart Email Classification using Machine Learning</h4>

<p>
Fast • Accurate • Intelligent • Secure
</p>

</div>
""", unsafe_allow_html=True)

# ==========================================================
# WELCOME CARD
# ==========================================================

st.markdown("""
<div class="glass">

<h2>👋 Welcome</h2>

<p style="font-size:18px;">

Welcome to the <b>AI Email Spam Classifier</b>.

This application analyzes your email using Machine Learning
and predicts whether it is:

✅ Safe Email

🚨 Spam Email

Paste your email below and click
<b>Predict</b>.

</p>

</div>
""", unsafe_allow_html=True)
# ==========================================================
# EMAIL INPUT SECTION
# ==========================================================

st.markdown('<div class="glass">', unsafe_allow_html=True)

st.subheader("📨 Enter Email")

email_text = st.text_area(
    label="Email Content",
    height=220,
    placeholder="""Example:

Congratulations!

You have won a FREE iPhone 16.

Click here to claim your prize.

https://example.com
"""
)

predict = st.button("🚀 Predict Email", use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)

# ==========================================================
# PREDICTION
# ==========================================================

if predict:

    if email_text.strip() == "":

        st.warning("⚠ Please enter an email before prediction.")

    else:

        cleaned_email = clean_text(email_text)

        vector = vectorizer.transform([cleaned_email])

        prediction = model.predict(vector)[0]

        probability = model.predict_proba(vector)[0]

        confidence = probability.max() * 100

        spam = probability[1] * 100

        safe = probability[0] * 100

        st.markdown("<br>", unsafe_allow_html=True)

        # ==============================================
        # RESULT CARD
        # ==============================================

        if prediction == 1:

            st.error("🚨 Spam Email Detected!")

        else:

            st.balloons()
            st.snow()

            st.success("🎉 Safe Email Detected")

        # ==============================================
        # METRICS
        # ==============================================

        c1, c2, c3 = st.columns(3)

        with c1:
            st.metric(
                "Prediction",
                "🚨 Spam" if prediction == 1 else "✅ Safe"
            )

        with c2:
            st.metric(
                "Confidence",
                f"{confidence:.2f}%"
            )

        with c3:
            st.metric(
                "Model",
                "Naive Bayes"
            )

        st.markdown("---")

        # ==============================================
        # PROBABILITY SECTION
        # ==============================================

        st.subheader("📊 Prediction Probability")

        left, right = st.columns(2)

        with left:

            st.write("🚨 Spam Probability")

            st.progress(min(int(spam), 100))

            st.write(f"**{spam:.2f}%**")

        with right:

            st.write("✅ Safe Probability")

            st.progress(min(int(safe), 100))

            st.write(f"**{safe:.2f}%**")

        st.markdown("---")

        # ==============================================
        # EMAIL PREVIEW
        # ==============================================

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("📧 Original Email")

            st.info(email_text)

        with col2:

            st.subheader("🧹 Cleaned Email")

            st.code(cleaned_email)

        st.success("✔ Prediction Completed Successfully")
        # ==========================================================
# DASHBOARD & VISUALIZATION
# ==========================================================

if "prediction" in locals():

    st.markdown("---")

    st.header("📊 AI Dashboard")

    import matplotlib.pyplot as plt

    d1, d2, d3, d4 = st.columns(4)

    with d1:
        st.metric("🤖 Model", "Naive Bayes")

    with d2:
        st.metric("📑 Vectorizer", "TF-IDF")

    with d3:
        st.metric("🎯 Confidence", f"{confidence:.2f}%")

    with d4:
        st.metric(
            "📧 Result",
            "Spam" if prediction == 1 else "Safe"
        )

    st.markdown("---")

    st.subheader("📈 Prediction Distribution")

    fig, ax = plt.subplots(figsize=(5,5))

    ax.pie(
        [spam, safe],
        labels=["Spam", "Safe"],
        autopct="%1.1f%%",
        startangle=90,
        colors=["#EF4444", "#10B981"],
        explode=(0.05,0)
    )

    ax.axis("equal")

    st.pyplot(fig)

# ==========================================================
# ABOUT PROJECT
# ==========================================================

st.markdown("---")

st.header("📖 About This Project")

st.info("""

📧 AI Email Spam Classifier

This application predicts whether an email is:

✅ Safe

🚨 Spam

using Machine Learning.

Algorithm:
• Multinomial Naive Bayes

Feature Extraction:
• TF-IDF Vectorizer

Dataset:
• Hugging Face SMS Spam Dataset

""")

# ==========================================================
# FEATURES
# ==========================================================

st.markdown("---")

left, right = st.columns(2)

with left:

    st.success("""

### 🚀 Features

✔ Real-Time Prediction

✔ Machine Learning

✔ Confidence Score

✔ Beautiful Dashboard

✔ Email Cleaning

""")

with right:

    st.info("""

### 💻 Technologies

✔ Python

✔ Streamlit

✔ Scikit-Learn

✔ Joblib

✔ Hugging Face

""")

# ==========================================================
# DOWNLOAD REPORT
# ==========================================================

if "prediction" in locals():

    st.markdown("---")

    report = f"""
EMAIL SPAM CLASSIFIER REPORT

Prediction:
{"Spam" if prediction == 1 else "Safe"}

Confidence:
{confidence:.2f} %

Spam Probability:
{spam:.2f} %

Safe Probability:
{safe:.2f} %

Developer:
Hania Eman
"""

    st.download_button(
        "📥 Download Prediction Report",
        data=report,
        file_name="Prediction_Report.txt",
        mime="text/plain"
    )

# ==========================================================
# FOOTER
# ==========================================================

st.markdown("---")

st.markdown("""
<div style="
background:linear-gradient(90deg,#2563EB,#06B6D4);
padding:25px;
border-radius:18px;
text-align:center;
color:white;
">

<h2>📧 AI Email Spam Classifier</h2>

<p>
Developed by <b>Hania Eman</b>
</p>

<p>
Machine Learning Internship Project
</p>

<p>
⭐ Thank you for using this application!
</p>

</div>
""", unsafe_allow_html=True)
# ==========================================================
# EXTRA FEATURES
# ==========================================================

st.markdown("---")

st.header("📌 Tips to Stay Safe from Spam Emails")

tip1, tip2 = st.columns(2)

with tip1:

    st.warning("""

### 🚨 Avoid Spam Emails

• Never click unknown links.

• Don't share OTPs.

• Ignore suspicious attachments.

• Verify sender email.

""")

with tip2:

    st.success("""

### ✅ Stay Secure

• Enable Two-Factor Authentication

• Use Strong Passwords

• Keep Antivirus Updated

• Report Spam Emails

""")

# ==========================================================
# SYSTEM STATUS
# ==========================================================

st.markdown("---")

st.subheader("⚙️ System Status")

status1, status2, status3 = st.columns(3)

with status1:
    st.success("🟢 Model Loaded")

with status2:
    st.success("🟢 Vectorizer Loaded")

with status3:
    st.success("🟢 Application Running")

# ==========================================================
# ABOUT DEVELOPER
# ==========================================================

st.markdown("---")

st.subheader("👩‍💻 About Developer")

st.info("""

**Name:** Hania Eman

🎓 AI & Data Science Student

🤖 Machine Learning Enthusiast

💻 Python Developer

📊 Streamlit Dashboard Developer

""")

# ==========================================================
# THANK YOU MESSAGE
# ==========================================================

st.markdown("---")

st.success("🎉 Thank you for using the AI Email Spam Classifier!")

st.caption(
    "Built with ❤️ using Python, Streamlit and Scikit-Learn"
)

# ==========================================================
# END OF APPLICATION
# ==========================================================