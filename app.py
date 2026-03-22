import streamlit as st
import sqlite3
import hashlib
import datetime
import matplotlib.pyplot as plt
import pandas as pd
import re
import speech_recognition as sr
from gtts import gTTS
import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import nltk
from nltk.corpus import stopwords

# Ensure NLTK data is available
nltk.download('stopwords')

# =============================
# 1. DATABASE & SECURITY
# =============================
conn = sqlite3.connect('patient_data.db', check_same_thread=False)
c = conn.cursor()
c.execute('CREATE TABLE IF NOT EXISTS users(username TEXT, password TEXT)')
c.execute('CREATE TABLE IF NOT EXISTS history(username TEXT, time TEXT, prediction TEXT, text TEXT)')
conn.commit()

def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_login(username, password):
    c.execute('SELECT * FROM users WHERE username =? AND password =?', (username, password))
    return c.fetchall()

# =============================
# 2. ML MODEL
# =============================
@st.cache_resource
def train_model():
    try:
        df = pd.read_csv("tweet_emotions.csv")
        df = df[["content", "sentiment"]].rename(columns={"content": "text", "sentiment": "label"})
    except FileNotFoundError:
        st.error("Dataset 'tweet_emotions.csv' not found.")
        st.stop()

    stop_words = set(stopwords.words('english'))
    def clean(text):
        text = str(text).lower()
        text = re.sub(r'[^a-zA-Z]', ' ', text)
        return " ".join([w for w in text.split() if w not in stop_words])

    df["clean_text"] = df["text"].apply(clean)
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df["clean_text"])
    model = LogisticRegression(max_iter=200).fit(X, df["label"])
    return model, vectorizer, clean

model, vectorizer, preprocess = train_model()

# =============================
# 3. STYLED LOGIN PAGE
# =============================
def login_page():
    # Use a high-quality, stable counselor image
    # Note: If this link ever breaks, the text remains professional
    avatar_url = "https://cdn-icons-png.flaticon.com/512/6833/6833605.png"

    st.markdown(f"""
        <style>
        .stApp {{ background: #1a1a2e; overflow: hidden; }}
        
        /* Character placement & Floating animation */
        .char-bg {{
            position: fixed; left: 10%; top: 15%; width: 400px; z-index: 0;
            animation: float 4s ease-in-out infinite;
        }}
        @keyframes float {{
            0% {{ transform: translateY(0px); }}
            50% {{ transform: translateY(-20px); }}
            100% {{ transform: translateY(0px); }}
        }}

        /* Sliding Glassmorphism Box */
        .login-box {{
            background: rgba(255, 255, 255, 0.08);
            backdrop-filter: blur(15px);
            padding: 40px; border-radius: 30px;
            border: 1px solid rgba(255, 255, 255, 0.15);
            box-shadow: 0 15px 45px rgba(0,0,0,0.6);
            animation: slideIn 1.2s cubic-bezier(0.16, 1, 0.3, 1);
            z-index: 1; color: white;
        }}
        @keyframes slideIn {{
            from {{ opacity: 0; transform: translateX(120px); }}
            to {{ opacity: 1; transform: translateX(0); }}
        }}
        .login-header {{ font-size: 30px; font-weight: bold; color: #ffffff; margin-bottom: 5px; }}
        label {{ color: white !important; font-weight: bold; }}
        </style>
        <img src="{avatar_url}" class="char-bg">
        """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1.6, 2, 0.8])
    with c2:
        st.markdown('<div class="login-box">', unsafe_allow_html=True)
        st.markdown('<p class="login-header">Register now</p>', unsafe_allow_html=True)
        st.caption("Secure your clinical AI session")
        
        t1, t2 = st.tabs(["🔒 Member Login", "📝 New Registration"])
        with t1:
            u = st.text_input("Username", key="l_user")
            p = st.text_input("Password", type="password", key="l_pw")
            if st.button("Unlock AI Counselor", key="l_btn"):
                if check_login(u, make_hashes(p)):
                    st.session_state['logged_in'] = True
                    st.session_state['user'] = u
                    st.rerun()
                else: st.error("Invalid Username or Password")
        with t2:
            nu = st.text_input("Patient Full Name")
            npw = st.text_input("Choose Password", type="password")
            if st.button("Create My Account", key="r_btn"):
                if nu and npw:
                    c.execute('INSERT INTO users(username, password) VALUES (?,?)', (nu, make_hashes(npw)))
                    conn.commit()
                    st.success("Registered! Now go to Member Login.")
                else: st.warning("Please fill all fields.")
        st.markdown('</div>', unsafe_allow_html=True)

# =============================
# 4. MAIN APPLICATION
# =============================
if 'logged_in' not in st.session_state: st.session_state['logged_in'] = False

if not st.session_state['logged_in']:
    login_page()
else:
    # --- Professional Dashboard Styles ---
    st.markdown("""
        <style>
        .stApp { background: #0e1117; color: white; }
        .main-card {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 15px; padding: 25px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            margin-bottom: 25px;
        }
        .bot-bubble {
            background-color: #1f2937; padding: 20px;
            border-radius: 15px 15px 15px 0px;
            border: 1px solid #3b82f6; margin-top: 15px;
        }
        </style>
        """, unsafe_allow_html=True)

    with st.sidebar:
        st.title(f"👤 Patient: {st.session_state['user'].capitalize()}")
        if st.button("🚪 Logout Session"):
            st.session_state['logged_in'] = False
            st.rerun()
        st.markdown("---")
        st.info("Recording Active: 60 Seconds")

    st.markdown("<h1 style='text-align: center; color: #60a5fa;'>🧠 Mental Health AI Assistant</h1>", unsafe_allow_html=True)
    
    responses = {
        "sadness": "I'm sorry you're feeling this way. [Tip: Try a short walk.]",
        "anger": "I understand you're upset. [Tip: Take a deep breath.]",
        "worry": "It's okay to feel anxious. [Tip: Focus on your breathing.]",
        "happiness": "That's great! Keep that energy up.",
        "neutral": "I'm here to listen. Tell me more."
    }

    # Input Control
    user_input = None
    with st.container():
        st.markdown('<div class="main-card">', unsafe_allow_html=True)
        st.subheader("📥 Consultation Inputs")
        col_mic, col_file = st.columns(2)
        with col_mic:
            if st.button("🎤 Start 60s Recording"):
                st.info("Recording...")
                rec = sd.rec(int(60 * 44100), samplerate=44100, channels=1, dtype='int16')
                sd.wait()
                write("output.wav", 44100, rec)
                try:
                    r = sr.Recognizer()
                    with sr.AudioFile("output.wav") as source:
                        user_input = r.recognize_google(r.record(source))
                        st.success(f"Recognized: {user_input}")
                except: st.error("Recording error.")
        with col_file:
            uploaded_file = st.file_uploader("Upload Session Audio (.wav)", type=["wav"])
            if uploaded_file:
                try:
                    r = sr.Recognizer()
                    with sr.AudioFile(uploaded_file) as source:
                        user_input = r.recognize_google(r.record(source))
                except: st.error("File error.")

        text_msg = st.chat_input("How are you feeling right now?")
        if text_msg: user_input = text_msg
        st.markdown('</div>', unsafe_allow_html=True)

    if user_input:
        clean_text = preprocess(user_input)
        pred = model.predict(vectorizer.transform([clean_text]))[0]
        reply = responses.get(pred, "I'm here for you.")
        
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        c.execute('INSERT INTO history(username, time, prediction, text) VALUES (?,?,?,?)', 
                  (st.session_state['user'], now, pred, user_input))
        conn.commit()

        st.markdown(f"**Patient:** {user_input}")
        st.markdown(f'<div class="bot-bubble"><b>AI Counselor:</b> {reply}</div>', unsafe_allow_html=True)
        
        tts = gTTS(text=reply, lang='en')
        tts.save("response.mp3")
        st.audio("response.mp3")

    # Analytics
    st.markdown("---")
    st.subheader("📊 Your Progress Insights")
    user_data = pd.read_sql(f"SELECT * FROM history WHERE username='{st.session_state['user']}'", conn)
    
    if not user_data.empty:
        g_col, d_col = st.columns([2, 1])
        with g_col:
            st.markdown('<div class="main-card">', unsafe_allow_html=True)
            counts = user_data['prediction'].value_counts()
            fig, ax = plt.subplots(figsize=(8, 4))
            fig.patch.set_facecolor('#0e1117')
            ax.set_facecolor('#0e1117')
            counts.plot(kind='bar', ax=ax, color='#60a5fa')
            ax.tick_params(colors='white')
            plt.xticks(rotation=45)
            st.pyplot(fig)
            st.markdown('</div>', unsafe_allow_html=True)
        with d_col:
            st.markdown('<div class="main-card">', unsafe_allow_html=True)
            st.dataframe(user_data[['time', 'prediction']].tail(8), use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

# Cleanup
if os.path.exists("output.wav"): os.remove("output.wav")