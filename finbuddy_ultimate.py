"""
Fin Buddy - The ultimate single-file Streamlit app.
Features:
 - Login / Signup (bcrypt)
 - AI Advisor (local transformer model with speech & vision)
 - Transactions & Investments (per-user JSON)
 - Tax calculator (Indian slabs simplified)
 - Gamification: XP / Levels + Achievements
 - Futuristic UI styling
"""

import streamlit as st
import json
from pathlib import Path
from datetime import date, datetime
import pandas as pd
import matplotlib.pyplot as plt
import bcrypt
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import io
from gtts import gTTS

# -------------------- FILE PATHS --------------------
USER_FILE = Path("users.json")
TRANSACTION_FILE = Path("transactions.json")
INVESTMENT_FILE = Path("investments.json")
ACHIEVEMENTS_FILE = Path("achievements.json")

# -------------------- UI STYLES --------------------
CSS = """
<style>
body { background: linear-gradient(135deg, #0f1724 0%, #071730 100%); color: #e6f7ff; }
.header { text-align:center; padding:10px; }
.card { background: rgba(255,255,255,0.03); padding:12px; border-radius:12px; margin-bottom:10px; }
.metric { background: linear-gradient(90deg,#1e293b,#0f1724); padding:8px; border-radius:8px; text-align:center; }
.small { font-size:0.9rem; color:#cfeffd; }
.badge { background:#ffd166; color:#04111d; padding:6px 10px; border-radius:999px; font-weight:700; }
.compact-label { margin-bottom: -15px; }
h5 { margin-top: 5px; margin-bottom: 5px; }
.stTextInput { margin-top: 0; margin-bottom: 0; }
.stMarkdown h3 { margin-bottom: 5px; }
</style>
"""

st.set_page_config(page_title="Fin Buddy", layout="wide")
st.markdown(CSS, unsafe_allow_html=True)
st.markdown("<div class='header'><h1>🎮 <span style='color:#ffd166'>Fin Buddy</span> — Local AI Finance Game</h1></div>", unsafe_allow_html=True)

# -------------------- UTILITIES --------------------

def load_json(p: Path):
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def save_json(p: Path, data):
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

for f in [USER_FILE, TRANSACTION_FILE, INVESTMENT_FILE, ACHIEVEMENTS_FILE]:
    if not f.exists():
        save_json(f, {})

# -------------------- AUTH --------------------
def signup(username: str, password: str):
    users = load_json(USER_FILE)
    if not username or not password:
        return False, "Username and password cannot be empty."
    if username in users:
        return False, "Username already exists."
    hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    users[username] = {"pw": hashed, "created": str(date.today())}
    save_json(USER_FILE, users)
    return True, "Signup successful. Please login."

def login_user(username: str, password: str):
    users = load_json(USER_FILE)
    if username not in users:
        return False, "Username not found."
    stored = users[username]["pw"].encode()
    if bcrypt.checkpw(password.encode(), stored):
        return True, "Login success."
    else:
        return False, "Incorrect password."

# -------------------- LOCAL AI & NEW FEATURES SETUP --------------------
if 'local_ai_pipeline' not in st.session_state:
    try:
        st.info("Downloading and loading ibm-granite/granite-3.3-2b-instruct model...")
        st.session_state.local_ai_pipeline = pipeline(
            "text-generation",
            model="ibm-granite/granite-3.3-2b-instruct",
            device=-1,
        )
        st.session_state.local_ai_error = None
    except Exception as e:
        st.session_state.local_ai_pipeline = None
        st.session_state.local_ai_error = f"Failed to load local AI model: {e}"

if "stt_pipe" not in st.session_state:
    try:
        st.session_state.stt_pipe = pipeline("automatic-speech-recognition", model="openai/whisper-tiny", device=-1)
    except Exception:
        st.session_state.stt_pipe = None

if "ocr_pipe" not in st.session_state:
    try:
        st.session_state.ocr_pipe = pipeline("image-to-text", model="microsoft/trocr-base", device=-1)
    except Exception:
        st.session_state.ocr_pipe = None

def local_ai_response(query: str) -> str:
    if not query or not query.strip():
        return "Please type a question."
    if st.session_state.local_ai_pipeline is None:
        return st.session_state.local_ai_error or "Local AI is not available."

    prompt = (
        "You are FinBuddy, a helpful financial advisor. "
        "Answer the user's question clearly, briefly, and only about personal finance.\n"
        f"User: {query}\nFinBuddy:"
    )

    try:
        result = st.session_state.local_ai_pipeline(
            prompt,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
        generated = result[0].get("generated_text", "")
        answer = generated.split("FinBuddy:")[-1].strip()
        return answer
    except Exception as e:
        return f"(Local AI error) {e}"

def text_to_audio(text: str):
    try:
        tts = gTTS(text, lang='en', slow=False)
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        return fp
    except Exception as e:
        st.error(f"Text-to-audio error: {e}")
    return None

# -------------------- TRANSACTIONS & INVESTMENTS --------------------
def add_transaction(username: str, d: str, category: str, amount: float):
    txs = load_json(TRANSACTION_FILE)
    user_tx = txs.get(username, [])
    user_tx.append([str(d), category, float(amount)])
    txs[username] = user_tx
    save_json(TRANSACTION_FILE, txs)

def get_transactions_df(username: str) -> pd.DataFrame:
    txs = load_json(TRANSACTION_FILE).get(username, [])
    if not txs:
        return pd.DataFrame(columns=["Date", "Category", "Amount"])
    df = pd.DataFrame(txs, columns=["Date", "Category", "Amount"])
    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce").fillna(0.0)
    return df

def add_investment(username: str, name: str, amount: float):
    invs = load_json(INVESTMENT_FILE)
    user_inv = invs.get(username, [])
    user_inv.append([name, float(amount)])
    invs[username] = user_inv
    save_json(INVESTMENT_FILE, invs)

def get_investments_df(username: str) -> pd.DataFrame:
    inv = load_json(INVESTMENT_FILE).get(username, [])
    if not inv:
        return pd.DataFrame(columns=["Investment", "Amount"])
    df = pd.DataFrame(inv, columns=["Investment", "Amount"])
    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce").fillna(0.0)
    return df

# -------------------- TAX --------------------
def calculate_indian_tax(income: float) -> float:
    tax = 0.0
    if income <= 250000:
        tax = 0
    elif income <= 500000:
        tax = (income - 250000) * 0.05
    elif income <= 1000000:
        tax = 12500 + (income - 500000) * 0.2
    else:
        tax = 112500 + (income - 1000000) * 0.3
    return tax

# -------------------- GAMIFICATION --------------------
LEVELS = ["Rookie", "Learner", "Explorer", "Strategist", "Investor", "Expert", "Top Professionalist"]

def compute_xp(username: str) -> int:
    tx_df = get_transactions_df(username)
    inv_df = get_investments_df(username)
    tx_xp = int(tx_df["Amount"].sum() // 100) if not tx_df.empty else 0
    inv_xp = int(inv_df["Amount"].sum() // 100) if not inv_df.empty else 0
    ach = load_json(ACHIEVEMENTS_FILE).get(username, {"badges": [], "chat_count": 0})
    chat_xp = int(ach.get("chat_count", 0)) * 5
    return tx_xp + inv_xp + chat_xp

def get_level_from_xp(xp: int):
    thresholds = [0, 100, 400, 1200, 3000, 7000, 15000]
    idx = 0
    for i, t in enumerate(thresholds):
        if xp >= t:
            idx = i
    idx = min(idx, len(LEVELS) - 1)
    return idx, LEVELS[idx], thresholds[idx]

def award_achievements(username: str):
    ach_store = load_json(ACHIEVEMENTS_FILE)
    user_ach = ach_store.get(username, {"badges": [], "chat_count": 0})
    tx_df = get_transactions_df(username)
    inv_df = get_investments_df(username)
    if len(tx_df) >= 1 and "First Transaction" not in user_ach["badges"]:
        user_ach["badges"].append("First Transaction")
    if inv_df["Amount"].sum() >= 50000 and "Investor Badge" not in user_ach["badges"]:
        user_ach["badges"].append("Investor Badge")
    if len(tx_df) >= 5 and "Active User" not in user_ach["badges"]:
        user_ach["badges"].append("Active User")
    ach_store[username] = user_ach
    save_json(ACHIEVEMENTS_FILE, ach_store)

# -------------------- SESSION STATE --------------------
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = ""
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'messages' not in st.session_state:
    st.session_state.messages = []

# -------------------- SIDEBAR --------------------
with st.sidebar:
    st.markdown("## Menu")
    if not st.session_state.logged_in:
        page = st.selectbox("Go to", ["Login / Signup", "About"])
    else:
        page = st.selectbox("Go to", ["Home", "AI Advisor", "Transactions", "Investments", "Tax Calculator", "Levels & Badges", "Logout"])
    st.markdown("---")
    st.markdown("**Daily Mission**")
    missions = ["Add a transaction today", "Ask AI Advisor a question", "Add an investment"]
    today_mission = missions[datetime.today().weekday() % len(missions)]
    st.info(f"🎯 {today_mission}")

# -------------------- PAGES --------------------
def page_login_signup():
    st.markdown("<div class='card'><h3>🔐 Login or Create Account</h3></div>", unsafe_allow_html=True)
    choice = st.radio("Please select an option:", ["Login", "Signup"])
    st.markdown("---")
    if choice == "Login":
        with st.form("login_form"):
            st.subheader("Login")
            st.markdown('<h5 style="margin: 0;">Username</h5>', unsafe_allow_html=True)
            lu = st.text_input("", key="login_user")
            st.markdown('<h5 style="margin: 0;">Password</h5>', unsafe_allow_html=True)
            lp = st.text_input("", type="password", key="login_pw")
            submit_login = st.form_submit_button("Login")
        if submit_login:
            ok, msg = login_user(lu.strip(), lp)
            if ok:
                st.session_state.logged_in = True
                st.session_state.username = lu.strip()
                st.success("Logged in ✅")
                award_achievements(st.session_state.username)
                st.rerun()
            else:
                st.error(msg)
    elif choice == "Signup":
        with st.form("signup_form"):
            st.subheader("Signup")
            st.markdown('<h5 style="margin: 0;">New Username</h5>', unsafe_allow_html=True)
            su = st.text_input("", key="signup_user")
            st.markdown('<h5 style="margin: 0;">New Password</h5>', unsafe_allow_html=True)
            sp = st.text_input("", type="password", key="signup_pw")
            st.markdown('<h5 style="margin: 0;">Confirm Password</h5>', unsafe_allow_html=True)
            cp = st.text_input("", type="password", key="confirm_pw")
            submit_signup = st.form_submit_button("Create account")
        if submit_signup:
            if sp == cp:
                ok, msg = signup(su.strip(), sp)
                if ok:
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)
            else:
                st.error("Passwords do not match.")

def page_about():
    st.markdown("<div class='card'><h3>About Fin Buddy</h3></div>", unsafe_allow_html=True)
    st.write("Fin Buddy is a gamified personal finance assistant with local AI, speech, and vision.")

def page_home():
    st.markdown("<div class='card'><h3>🏠 Dashboard</h3></div>", unsafe_allow_html=True)
    username = st.session_state.username
    tx_df = get_transactions_df(username)
    inv_df = get_investments_df(username)
    balance = tx_df["Amount"].sum() if not tx_df.empty else 0.0
    invest_total = inv_df["Amount"].sum() if not inv_df.empty else 0.0
    xp = compute_xp(username)
    _, level_name, lvl_threshold = get_level_from_xp(xp)

    c1, c2, c3 = st.columns(3)
    c1.markdown(f"<div class='metric'><h3>💰 Balance</h3><div class='small'>₹{balance:,.2f}</div></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='metric'><h3>📊 Investments</h3><div class='small'>₹{invest_total:,.2f}</div></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='metric'><h3>🏷️ Level</h3><div class='small'>{level_name} ({xp} XP)</div></div>", unsafe_allow_html=True)

    progress = min(max((xp - lvl_threshold) / max(1, (15000 - lvl_threshold)), 0.0), 1.0)
    st.progress(progress)

    st.markdown("### Recent Transactions")
    st.dataframe(tx_df.tail(10))

    st.markdown("### Investments")
    st.dataframe(inv_df)

    if not inv_df.empty:
        fig, ax = plt.subplots()
        ax.pie(inv_df["Amount"], labels=inv_df["Investment"], autopct='%1.1f%%', startangle=90)
        st.pyplot(fig)

def page_ai_advisor():
    st.markdown("<div class='card'><h3>🤖 AI Advisor (Local + Voice + Vision)</h3></div>", unsafe_allow_html=True)
    if st.session_state.local_ai_pipeline is None:
        st.error(st.session_state.local_ai_error)
        st.info("The local AI model could not be loaded. Please check your terminal for more details.")
        return

    # Chat with Advisor
    st.markdown("### Chat with Advisor")
    user_q = st.text_input("💬 Ask a finance question:", key="ai_input")

    if st.button("Send to Advisor"):
        ans = local_ai_response(user_q)
        st.session_state.messages.append(("You", user_q))
        st.session_state.messages.append(("FinBuddy", ans))
        # track chat_count
        ach_store = load_json(ACHIEVEMENTS_FILE)
        u_ach = ach_store.get(st.session_state.username, {"badges": [], "chat_count": 0})
        u_ach["chat_count"] = u_ach.get("chat_count", 0) + 1
        ach_store[st.session_state.username] = u_ach
        save_json(ACHIEVEMENTS_FILE, ach_store)
        award_achievements(st.session_state.username)
        st.rerun()

    # Display chat history
    if st.session_state.messages:
        for who, msg in reversed(st.session_state.messages[-10:]):
            st.markdown(f"**{who}:** {msg}")
            # 🔊 Speak last answer
            if who == "FinBuddy":
                audio_data = text_to_audio(msg)
                if audio_data:
                    st.audio(audio_data.getvalue(), format='audio/wav')
                else:
                    st.info("Text-to-speech failed. Please check the terminal for errors.")

    # Speech and Vision features
    st.markdown("---")
    st.subheader("🎤 Speech → Text")
    audio_file = st.file_uploader("Upload audio", type=["wav", "mp3"])
    if st.button("🎙️ Convert Speech"):
        if audio_file and st.session_state.stt_pipe:
            result = st.session_state.stt_pipe(audio_file.read())
            st.success(result["text"])
            st.session_state.ai_input = result["text"]
            st.rerun()
        else:
            st.warning("Upload audio and ensure STT model loaded.")

    st.markdown("---")
    st.subheader("📷 Image → Text")
    image_file = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"])
    if st.button("📷 Extract Text"):
        if image_file and st.session_state.ocr_pipe:
            result = st.session_state.ocr_pipe(image_file.read())
            st.success(result[0]['generated_text'])
            st.rerun()
        else:
            st.warning("Upload image and ensure OCR model loaded.")

def page_transactions():
    st.markdown("<div class='card'><h3>💳 Transactions</h3></div>", unsafe_allow_html=True)
    with st.form("tx_form"):
        d = st.date_input("Date", date.today())
        cat = st.text_input("Category", placeholder="Food / Salary / Rent")
        amt = st.number_input("Amount (₹)", min_value=0.0, step=1.0)
        submit = st.form_submit_button("Add Transaction")
        if submit:
            add_transaction(st.session_state.username, d, cat, amt)
            st.success("Transaction added.")
            award_achievements(st.session_state.username)
            st.rerun()
    st.markdown("#### Your Transactions")
    st.dataframe(get_transactions_df(st.session_state.username))

def page_investments():
    st.markdown("<div class='card'><h3>📈 Investments</h3></div>", unsafe_allow_html=True)
    with st.form("inv_form"):
        name = st.text_input("Investment name", placeholder="Mutual Fund / Stock / Gold")
        amt = st.number_input("Amount (₹)", min_value=0.0, step=1.0)
        submit = st.form_submit_button("Add Investment")
        if submit:
            add_investment(st.session_state.username, name, amt)
            st.success("Investment added.")
            award_achievements(st.session_state.username)
            st.rerun()
    st.markdown("#### Your Investments")
    inv_df = get_investments_df(st.session_state.username)
    st.dataframe(inv_df)
    if not inv_df.empty:
        fig, ax = plt.subplots()
        ax.pie(inv_df["Amount"], labels=inv_df["Investment"], autopct='%1.1f%%', startangle=90)
        st.pyplot(fig)

def page_tax():
    st.markdown("<div class='card'><h3>🧾 Tax Calculator</h3></div>", unsafe_allow_html=True)
    with st.form("tax_form"):
        income = st.number_input("Annual income (₹)", min_value=0.0, step=1000.0)
        deductions = st.number_input("Deductions (₹)", min_value=0.0, step=1000.0)
        submit_tax = st.form_submit_button("Calculate Tax")
    if submit_tax:
        tax = calculate_indian_tax(income - deductions)
        st.success(f"Estimated Tax (simplified): ₹{tax:,.2f}")

def page_levels_badges():
    st.markdown("<div class='card'><h3>🏆 Levels & Achievements</h3></div>", unsafe_allow_html=True)
    xp = compute_xp(st.session_state.username)
    idx, name, thr = get_level_from_xp(xp)
    st.markdown(f"### Level: **{name}** (XP: {xp})")
    st.progress(min(xp / 15000, 1.0))
    ach_store = load_json(ACHIEVEMENTS_FILE)
    user_ach = ach_store.get(st.session_state.username, {"badges": [], "chat_count": 0})
    badges = user_ach.get("badges", [])
    if badges:
        cols = st.columns(3)
        for i, b in enumerate(badges):
            cols[i % 3].markdown(f"<div class='badge'>{b}</div>", unsafe_allow_html=True)
    else:
        st.info("No badges yet. Do transactions, invest, and chat with AI to earn badges!")

# -------------------- MAIN ROUTER --------------------
if not st.session_state.logged_in:
    if page == "Login / Signup":
        page_login_signup()
    else:
        page_about()
else:
    if page == "Home":
        page_home()
    elif page == "AI Advisor":
        page_ai_advisor()
    elif page == "Transactions":
        page_transactions()
    elif page == "Investments":
        page_investments()
    elif page == "Tax Calculator":
        page_tax()
    elif page == "Levels & Badges":
        page_levels_badges()
    elif page == "Logout":
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.success("Logged out. Refresh to continue.")
        st.rerun()

# -------------------- FOOTER / NOTES --------------------
st.markdown("---")
st.markdown("<div class='small'>Fin Buddy • Local demo. Local AI features require 'transformers' installed. Data stored in JSON files in app folder.</div>", unsafe_allow_html=True)
