# api_backend.py
"""
Backend API for Fin Buddy application.
Handles user authentication, data storage, and AI logic.
Run with: uvicorn api_backend:app --reload
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import bcrypt
import json
from pathlib import Path
from datetime import date
from transformers import pipeline
import uvicorn

# --- FILE PATHS (Backend manages its own data) ---
USER_FILE = Path("users.json")
TRANSACTION_FILE = Path("transactions.json")
INVESTMENT_FILE = Path("investments.json")
ACHIEVEMENTS_FILE = Path("achievements.json")

# --- FastAPI App Initialization ---
app = FastAPI()

# --- Data Models for API ---
class UserData(BaseModel):
    username: str
    password: str

class SignupData(BaseModel):
    username: str
    password: str
    confirm_password: str

class TransactionData(BaseModel):
    username: str
    date: str
    category: str
    amount: float

class InvestmentData(BaseModel):
    username: str
    name: str
    amount: float

class QueryData(BaseModel):
    username: str
    query: str

# --- Utility Functions (moved from original code) ---
def load_json(p: Path):
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except:
            return {}
    return {}

def save_json(p: Path, data):
    p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

def signup(username: str, password: str):
    users = load_json(USER_FILE)
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

def calculate_indian_tax(income: float):
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

# --- LOCAL AI SETUP (Backend handles model loading) ---
try:
    local_ai_pipeline = pipeline("text-generation", model="gpt2")
    local_ai_error = None
except Exception as e:
    local_ai_pipeline = None
    local_ai_error = f"Failed to load local AI model: {e}"

def local_ai_response(query: str):
    if not query.strip():
        return "Please type a question."
    if local_ai_pipeline:
        try:
            result = local_ai_pipeline(query, max_length=100, do_sample=True)
            return result[0]['generated_text']
        except Exception as e:
            return f"(Local AI error) {e}"
    else:
        return "Local AI is not available. Please check the terminal for model download errors."

# --- API Endpoints ---

@app.post("/api/signup")
def api_signup(data: SignupData):
    if data.password != data.confirm_password:
        raise HTTPException(status_code=400, detail="Passwords do not match.")
    ok, msg = signup(data.username, data.password)
    if not ok:
        raise HTTPException(status_code=400, detail=msg)
    return {"message": msg}

@app.post("/api/login")
def api_login(data: UserData):
    ok, msg = login_user(data.username, data.password)
    if not ok:
        raise HTTPException(status_code=401, detail=msg)
    return {"message": msg}

@app.post("/api/ai_advisor")
def api_ai_advisor(data: QueryData):
    if local_ai_error:
        raise HTTPException(status_code=500, detail=local_ai_error)
    response = local_ai_response(data.query)
    return {"response": response}

@app.post("/api/calculate_tax")
def api_calculate_tax(income: float):
    tax = calculate_indian_tax(income)
    return {"tax": tax}

# (Add more endpoints for transactions, investments, etc.)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)