# 🚀 Complete Run Guide: AI Autonomous Data Scientist Agent

Yeh guide aapko step-by-step batati hai ke **AI Autonomous Data Scientist Agent** project ko apne system par kaise install, configure aur run karna hai.

---

## 📋 Table of Contents

1. [Prerequisites (Zaroori Cheezain)](#1-prerequisites-zaroori-cheezain)
2. [Database Setup (Microsoft SQL Server)](#2-database-setup-microsoft-sql-server)
3. [Environment Configuration (.env)](#3-environment-configuration-env)
4. [Method 1: One-Click Quick Run (Subse Aasan Tareeqa)](#4-method-1-one-click-quick-run-subse-aasan-tareeqa)
5. [Method 2: Manual Terminal Execution (Step-by-Step)](#5-method-2-manual-terminal-execution-step-by-step)
   - [A. Backend Setup & Run](#a-backend-setup--run)
   - [B. Frontend Setup & Run](#b-frontend-setup--run)
6. [Access URLs & Verification](#6-access-urls--verification)
7. [Step-by-Step User Workflow (App Kaise Use Karein)](#7-step-by-step-user-workflow-app-kaise-use-karein)
8. [Troubleshooting & Common Fixes (Maslay Aur Unka Hal)](#8-troubleshooting--common-fixes-maslay-aur-unka-hal)

---

## 1. Prerequisites (Zaroori Cheezain)

Pehle check kar lein ke aapke computer par yeh softwares installed hain:

| Software | Required Version | Verification Command | Notes |
| :--- | :--- | :--- | :--- |
| **Python** | `3.10` ya `3.11+` | `python --version` | "Add Python to PATH" checked hona chahiye |
| **Node.js & npm** | `v18.x` ya `v20.x LTS` | `node -v` aur `npm -v` | Frontend Angular ke liye zaroori hai |
| **Microsoft SQL Server** | Express / Developer / 2019+ | SSMS ya Services check karein | Database tables aur runs store karne ke liye |
| **ODBC Driver 18** | Version 18 for SQL Server | ODBC Data Sources (64-bit) | Python ko SQL Server se connect karne ke liye |

> 📥 **ODBC Driver 18 Download**: Agar installed nahi hai toh Microsoft ki official site se download karein:  
> [Download Microsoft ODBC Driver 18 for SQL Server (x64)](https://learn.microsoft.com/en-us/sql/connect/odbc/download-odbc-driver-for-sql-server)

---

## 2. Database Setup (Microsoft SQL Server)

1. **SQL Server Management Studio (SSMS)** open karein ya Windows Services (`services.msc`) mein check karein ke **SQL Server (SQLEXPRESS)** running hai.
2. SSMS mein connect karein aur ek new database create karein:
   ```sql
   CREATE DATABASE AI_Autonomous_Data_Scientist_Agent;
   ```
3. Check karein aapka Server Name kya hai. Default cases mein:
   - `localhost\SQLEXPRESS`
   - Ya `YOUR_COMPUTER_NAME\SQLEXPRESS` (jaise `AWAIS\SQLEXPRESS`)
   - Ya sirf `.` ya `localhost` (agar default instance hai)

---

## 3. Environment Configuration (.env)

Project ke `backend/` folder mein file exist karti hai: `backend/.env`.  
Isko Notepad ya VS Code mein check/update karein:

```env
# ── Application ─────────────────────────────────────────────────────────────
APP_NAME="AI Autonomous Data Scientist Agent"
APP_VERSION=1.0.0
DEBUG=true
ALLOWED_ORIGINS=http://localhost:4200,http://localhost:3000,http://127.0.0.1:4200

# ── Microsoft SQL Server Connection ──────────────────────────────────────
# Agar aapka server name mukhtalif hai toh yahan change karein:
SQL_SERVER=AWAIS\SQLEXPRESS
SQL_DATABASE=AI_Autonomous_Data_Scientist_Agent
SQL_DRIVER=ODBC Driver 18 for SQL Server
SQL_TRUSTED_CONNECTION=yes
SQL_TRUST_SERVER_CERTIFICATE=yes

# ── Security (JWT Secret) ────────────────────────────────────────────────
SECRET_KEY=openrouter-autonomous-data-scientist-secret-key-2026
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=120

# ── OpenRouter LLM ───────────────────────────────────────────────────────
# LLM reasoning aur AI agent ke liye OpenRouter API key:
OPENROUTER_API_KEY=your-openrouter-api-key-here
OPENROUTER_MODEL=openai/gpt-4o-mini
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
LLM_TEMPERATURE=0.1

# ── Directories ───────────────────────────────────────────────────────────
UPLOAD_DIR=./uploads
REPORTS_DIR=./reports
MLFLOW_TRACKING_URI=./mlruns
```

> 💡 **Note**: Agar aapke paas OpenRouter key nahi hai ya expire ho jaye, system automatically deterministic mock reasoning mode par switch ho jata hai taake training aur pipelines fail na hon.

---

## 4. Method 1: One-Click Quick Run (Subse Aasan Tareeqa)

Windows users ke liye project ke main root folder mein automated batch files bani hui hain:

### Single Click Start:
Double-click karein:
```bat
start_all.bat
```
Yeh script automatically do alag-alag terminal windows open karega:
1. **Backend Window**: Virtual environment activate karega, database tables verify karega aur FastAPI uvicorn server start karega (`http://localhost:8000`).
2. **Frontend Window**: Angular development server start karega (`http://localhost:4200`).

---

## 5. Method 2: Manual Terminal Execution (Step-by-Step)

Agar aap manual command line se run karna chahte hain toh do terminals open karein:

### A. Backend Setup & Run

**Terminal 1 (PowerShell ya CMD):**

1. Project ke `backend` folder mein jayein:
   ```powershell
   cd "d:\Machine Learning Alogrithms\AI Autonomous Data Scientist Agent\backend"
   ```

2. Python Virtual Environment activate karein:
   ```powershell
   .\venv\Scripts\activate
   ```
   *(Agar venv pehle se nahi bana hua toh run karein: `python -m venv venv` aur phir activate karein)*

3. Dependencies verify ya install karein:
   ```powershell
   pip install -r requirements.txt
   ```

4. Database tables create/initialize karein:
   ```powershell
   python init_db.py
   ```
   *Expected output:* `[OK] Successfully initialized all tables in Microsoft SQL Server!`

5. FastAPI Server start karein:
   ```powershell
   uvicorn app.main:app --reload --port 8000 --host 0.0.0.0
   ```
   *Server is now live on `http://localhost:8000`*

---

### B. Frontend Setup & Run

**Terminal 2 (PowerShell ya CMD):**

1. Project ke `frontend` folder mein jayein:
   ```powershell
   cd "d:\Machine Learning Alogrithms\AI Autonomous Data Scientist Agent\frontend"
   ```

2. Node packages install karein (agar pehle nahi kiye):
   ```powershell
   npm install
   ```

3. Angular Development Server start karein:
   ```powershell
   npm start
   ```
   *(Yeh internally `ng serve` run karta hai)*

4. Terminal mein jab `Application bundle generation complete` aa jaye toh browser mein open karein:
   `http://localhost:4200`

---

## 6. Access URLs & Verification

Dono services run hone ke baad browser mein yeh URLs open karein:

| Service | URL | Purpose |
| :--- | :--- | :--- |
| **Frontend Web App** | [`http://localhost:4200`](http://localhost:4200) | Main User Dashboard & UI |
| **Backend Swagger API Docs** | [`http://localhost:8000/docs`](http://localhost:8000/docs) | Interactive API Testing & Schema |
| **Backend ReDoc** | [`http://localhost:8000/redoc`](http://localhost:8000/redoc) | Clean API Reference |
| **System Health Check** | [`http://localhost:8000/health`](http://localhost:8000/health) | DB & LLM Status Check |

### Health Check Response:
Jab aap `http://localhost:8000/health` visit karein ge toh aisa response aana chahiye:
```json
{
  "status": "online",
  "service": "AI Autonomous Data Scientist Agent",
  "version": "1.0.0",
  "database": "connected",
  "llm_model": "openai/gpt-4o-mini"
}
```

---

## 7. Step-by-Step User Workflow (App Kaise Use Karein)

Jab app run ho jaye toh complete workflow yeh hai:

```
[1. Register/Login] ──> [2. Upload Dataset] ──> [3. Start Experiment]
                                                       │
                                                       ▼
[6. Real-time Predictions] <── [5. View XAI/SHAP] <── [4. 10 Agents Run]
         │                              │
         ▼                              ▼
  [Single/Batch]             [Download PDF Report]
```

1. **Account Banayein**:
   - `http://localhost:4200` par jayein.
   - **Register** par click karein, Apna Name, Email aur Password enter karein.
   - Login karein.

2. **Dataset Upload Karein**:
   - Left sidebar se **Datasets** par jayein.
   - **Upload Dataset** button dabayein aur koi bhi `.csv` ya `.xlsx` file select karein (e.g. Titanic, Iris, Churn dataset).
   - Upload ke baad dataset ki summary, columns, missing values preview dekhein.

3. **Autonomous Experiment Run Karein**:
   - **Experiments** page par jayein -> **New Experiment**.
   - Apna uploaded dataset select karein.
   - Target Column chunein (jis cheez ko predict karna hai).
   - Problem Type select karein (Classification / Regression / Auto-detect).
   - **Start Experiment** par click karein.

4. **Multi-Agent Pipeline Live Dekhein**:
   - Screen par live status dekhein jahan 10 agents ek ek karke chalte hain:
     - Data Understanding ➔ Cleaning ➔ EDA ➔ Feature Engineering ➔ Model Selection ➔ Training (Optuna) ➔ Evaluation ➔ SHAP Explainability ➔ Report Generation.

5. **Results & Explainability**:
   - Models leaderboard dekhein (Best Model highlight hoga).
   - Accuracy, Precision, Recall, F1 Score, Confusion Matrix dekhein.
   - **Explainability** tab mein SHAP plots aur feature importance check karein.

6. **Executive PDF Report Download Karein**:
   - **Reports** tab par jayein aur **Download PDF Report** par click karein.

7. **Predictions Karein**:
   - **Predictions** section mein jayein.
   - Single row values manually enter karke live prediction lein.
   - Ya batch CSV upload karke bulk predictions download karein.

8. **AI Chat Assistant**:
   - **Chat** tab mein AI Data Scientist se sawal poochein:
     - *"What were the top features driving predictions?"*
     - *"Why was XGBoost selected over Random Forest?"*

---

## 8. Troubleshooting & Common Fixes (Maslay Aur Unka Hal)

### ❌ Issue 1: `[IM002] [Microsoft][ODBC Driver Manager] Data source name not found`
- **Wajah**: System mein Microsoft ODBC Driver 18 for SQL Server install nahi hai.
- **Hal**: [Microsoft ODBC Driver 18 x64](https://learn.microsoft.com/en-us/sql/connect/odbc/download-odbc-driver-for-sql-server) download karke install karein aur system restart/terminal reopen karein.

### ❌ Issue 2: `Cannot connect to SQL Server / Login failed`
- **Wajah**: `backend/.env` mein `SQL_SERVER` ka naam match nahi ho raha ya SQL Server service stop hai.
- **Hal**:
  1. Windows Run (`Win + R`) dabayein, `services.msc` likhein aur dekhein **SQL Server (SQLEXPRESS)** status "Running" ho.
  2. `backend/.env` open karein aur `SQL_SERVER` ko apne PC ke mutabiq set karein (e.g. `localhost\SQLEXPRESS` ya `YOUR_PC\SQLEXPRESS`).

### ❌ Issue 3: `Port 8000 ya 4200 already in use`
- **Wajah**: Pehle se koi uvicorn ya node process background mein chal raha hai.
- **Hal**:
  - PowerShell mein check & kill karein:
    ```powershell
    # Port 8000 ke liye:
    netstat -ano | findstr :8000
    taskkill /PID <PID_NUMBER> /F

    # Port 4200 ke liye:
    netstat -ano | findstr :4200
    taskkill /PID <PID_NUMBER> /F
    ```

### ❌ Issue 4: `OpenRouter 401 Unauthorized / Insufficient Quota`
- **Wajah**: API key expire ya galat hai.
- **Hal**: `backend/.env` mein `OPENROUTER_API_KEY` ko fresh key se replace karein ya default mock agent par depend karein jo offline bhi chal jata hai.

### ❌ Issue 5: `npm start` ya `ng` command not recognized
- **Hal**: Frontend folder mein `npm install` run karein aur `npx ng serve` use karein.

---

🎉 **Mubarak ho! Aapka AI Autonomous Data Scientist Agent ab puri tarah tayyar aur running hai!**
