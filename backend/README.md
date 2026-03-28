# Drug Compatibility API - FastAPI Backend

This is the FastAPI backend for the Drug Compatibility Checker application. It provides REST API endpoints for drug search, compatibility checking, and AI-powered drug assistance.

## Prerequisites

- Python 3.9+
- pip or conda for package management

## Setup

### 1. Create a virtual environment (recommended)

```bash
cd backend
python -m venv venv

# On Windows
.\venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up environment variables

Create a `.env` file in the `backend` directory:

```env
# Required for AI chat functionality
OPENAI_API_KEY=your_openai_api_key_here

# Optional
ANTHROPIC_API_KEY=your_anthropic_api_key_here
ENABLE_OPENFDA_DATA=true
CORS_ORIGINS=http://localhost:5173,http://localhost:3000
```

### 4. Add the drug database

Copy the drug database JSON files to the `backend/data` directory:

```bash
mkdir data
cp ../drug-app/comprehensive_drug_database_compact.json ./data/
cp ../drug-app/OpenFDAfull.json ./data/  # Optional, for enhanced dosing info
```

Or copy them directly to the backend directory:

```bash
cp ../drug-app/comprehensive_drug_database_compact.json ./
```

## Running the Server

### Development mode (with auto-reload)

```bash
# From the backend directory
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Or using Python directly:

```bash
python main.py
```

### Production mode

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

## API Documentation

Once the server is running, you can access:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

## API Endpoints

### Drugs

- `GET /drugs/search?q={query}` - Search drugs by name
- `GET /drugs/info/{drug_name}` - Get detailed drug information
- `GET /drugs/categories` - Get all drug categories
- `GET /drugs/categories/{category}` - Get drugs by category
- `GET /drugs/alternatives/{drug_name}` - Get alternative drugs
- `GET /drugs/database/info` - Get database statistics

### Compatibility

- `POST /compatibility/check` - Check compatibility between two drugs
- `POST /compatibility/check-multi` - Check compatibility between multiple drugs

### AI Chat

- `POST /chat/message` - Send a message to the AI assistant
- `GET /chat/status` - Check if AI chat is available

### Health

- `GET /` - API root information
- `GET /health` - Health check endpoint

## Example Usage

### Search for drugs

```bash
curl "http://localhost:8000/drugs/search?q=aspirin"
```

### Check compatibility

```bash
curl -X POST "http://localhost:8000/compatibility/check" \
  -H "Content-Type: application/json" \
  -d '{"drug1": "Aspirin", "drug2": "Warfarin"}'
```

### AI Chat

```bash
curl -X POST "http://localhost:8000/chat/message" \
  -H "Content-Type: application/json" \
  -d '{"message": "Are Aspirin and Warfarin compatible for 3D printing?"}'
```

## Project Structure

```
backend/
├── main.py                 # FastAPI application entry point
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables (create this)
├── app/
│   ├── __init__.py
│   ├── models/             # Pydantic models
│   │   ├── __init__.py
│   │   ├── drug.py
│   │   └── compatibility.py
│   ├── routers/            # API route handlers
│   │   ├── __init__.py
│   │   ├── drugs.py
│   │   ├── compatibility.py
│   │   └── chat.py
│   ├── services/           # Business logic
│   │   ├── __init__.py
│   │   └── drug_service.py
│   └── agents/             # AI agents (from athero)
│       ├── __init__.py
│       ├── qa_agent.py
│       ├── synthesis_agent.py
│       └── publication_analyzer.py
└── data/                   # Database files (create this)
    ├── comprehensive_drug_database_compact.json
    └── OpenFDAfull.json
```

## Connecting with the Frontend

The frontend is configured to connect to `http://localhost:8000` by default. Make sure the backend is running before starting the frontend.

To use a different backend URL, set the `VITE_API_URL` environment variable in the frontend:

```bash
# In frontend/.env
VITE_API_URL=http://your-backend-url:8000
```
