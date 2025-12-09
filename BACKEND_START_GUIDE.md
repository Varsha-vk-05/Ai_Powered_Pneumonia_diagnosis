# Backend Server Start Guide

## Quick Start

To fix the "Failed to fetch" error, you need to start the backend server.

### Option 1: Using the startup script (Recommended)

```bash
cd ml/server
python start_server.py
```

### Option 2: Direct start

```bash
cd ml/server
python app.py
```

### Option 3: Using uvicorn directly

```bash
cd ml/server
uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

## Verify Backend is Running

Once started, you should see:
- Server running at `http://127.0.0.1:8000`
- Model loaded successfully
- Health endpoint available at `http://127.0.0.1:8000/api/health`

Test the health endpoint:
```bash
curl http://127.0.0.1:8000/api/health
```

Or open in browser: http://127.0.0.1:8000/api/health

## Troubleshooting

### "Model not found" error
- Check that the model file exists at: `ml/runs/medmnist_pneumonia/best.pt`
- If missing, you may need to train the model first

### "Failed to fetch" in frontend
1. Make sure backend is running (check terminal for server logs)
2. Check that backend is on port 8000
3. Verify CORS is configured (should allow localhost:5173, etc.)
4. Check browser console for detailed error messages

### Import errors
- Make sure you're in the correct directory
- Install dependencies: `pip install -r ml/server/requirements.txt`
- Check Python path includes the `ml` directory

## API Endpoints

- **Health Check**: `GET http://127.0.0.1:8000/api/health`
- **Inference**: `POST http://127.0.0.1:8000/api/infer`

## Frontend Connection

The frontend is configured to connect to:
- `http://localhost:8000/api` (or `http://127.0.0.1:8000/api`)

Make sure the backend is running before using the frontend!

