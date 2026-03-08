"""
FastAPI Web Application for Neural Network Comparison.
Serves prediction endpoints and a professional dashboard.
"""
import os
import json
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import tensorflow as tf
from tensorflow import keras

from models import CLASS_NAMES

BASE_DIR = os.path.dirname(__file__)
MODELS_DIR = os.path.join(BASE_DIR, 'saved_models')
PLOTS_DIR = os.path.join(BASE_DIR, 'static', 'plots')

app = FastAPI(title="Panel de Comparación de Redes Neuronales")

# Mount static files
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))

# Global state
loaded_models = {}
results_data = {}


def load_models():
    """Load trained models and results."""
    global loaded_models, results_data

    results_path = os.path.join(MODELS_DIR, 'results.json')
    if not os.path.exists(results_path):
        return False

    with open(results_path, 'r') as f:
        results_data = json.load(f)

    for name in ['Dense_MLP', 'CNN', 'RNN_LSTM']:
        model_path = os.path.join(MODELS_DIR, f'{name}.keras')
        if os.path.exists(model_path):
            loaded_models[name] = keras.models.load_model(model_path)
            print(f"  Modelo cargado: {name}")

    return True


@app.on_event("startup")
async def startup():
    """Try loading models on startup."""
    print("\nIniciando servidor...")
    if load_models():
        print(f"  Mejor modelo: {results_data.get('best_model', 'N/A')}")
    else:
        print("  Modelos no encontrados. Ejecuta 'python train.py' primero.")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve main dashboard."""
    # Get list of available plot images
    plots = []
    if os.path.exists(PLOTS_DIR):
        plots = sorted([f for f in os.listdir(PLOTS_DIR) if f.endswith('.png')])

    models_ready = len(loaded_models) > 0
    best_model = results_data.get('best_model', 'N/A')
    metrics = {}
    if results_data:
        for name in ['Dense_MLP', 'CNN', 'RNN_LSTM']:
            if name in results_data:
                metrics[name] = results_data[name]['metrics']

    return templates.TemplateResponse("index.html", {
        "request": request,
        "models_ready": models_ready,
        "best_model": best_model,
        "metrics": metrics,
        "plots": plots,
        "class_names": CLASS_NAMES,
    })


@app.post("/predict")
async def predict(request: Request):
    """Predict using the best model or a specific model."""
    if not loaded_models:
        return JSONResponse({"error": "Modelos no entrenados. Ejecuta el entrenamiento primero."}, status_code=400)

    body = await request.json()
    pixel_data = body.get('pixels', [])
    model_choice = body.get('model', results_data.get('best_model', 'CNN'))

    if not pixel_data or len(pixel_data) != 784:
        return JSONResponse({"error": "Se requieren exactamente 784 valores de píxeles (28x28)."}, status_code=400)

    pixels = np.array(pixel_data, dtype='float32').reshape(1, 28, 28) / 255.0 if max(pixel_data) > 1 else np.array(pixel_data, dtype='float32').reshape(1, 28, 28)

    # Prepare data for each model
    predictions = {}
    for name, model in loaded_models.items():
        if name == 'Dense_MLP':
            x = pixels.reshape(1, 784)
        elif name == 'CNN':
            x = pixels.reshape(1, 28, 28, 1)
        else:  # RNN
            x = pixels.reshape(1, 28, 28)

        pred = model.predict(x, verbose=0)[0]
        predicted_class = int(np.argmax(pred))
        predictions[name] = {
            'class_id': predicted_class,
            'class_name': CLASS_NAMES[predicted_class],
            'confidence': float(pred[predicted_class]),
            'probabilities': {CLASS_NAMES[i]: round(float(pred[i]), 4) for i in range(len(CLASS_NAMES))},
        }

    best = model_choice if model_choice in predictions else results_data.get('best_model', list(predictions.keys())[0])

    return JSONResponse({
        "best_model": best,
        "selected_prediction": predictions[best],
        "all_predictions": predictions,
        "metrics": {n: results_data.get(n, {}).get('metrics', {}) for n in predictions},
    })


@app.get("/train-models")
async def train_models():
    """Trigger model training."""
    from train import run_training
    try:
        run_training()
        load_models()
        return JSONResponse({"status": "success", "message": "Entrenamiento completado con éxito."})
    except Exception as e:
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


@app.get("/api/results")
async def get_results():
    """Return training results as JSON."""
    if not results_data:
        return JSONResponse({"error": "No hay resultados disponibles."}, status_code=404)
    return JSONResponse(results_data)


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8500)
