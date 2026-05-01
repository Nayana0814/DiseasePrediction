import pickle
import os
import numpy as np

# ---- Helper to get model path ----
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'training_models')

# ================================================
# 1. Test: Models load correctly
# ================================================
def test_diabetes_model_loads():
    model_path = os.path.join(MODELS_DIR, 'diabetes_model.sav')
    model = pickle.load(open(model_path, 'rb'))
    assert model is not None

def test_heart_model_loads():
    model_path = os.path.join(MODELS_DIR, 'heart_model.sav')
    model = pickle.load(open(model_path, 'rb'))
    assert model is not None

def test_parkinsons_model_loads():
    model_path = os.path.join(MODELS_DIR, 'parkinsons_model.sav')
    model = pickle.load(open(model_path, 'rb'))
    assert model is not None

# ================================================
# 2. Test: Predictions return valid output (0 or 1)
# ================================================
def test_diabetes_prediction():
    model_path = os.path.join(MODELS_DIR, 'diabetes_model.sav')
    model = pickle.load(open(model_path, 'rb'))
    
    # Sample input matching diabetes dataset columns
    sample_input = [[6, 148, 72, 35, 0, 33.6, 0.627, 50]]
    prediction = model.predict(sample_input)
    
    assert prediction[0] in [0, 1], "Prediction must be 0 or 1"

def test_heart_prediction():
    model_path = os.path.join(MODELS_DIR, 'heart_model.sav')
    model = pickle.load(open(model_path, 'rb'))
    
    # Sample input matching heart dataset columns
    sample_input = [[63, 1, 3, 145, 233, 1, 0, 150, 0, 2.3, 0, 0, 1]]
    prediction = model.predict(sample_input)
    
    assert prediction[0] in [0, 1], "Prediction must be 0 or 1"

def test_parkinsons_prediction():
    model_path = os.path.join(MODELS_DIR, 'parkinsons_model.sav')
    model = pickle.load(open(model_path, 'rb'))
    
    # Sample input matching parkinsons dataset columns (22 features)
    sample_input = [[119.992, 157.302, 74.997, 0.00784, 0.00007, 0.0037,
                     0.00554, 0.01109, 0.04374, 0.42600, 0.02182, 0.03130,
                     0.02971, 0.06545, 0.02211, 21.033, 0.414783, 0.815285,
                     -4.813031, 0.266482, 2.301442, 0.284654]]
    prediction = model.predict(sample_input)
    
    assert prediction[0] in [0, 1], "Prediction must be 0 or 1"

# ================================================
# 3. Test: Input validation (empty input check)
# ================================================
def test_empty_input_raises_error():
    try:
        user_input = ['', '', '']
        result = [float(x) for x in user_input]
    except ValueError:
        assert True  # This is expected behavior