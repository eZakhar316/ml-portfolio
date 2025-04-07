import pandas as pd
import dill
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.pipeline import Pipeline
import logging

app = FastAPI()


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


try:
    with open('model/target_pipe.pkl', 'rb') as file:
        model_data = dill.load(file)
    preprocessor = model_data['preprocessor']
    model = model_data['model']
except FileNotFoundError:
    logger.error("Model file not found. Please check the path.")
    raise ValueError("Model file not found. Please check the path.")
except Exception as e:
    logger.error(f"Error loading the model: {str(e)}")
    raise ValueError(f"Error loading the model: {str(e)}")


if preprocessor is None or model is None:
    logger.error("Preprocessor or model is not loaded.")
    raise ValueError("Preprocessor or model is not loaded.")

full_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', model)
])


@app.get('/status')
def status():
    return "I'm OK"

@app.get('/version')
def version():
    if 'metadata' not in model_data:
        return {"error": "Metadata not found in the model data."}
    return model_data['metadata']

class Form(BaseModel):
    client_id: float
    visit_number: str
    utm_source: str
    utm_medium: str
    utm_campaign: str
    utm_adcontent: str
    device_category: str
    device_brand: str
    device_browser: str
    geo_country: str
    geo_city: str
    device_screen_width: int
    device_screen_height: int

class Prediction(BaseModel):
    client_id: float
    pred: float

@app.post('/predict', response_model=Prediction)
def predict(form: Form):
    try:
        df = pd.DataFrame.from_dict([form.dict()])
        y = full_pipeline.predict(df)

        if len(y) == 0:
            raise ValueError("Model returned an empty prediction.")

        return {
            'client_id': form.client_id,
            'pred': float(y[0])
        }
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))