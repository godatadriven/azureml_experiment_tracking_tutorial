from typing import List
from fastapi import FastAPI
from pydantic import BaseModel

import pandas as pd
import joblib
from fastapi.responses import JSONResponse

app = FastAPI()

# Pydantic models for new data points.
class TestDatapoint(BaseModel):
    x1: float = None
    x2: float = None


class PredictionRequest(BaseModel):
    data: List[TestDatapoint] = []


# The model is loaded outside the /score path.
# As we do not want to load it everytime the /score is called.
with open("models/model.joblib", "rb") as model_pickle:
    try:
        model = joblib.load(model_pickle)
    except:
        print("Model not loaded.")


@app.get("/")
async def root():
    return JSONResponse(content={"message": "Try /score for prediction."})


@app.post("/score")
async def score(prediction_request: PredictionRequest):
    df = pd.DataFrame(prediction_request.dict()["data"])
    df["y"] = model.predict(df)
    return JSONResponse(content=df.to_dict("records"))
