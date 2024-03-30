import uvicorn
from joblib import load
from schema import VineOutput, VineInput
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import pandas as pd
import subprocess

# Выполнить команду dvc pull
subprocess.run(["dvc", "pull"])


try:
    model = load("../model")
except FileNotFoundError:
    model = None
    reason = "File 'model.pkl' not found. Model set to None."
except Exception as e1:
    model = None
    reason = "An error occurred while loading the model:" + str(e1)


app = FastAPI()


@app.get("/healthcheck/")
def healthcheck():
    try:
        if model is not None:
            return JSONResponse(content={"message": "Service is ready"}, status_code=200)
        else:
            return JSONResponse(content={"message": f"Service is not ready. {reason}"}, status_code=501)
    except Exception as e2:
        return JSONResponse(content={"message": f"An error occurred: {e2})"}, status_code=501)


@app.post("/predict/")
def predict(data: VineInput):
    if model is None:
        return JSONResponse(content={"message": f"Service is not ready. {reason}"}, status_code=501)

    try:
        input_data = data.dict()
        input_df = pd.DataFrame(input_data, index=[0])
        quality = model.predict(input_df)[0]
        return VineOutput(predicted_quality=quality)
    except Exception as e3:
        return JSONResponse(content={"message": f"An error occurred: {e3})"}, status_code=501)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
