import json
import os

import joblib
import pandas as pd


def init():
    global model

    model_path = os.path.join(
        os.getenv("AZUREML_MODEL_DIR"),
        "model.pkl"
    )

    model = joblib.load(model_path)


def run(raw_data):
    try:
        data = json.loads(raw_data)

        input_data = pd.DataFrame(data["data"])

        predictions = model.predict(input_data)

        return {
            "predictions": predictions.tolist()
        }

    except Exception as e:
        return {
            "error": str(e)
        }