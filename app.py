
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle
import joblib
import main
main = FastAPI()

model = joblib.load(open("model.pkl", "rb"))


class InputData(BaseModel):
    Age: int
    Gender: str
    Academic_Level: str
    Country: str
    Avg_Daily_Usage_Hours: float
    Most_Used_Platform: str
    Affects_Academic_Performance: str
    Sleep_Hours_Per_Night: float
    Mental_Health_Score: float
@main.post("/predict")
def predict(data: InputData):

    # Encode
    Gender = encoders["Gender"].transform([data.Gender])[0]
    Academic_Level = encoders["Academic_Level"].transform([data.Academic_Level])[0]
    Country = encoders["Country"].transform([data.Country])[0]
    Most_Used_Platform = encoders["Most_Used_Platform"].transform([data.Most_Used_Platform])[0]
    Affects_Academic_Performance = encoders["Affects_Academic_Performance"].transform([data.Affects_Academic_Performance])[0]

    # Prepare input
    input_data = np.array([[data.Age, Gender, Academic_Level, Country,
                            data.Avg_Daily_Usage_Hours,
                            Most_Used_Platform,
                            Affects_Academic_Performance,
                            data.Sleep_Hours_Per_Night,
                            data.Mental_Health_Score]])

    prediction = model.predict(input_data)
    result = encoders["Overall_Impact"].inverse_transform(prediction)

    return {"Prediction": result[0]}