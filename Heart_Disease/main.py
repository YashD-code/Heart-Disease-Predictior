from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle

# Load model and scaler
with open("Heart_Disease_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("feature_columns.json", "r") as f:
    import json
    feature_cols = json.load(f)

class InputData(BaseModel):
    age: int                     
    sex: int                     
    chest_pain_type: int         
    resting_blood_pressure: int  
    cholesterol: int             
    fasting_blood_sugar: int    
    resting_ecg: int             
    max_heart_rate: int          
    exercise_induced_angina: int 
    st_depression: float         
    st_slope: int                
    num_major_vessels: int       
    thalassemia: int  

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to the Heart Disease Classification API!"}
@app.post("/predict")
def predict(data: InputData):
    try:
        input_dict = data.dict()
        input_df = [input_dict[col] for col in feature_cols]
        scaled_input = scaler.transform([input_df])
        prediction = model.predict(scaled_input)
        return {"Heart Disease Classified":(prediction[0])}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/health")
def health_check():
    return {"status": "Model is loaded and API is healthy"}