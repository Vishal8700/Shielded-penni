import database
import model
import functions


from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
app = FastAPI()
# Define request body schema
class Item(BaseModel):
    user_count: float

# Prediction endpoint
@app.post("/predict/")
async def predict_price(item: Item):
    user_count = item.user_count
    scaled_user_count = scaler.transform([[user_count]])
    prediction = model.predict(scaled_user_count)
    return {"predicted_price": prediction[0][0]}

# Evaluation endpoint
@app.get("/evaluate/")
async def evaluate_model():
    train_loss = model.evaluate(X_train_scaled, y_train, verbose=0)
    test_loss = model.evaluate(X_test_scaled, y_test, verbose=0)
    return {"train_loss": train_loss, "test_loss": test_loss}