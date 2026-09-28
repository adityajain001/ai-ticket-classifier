from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Hello Aditya"}

from pydantic import BaseModel, StringConstraints
from typing import Annotated
class Ticket(BaseModel):
    message: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]

from predict import predict_category
@app.post("/predict")
def predict_ticket(ticket: Ticket):
    category = predict_category(ticket.message)
    return {"category": category}

