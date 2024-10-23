from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()
classifier = pipeline("sentiment-analysis", model="blanchefort/rubert-base-cased-sentiment")
class Text(BaseModel):
    content: str

@app.post("/analyze")
async def analyze_sentiment(text: Text):
    result = classifier(text.content)
    return {"label": result[0]['label'], "score": result[0]['score']}

@app.get("/")
async def root():
    return {"message": "Hello World! Use POST /analyze to analyze text sentiment."}
