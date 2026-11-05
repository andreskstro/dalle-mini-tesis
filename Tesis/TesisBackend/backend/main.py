from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Prompt(BaseModel):
    text: str

@app.post("/generate")
async def generate(prompt: Prompt):
    dummy_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAUA" \
                   "AAAFCAYAAACNbyblAAAAHElEQVQI12P4" \
                   "//8/w38GIAXDIBKE0DHxgljNBAAO" \
                   "9TXL0Y4OHwAAAABJRU5ErkJggg=="
    return {"images": [dummy_base64]}
