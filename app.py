from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.requests import Request

app = FastAPI()

# Templates folder
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict", response_class=HTMLResponse)
def predict(
    request: Request,
    age: int = Form(...),
    Mental_Health_Score: float = Form(...)
):
    # 🔹 Dummy prediction logic (replace with ML model)
    if Mental_Health_Score> 5:
        result = "High Social Media Usage"
    else:
        result = "Low Social Media Usage"

    return templates.TemplateResponse(
        "index.html",
        {"request": request, "prediction": result}
    )