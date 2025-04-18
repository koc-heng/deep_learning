from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from contextlib import asynccontextmanager
from .model_tool import index_label, label_chinese  
from .model import predict
from .feedback import save_feedback

# ----- 預先載模型 -----
@asynccontextmanager
async def lifespan(app: FastAPI):
    from . import model   # 模型預載（會觸發 model.py 執行）
    yield            

app = FastAPI(lifespan=lifespan)

# ---------- 註冊靜態檔案路徑與模板目錄 ----------
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# ---------- 首頁 ----------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("index.html", {
        "request": request,
        "labels": list(index_label.values()),
        "label_chinese": label_chinese  
    })

# ---------- 預測 API ----------
@app.get("/api/model/prediction")
def prediction_api(title: str):
    result = predict(title)    
    return result

# ---------- 回饋 API ----------
@app.post("/api/model/feedback")
def feedback_api(
    title: str = Form(...),
    model_label: str = Form(...),  
    confidence: float = Form(...),
    user_label: str = Form("")
):
    save_feedback(title, model_label, confidence, user_label)
    return {"message": "feedback saved"}


