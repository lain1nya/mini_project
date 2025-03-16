from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from nagging_graph import supervisor_executor
import uvicorn

app = FastAPI()

# CORS 미들웨어 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 origin 허용
    allow_credentials=True,
    allow_methods=["*"],  # 모든 HTTP 메소드 허용
    allow_headers=["*"],  # 모든 HTTP 헤더 허용
)

# 요청 모델
class RemarkInput(BaseModel):
    remark: str

# 응답 모델
class RemarkOutput(BaseModel):
    category: str
    price: int

@app.post("/process_remark", response_model=RemarkOutput)
def process_remark(input_data: RemarkInput):
    state = {"remark": input_data.remark, "category": "", "price": 0, "explanation" : ''}
    result = supervisor_executor.invoke(state)
    return {"category": result["category"], "price": result["price"], "explanation": result["explanation"]}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
