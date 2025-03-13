from fastapi import FastAPI
from pydantic import BaseModel
import os
import json
import re
from openai import AzureOpenAI
from pprint import pprint
from dotenv import load_dotenv
from db import search_similar_remarks, add_remarks_to_faiss
import numpy as np
import faiss
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()

AOAI_ENDPOINT=os.getenv("AOAI_ENDPOINT")
AOAI_API_KEY=os.getenv("AOAI_API_KEY")
AOAI_DEPLOY_GPT4O=os.getenv("AOAI_DEPLOY_GPT4O")
AOAI_DEPLOY_GPT4O_MINI=os.getenv("AOAI_DEPLOY_GPT4O_MINI")
AOAI_DEPLOY_EMBED_3_LARGE=os.getenv("AOAI_DEPLOY_EMBED_3_LARGE")
AOAI_DEPLOY_EMBED_3_SMALL=os.getenv("AOAI_DEPLOY_EMBED_3_SMALL")
AOAI_DEPLOY_EMBED_ADA=os.getenv("AOAI_DEPLOY_EMBED_ADA")

client = AzureOpenAI(
  azure_endpoint = AOAI_ENDPOINT, 
  api_key=AOAI_API_KEY,  
  api_version="2024-10-21"
)

# FastAPI 앱 생성
app = FastAPI()

# CORS 미들웨어 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 origin 허용 (프로덕션에서는 특정 origin만 허용하는 것이 좋습니다)
    allow_credentials=True,
    allow_methods=["*"],  # 모든 HTTP 메소드 허용
    allow_headers=["*"],  # 모든 HTTP 헤더 허용
)

# 요청 모델 정의
class RemarkRequest(BaseModel):
    remark: str

class FeedbackRequest(BaseModel):
    remark: str
    is_positive: bool

class PriceSuggestionRequest(BaseModel):
    remark: str
    suggested_price: int
    reason: str

# FAISS 인덱스에서 유사한 잔소리 검색하는 함수
def retrieve_remarks(query, top_k=1):
    # FAISS 인덱스를 사용하여 유사한 잔소리 검색
    similar_results = search_similar_remarks(query, top_k)
    
    if similar_results and len(similar_results) > 0:
        # 첫 번째 결과 사용
        result = similar_results[0]
        metadata = result["metadata"]
        return metadata["remark"], metadata["explanation"], metadata["price"]
    else:
        return None, None, None


# CoT + Few-shot + 템플릿 적용
def generate_prompt(remark: str):
    return f"""
    다음의 기준을 사용하여 '{remark}' 잔소리 가격을 측정하세요:
    1. 반복 빈도 (1~10) - 자주 들을수록 높음
    2. 정신적 데미지 (1~10) - 듣기 싫을수록 높음
    3. 피할 수 있는 난이도 (1~10) - 회피 어려울수록 높음
    4. 대체 가능성 (1~10) - 영원히 사라지지 않을수록 높음

    예시:
    - "너 언제 결혼하니?"  
      → 반복 빈도: 10, 정신적 데미지: 9, 피할 수 있는 난이도: 9, 대체 가능성: 10  
      → 가격: 15만 원
    - "공부 좀 해라."  
      → 반복 빈도: 8, 정신적 데미지: 6, 피할 수 있는 난이도: 5, 대체 가능성: 7  
      → 가격: 8만 원

    먼저 각 기준을 분석한 후, 만원 단위로 최종 가격을 산출하세요.
    최저 가격은 1만원, 최대 가격은 15만원입니다.

    **예시:**
    ```json
    {{
        "explanation": "이 잔소리는 반복 빈도가 높고 정신적 데미지가 크므로 높은 가격을 매길 수 있습니다.",
        "price": 7
    }}
    ```

    이제 '{remark}' 잔소리에 대한 설명과 가격을 위의 JSON 형식으로 반환해주세요.
    """

def get_ai_response(prompt: str):
    response = client.chat.completions.create(

        model=AOAI_DEPLOY_GPT4O_MINI,
        messages=[
            {"role": "system", "content": "너는 명절 잔소리에 가격을 매기는 AI야."},
            {"role": "user", "content": prompt}
        ]
    )

    ai_response = response.choices[0].message.content
    
    json_match = re.search(r"\{.*\}", ai_response, re.DOTALL)  # `{}`로 감싸진 JSON 부분 찾기
    if json_match:
        json_text = json_match.group(0)  # JSON 문자열 추출
    else:
        print("⚠️ JSON이 올바르게 감지되지 않았습니다.")
        return "AI 응답이 JSON 형식이 아닙니다.", "가격을 파싱할 수 없습니다."

    # JSON 파싱
    try:
        result = json.loads(json_text)
        explanation = result.get("explanation", "설명이 없습니다.")
        price = result.get("price", "가격 정보 없음")
    except json.JSONDecodeError as e:
        print(f"⚠️ JSON 파싱 오류: {e}")
        explanation = "AI 응답이 JSON 형식이 아닙니다."
        price = "가격을 파싱할 수 없습니다."

    return explanation, f"{price}만원"

# TODO: 입력된 잔소리가 아닐 경우 잔소리 백터 디비에 전달하는 기준 다시 만들기
@app.post("/get_price/")
async def get_price(request: RemarkRequest):
    print(f"입력된 잔소리: {request.remark}")
    retrieved_text, retrieved_explanation, retrieved_price = retrieve_remarks(request.remark)
    if retrieved_text:
        return {
            "remark": request.remark,
            "retrieved_remark": retrieved_text,
            "explanation" : retrieved_explanation,
            "price": f"{retrieved_price}만원",
            "new": False,
        }

    prompt = generate_prompt(request.remark)
    explanation, price_str = get_ai_response(prompt)
    
    # 가격에서 '만원' 제거하고 정수로 변환
    price = int(price_str.replace('만원', ''))
    
    # 새로운 잔소리를 vector db에 추가
    print("새로운 잔소리를 데이터베이스에 추가합니다.")
    new_remarks = [{
        "remark": request.remark,
        "explanation": explanation,
        "price": price
    }]
    try:
        add_remarks_to_faiss(new_remarks)
        print("데이터베이스 추가 완료")
    except Exception as e:
        print(f"데이터베이스 추가 중 오류 발생: {e}")

    return {
        "new": True,
        "remark": request.remark,
        "explanation": explanation,
        "price": price_str
    }

def generate_price_suggestion_prompt(base_explanation: str, positive_count: int, negative_count: int, original_price: int, suggested_price: int, reason: str) -> str:
    return f"""
    다음은 잔소리에 대한 기본 설명입니다:
    "{base_explanation}"

    이 잔소리에 대한 피드백 현황:
    - 긍정적 평가: {positive_count}회
    - 부정적 평가: {negative_count}회

    현재 상황:
    - 기존 가격: {original_price}만원
    - 제안된 가격: {suggested_price}만원
    - 가격 제안 이유: {reason}

    위 정보를 바탕으로 잔소리에 대한 새로운 설명을 생성해주세요.
    기본 설명의 본질은 유지하면서, 사용자들의 피드백을 자연스럽게 반영해주세요.
    설명은 한 문장으로 작성하고, 가격 제안이 적합한지 이유와 함께 얘기해주세요.

    응답 형식:
    ```json
    {{
        "explanation": "설명 내용"
    }}
    ```
    """

def get_updated_explanation(prompt: str) -> str:
    response = client.chat.completions.create(
        model=AOAI_DEPLOY_GPT4O_MINI,
        messages=[
            {"role": "system", "content": "너는 명절 잔소리에 대한 설명을 생성하는 AI야. 사용자들의 피드백을 반영하여 자연스러운 설명을 만들어내야 해."},
            {"role": "user", "content": prompt}
        ]
    )

    ai_response = response.choices[0].message.content
    
    json_match = re.search(r"\{.*\}", ai_response, re.DOTALL)
    if json_match:
        try:
            result = json.loads(json_match.group(0))
            return f"{result['explanation']}"
        except json.JSONDecodeError as e:
            print(f"⚠️ JSON 파싱 오류: {e}")
            return None
    return None

@app.post("/feedback/")
async def handle_feedback(request: FeedbackRequest):
    print(f"피드백 받음: {request}")
    
    # 기존 데이터 검색
    similar_results = search_similar_remarks(request.remark, top_k=1)
    if not similar_results or len(similar_results) == 0:
        return {"status": "error", "message": "피드백을 줄 잔소리를 찾을 수 없습니다."}
    
    result = similar_results[0]
    metadata = result["metadata"]
    
    # 기존 피드백 데이터 가져오기 (없으면 기본값 사용)
    feedback_count = metadata.get("feedback_count", 0) + 1
    positive_feedback = metadata.get("positive_feedback", 0)
    negative_feedback = metadata.get("negative_feedback", 0)
    feedback_history = metadata.get("feedback_history", [])
    
    if request.is_positive:
        print("긍정적인 피드백 - 현재 가격 유지")
        positive_feedback += 1
        feedback_history.append({
            "is_positive": True,
            "timestamp": datetime.now().isoformat()
        })
    else:
        print("부정적인 피드백 - 현재 가격 유지")
        negative_feedback += 1
        feedback_history.append({
            "is_positive": False,
            "timestamp": datetime.now().isoformat()
        })
    
    # 기존 설명에서 핵심 내용 추출
    base_explanation = metadata['explanation']
    
    final_price = metadata["price"]  # 가격 유지
    
    # 업데이트된 데이터로 새 벡터 생성 및 저장
    new_remarks = [{
        "remark": request.remark,
        "explanation": base_explanation,
        "price": final_price,
        "feedback_count": feedback_count,
        "positive_feedback": positive_feedback,
        "negative_feedback": negative_feedback,
        "feedback_history": feedback_history,
        "last_updated": datetime.now().isoformat()
    }]
    
    try:
        add_remarks_to_faiss(new_remarks, update_existing=True)
        print("피드백이 데이터베이스에 반영되었습니다.")
        return {
            "status": "success", 
            "message": "피드백이 반영되었습니다.",
            "updated_price": f"{final_price}만원"
        }
    except Exception as e:
        print(f"피드백 처리 중 오류 발생: {e}")
        return {"status": "error", "message": "피드백 처리 중 오류가 발생했습니다."}

@app.post("/suggest-price/")
async def suggest_price(request: PriceSuggestionRequest):
    print(f"가격 제안 받음: {request}")
    
    # 기존 데이터 검색
    similar_results = search_similar_remarks(request.remark, top_k=1)
    if not similar_results or len(similar_results) == 0:
        return {"status": "error", "message": "가격을 제안할 잔소리를 찾을 수 없습니다."}
    
    result = similar_results[0]
    metadata = result["metadata"]
    
    # 기존 피드백 데이터 가져오기
    feedback_count = metadata.get("feedback_count", 0)
    positive_feedback = metadata.get("positive_feedback", 0)
    negative_feedback = metadata.get("negative_feedback", 0)
    feedback_history = metadata.get("feedback_history", [])
    
    # 가격 제안 및 이유 추가 반영
    feedback_history.append({
        "is_positive": False,
        "suggested_price": request.suggested_price,
        "reason": request.reason if request.reason else "이유 없음",
        "timestamp": datetime.now().isoformat()
    })
    
    # 가중치를 적용한 가격 계산
    weight_existing = positive_feedback / (positive_feedback + negative_feedback)
    weight_new = negative_feedback / (positive_feedback + negative_feedback)
    final_price = int(metadata["price"] * weight_existing + request.suggested_price * weight_new)
    
    # 기존 설명에서 핵심 내용 추출
    base_explanation = metadata['explanation'].split('📖 설명: ')[0].split('.')[0].strip()
    
    # AI를 사용하여 새로운 설명 생성
    prompt = generate_price_suggestion_prompt(
        base_explanation,
        positive_feedback,
        negative_feedback,
        metadata["price"], 
        request.suggested_price,
        request.reason if request.reason else "이유 없음"
    )

    new_explanation = get_updated_explanation(prompt)
    if not new_explanation:
        new_explanation = f"{base_explanation}"
    
    # 업데이트된 데이터로 새 벡터 생성 및 저장
    new_remarks = [{
        "remark": request.remark,
        "explanation": new_explanation,
        "price": final_price,
        "feedback_count": feedback_count,
        "positive_feedback": positive_feedback,
        "negative_feedback": negative_feedback,
        "feedback_history": feedback_history,
        "last_updated": datetime.now().isoformat()
    }]
    
    try:
        add_remarks_to_faiss(new_remarks, update_existing=True)
        print("가격 제안이 데이터베이스에 반영되었습니다.")
        return {
            "status": "success", 
            "message": "가격 제안이 반영되었습니다.",
            "updated_price": f"{final_price}만원"
        }
    except Exception as e:
        print(f"가격 제안 처리 중 오류 발생: {e}")
        return {"status": "error", "message": "가격 제안 처리 중 오류가 발생했습니다."}