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

# 요청 모델 정의
class RemarkRequest(BaseModel):
    remark: str


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