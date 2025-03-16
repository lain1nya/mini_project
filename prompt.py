from fastapi import FastAPI
from nagging_graph import llm, embeddings, supervisor_executor
from models import RemarkRequest, FeedbackRequest, PriceSuggestionRequest, PriceAnalysisOutput
import json
import re
from openai import AzureOpenAI
from pprint import pprint
from db import add_remarks_to_faiss
import numpy as np
import faiss
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware
from langchain_openai import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain_community.embeddings import AzureOpenAIEmbeddings
from langchain.output_parsers import ResponseSchema, PydanticOutputParser
from pydantic import BaseModel, Field

output_parser = PydanticOutputParser(pydantic_object=PriceAnalysisOutput)

# FastAPI 앱 생성
app = FastAPI()

# CORS 미들웨어 추가
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 origin 허용
    allow_credentials=True,
    allow_methods=["*"],  # 모든 HTTP 메소드 허용
    allow_headers=["*"],  # 모든 HTTP 헤더 허용
)

# CoT + Few-shot + 템플릿 적용
def generate_prompt(remark: str):
    template = """
    다음의 기준을 사용하여 '{remark}' 잔소리의 가격을 측정하세요.

    # 평가 기준
    1. 반복 빈도 (1~10) - 자주 들을수록 높음
    2. 정신적 데미지 (1~10) - 듣기 싫을수록 높음
    3. 피할 수 있는 난이도 (1~10) - 회피 어려울수록 높음
    4. 대체 가능성 (1~10) - 영원히 사라지지 않을수록 높음

    # 참고 사항
    - 최저 가격은 1만원, 최대 가격은 15만원입니다.
    - 각 기준별로 점수와 이유를 상세히 설명해주세요.
    - 최종 가격은 각 기준의 점수를 종합적으로 고려하여 결정합니다.

    # 예시 분석
    잔소리: "너 언제 결혼하니?"
    
    사고 과정:
    1. 결혼 관련 잔소리는 특히 명절이나 가족 모임에서 자주 발생
    2. 개인의 선택과 상황을 고려하지 않는 전형적인 잔소리
    3. 결혼은 매우 개인적인 문제라 정신적 부담이 큼
    
    분석:
    - 반복 빈도: 10점 (명절, 가족 모임마다 반복되는 단골 잔소리)
    - 정신적 데미지: 9점 (개인의 상황과 무관하게 사회적 압박을 주는 발언)
    - 피할 수 있는 난이도: 9점 (가족 모임을 피하기 어려움)
    - 대체 가능성: 10점 (결혼할 때까지 계속되는 영원한 잔소리)
    
    최종 설명: 결혼 관련 잔소리는 개인의 선택을 존중하지 않고 지속적인 정신적 압박을 주는 대표적인 잔소리입니다.
    최종 가격: 15만원

    이제 '{remark}' 잔소리에 대해 위 예시와 같은 형식으로 분석해주세요.

    {format_instructions}
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    return prompt.format_messages(
        remark=remark,
        format_instructions=output_parser.get_format_instructions()
    )

def get_ai_response(prompt_messages):
    # 해석: 데이터가 흐르는 순서
    # 프롬프트 -> LLM 응답 -> 응답 파싱
    chain = prompt_messages | llm | output_parser
    
    print(f"prompt_messages: {prompt_messages}")
    try:
        # invoke 메서드 사용
        parsed_response = chain.invoke({}) # 프롬포트에 이미 모든 값이 포함되어 있음
        # PriceAnalysisOutput 모델을 사용하여 응답 검증
        analysis_output = PriceAnalysisOutput(
            thinking_steps=parsed_response["thinking_steps"],
            analysis=parsed_response["analysis"],
            final_explanation=parsed_response["final_explanation"],
            price=parsed_response["price"]
        )
        return analysis_output.final_explanation, f"{analysis_output.price}만원"
    except Exception as e:
        print(f"⚠️ 응답 파싱 오류: {e}")
        return "AI 응답을 파싱할 수 없습니다.", "가격을 파싱할 수 없습니다."

def generate_price_suggestion_prompt(base_explanation: str, positive_count: int, negative_count: int, original_price: int, suggested_price: int, reason: str) -> ChatPromptTemplate:
    # 설명 생성을 위한 출력 파서 스키마
    class ExplanationResponse(BaseModel):
        explanation: str = Field(description="잔소리에 대한 새로운 설명")
        price: int = Field(description="피드백을 반영한 가격")

    explanation_parser = PydanticOutputParser(pydantic_object=ExplanationResponse)
    
    template = """
    다음은 잔소리에 대한 기본 설명입니다:
    "{base_explanation}"

    이 잔소리에 대한 피드백 현황:
    - 긍정적 평가: {positive_count}회
    - 부정적 평가: {negative_count}회

    현재 상황:
    - 기존 가격: {original_price}만원
    - 제안된 가격: {suggested_price}만원
    - 가격 제안 이유: {reason}

    위 정보를 바탕으로 잔소리에 대한 새로운 설명과 가격을 생성해주세요.
    설명의 경우 기본 설명의 본질은 유지하면서, 사용자들의 피드백을 자연스럽게 반영해주세요.
    가격 제안과 비용에 대한 이유는 설명에 포함하지 않고, 설명은 한 문장으로 작성해주세요.
    가격의 경우 긍정적 평가의 횟수와 부정적 평가의 횟수, 기존 가격과, 제안된 가격과 이유를 모두 고려하여 적정한 가격을 책정해주세요.
    가격의 숫자는 1 ~ 15 사이로만 리턴해주세요.

    {format_instructions}
    """

    
    prompt = ChatPromptTemplate.from_template(
        template=template,
        partial_variables={"format_instructions" : explanation_parser.get_format_instructions()}
    )

    # 프롬프트 템플릿 자체를 반환
    return prompt, explanation_parser

def get_updated_explanation(prompt_template: ChatPromptTemplate, explanation_parser: PydanticOutputParser, **kwargs) -> str:

    # 새로운 방식으로 체인 구성
    chain = prompt_template | llm | explanation_parser
    
    try:
        # invoke 메서드 사용 (입력 값 없이)
        parsed_response = chain.invoke(kwargs)
        print(f"AI응답: {parsed_response}")
        return parsed_response.explanation, parsed_response.price
    except Exception as e:
        print(f"⚠️ 응답 파싱 오류: {e}")
        return None

@app.post("/get_price/")
async def get_price(request: RemarkRequest):
    print(f"입력된 잔소리: {request.remark}")
    state = {
        "remark": request.remark,
        "category": "",
        "price": 0,
        "explanation": ""
    }
    
    session_id = supervisor_executor.invoke(state)
    return {
        "session_id" : session_id,
        "result" : session_id,
        "message" : "분석이 완료되었습니다. 해당 가격이 어떤지 알려주세요."
    }

    # # 가격 출력
    # similar_results = supervisor_executor.invoke(request.remark)

    # prompt = generate_prompt(request.remark)
    # explanation, price_str = get_ai_response(prompt)
    
    # # 가격에서 '만원' 제거하고 정수로 변환
    # price = int(price_str.replace('만원', ''))
    
    # # 새로운 잔소리를 vector db에 추가
    # print("새로운 잔소리를 데이터베이스에 추가합니다.")
    # new_remarks = [{
    #     "remark": request.remark,
    #     "explanation": explanation,
    #     "price": price
    # }]
    # try:
    #     add_remarks_to_faiss(new_remarks)
    #     print("데이터베이스 추가 완료")
    # except Exception as e:
    #     print(f"데이터베이스 추가 중 오류 발생: {e}")

    # return {
    #     "new": True,
    #     "remark": request.remark,
    #     "explanation": explanation,
    #     "price": price_str
    # }

@app.post("/feedback/")
async def handle_feedback(request: FeedbackRequest):
    """프론트에서 버튼을 누르면 멈춘 Graph를 다시 실행"""
    session_id = request.session_id
    updated_state = supervisor_executor.invoke(session_id, request.dict())

    return {
        "status": "success",
        "message": "피드백이 반영되었습니다.",
        "updated_price": f"{updated_state['price']}만원"
    }
    # print(f"피드백 받음: {request}")
    
    # # 기존 데이터 검색
    # similar_results = process_remark_with_tool_calling(request.remark, top_k=1)
    # if not similar_results or len(similar_results) == 0:
    #     return {"status": "error", "message": "피드백을 줄 잔소리를 찾을 수 없습니다."}
    
    # result = similar_results[0]
    # metadata = result["metadata"]
    
    # # 기존 피드백 데이터 가져오기 (없으면 기본값 사용)
    # feedback_count = metadata.get("feedback_count", 0) + 1
    # positive_feedback = metadata.get("positive_feedback", 0)
    # negative_feedback = metadata.get("negative_feedback", 0)
    # feedback_history = metadata.get("feedback_history", [])
    
    # if request.is_positive:
    #     print("긍정적인 피드백 - 현재 가격 유지")
    #     positive_feedback += 1
    #     feedback_history.append({
    #         "is_positive": True,
    #         "timestamp": datetime.now().isoformat()
    #     })
    # else:
    #     print("부정적인 피드백 - 현재 가격 유지")
    #     negative_feedback += 1
    #     feedback_history.append({
    #         "is_positive": False,
    #         "timestamp": datetime.now().isoformat()
    #     })
    
    # # 기존 설명에서 핵심 내용 추출
    # base_explanation = metadata['explanation']
    # final_price = metadata["price"]  # 가격 유지
    
    # # 업데이트된 데이터로 새 벡터 생성 및 저장
    # new_remarks = [{
    #     "remark": request.remark,
    #     "explanation": base_explanation,
    #     "price": final_price,
    #     "feedback_count": feedback_count,
    #     "positive_feedback": positive_feedback,
    #     "negative_feedback": negative_feedback,
    #     "feedback_history": feedback_history,
    #     "last_updated": datetime.now().isoformat()
    # }]
    
    # try:
    #     add_remarks_to_faiss(new_remarks, update_existing=True)
    #     print("피드백이 데이터베이스에 반영되었습니다.")
    #     return {
    #         "status": "success", 
    #         "message": "피드백이 반영되었습니다.",
    #         "updated_price": f"{final_price}만원"
    #     }
    # except Exception as e:
    #     print(f"피드백 처리 중 오류 발생: {e}")
    #     return {"status": "error", "message": "피드백 처리 중 오류가 발생했습니다."}

@app.post("/suggest-price/")
async def suggest_price(request: PriceSuggestionRequest):
    print(f"가격 제안 받음: {request}")
    
    # 기존 데이터 검색
    similar_results = process_remark_with_tool_calling(request.remark, top_k=1)
    if not similar_results:
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

    print(f"제안 가격: {request.suggested_price}")
    
    # 가중치를 적용한 가격 계산
    weight_existing = positive_feedback / (positive_feedback + negative_feedback)
    weight_new = negative_feedback / (positive_feedback + negative_feedback)
    final_price = int(metadata["price"] * weight_existing + request.suggested_price * weight_new)
    
    # 기존 설명에서 핵심 내용 추출
    base_explanation = metadata['explanation'].split('📖 설명: ')[0].split('.')[0].strip()
    print(f"최종 가격: {final_price}")
    
    # AI를 사용하여 새로운 설명 생성
    prompt_template, explanation_parser = generate_price_suggestion_prompt(
        base_explanation,
        positive_feedback,
        negative_feedback,
        metadata["price"], 
        request.suggested_price,
        request.reason if request.reason else "이유 없음"
    )

    new_explanation, new_price = get_updated_explanation(
        prompt_template,
        explanation_parser,
        base_explanation=base_explanation,
        positive_count=positive_feedback,
        negative_count=negative_feedback,
        original_price=metadata["price"],
        suggested_price=request.suggested_price,
        reason=request.reason if request.reason else "이유 없음"
    )

    print(f"제안된 가격: {request.suggested_price}")

    if not new_explanation:
        new_explanation = f"{base_explanation}"
    
    # 업데이트된 데이터로 새 벡터 생성 및 저장
    new_remarks = [{
        "remark": request.remark,
        "explanation": new_explanation,
        "price": new_price,
        "feedback_count": feedback_count,
        "positive_feedback": positive_feedback,
        "negative_feedback": negative_feedback,
        "feedback_history": feedback_history,
        "last_updated": datetime.now().isoformat()
    }]

    print(f"새로운 설명 {new_explanation}")
    
    try:
        add_remarks_to_faiss(new_remarks, update_existing=True)
        print("가격 제안이 데이터베이스에 반영되었습니다.")
        return {
            "status": "success", 
            "message": "가격 제안이 반영되었습니다.",
            "updated_price": f"{new_price}만원"
        }
    except Exception as e:
        print(f"가격 제안 처리 중 오류 발생: {e}")
        return {"status": "error", "message": "가격 제안 처리 중 오류가 발생했습니다."}