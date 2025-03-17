from fastapi import FastAPI
from nagging_graph import llm, embeddings, supervisor_executor
from models import RemarkRequest, FeedbackRequest, PriceSuggestionRequest, PriceAnalysisOutput
from openai import AzureOpenAI
from pprint import pprint
from langchain_community.vectorstores import FAISS
from datetime import datetime
from fastapi.middleware.cors import CORSMiddleware
from langchain_openai import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain
from langchain_community.embeddings import AzureOpenAIEmbeddings
from langchain.output_parsers import ResponseSchema, PydanticOutputParser
from pydantic import BaseModel, Field
from faiss_db import replace_remark_in_faiss, search_similar_remark, add_remarks_to_faiss

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

# 피드백을 적용할 데이터를 저장하는 딕셔너리
remark_store = {}

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

    # 기존에 explanation 업데이트 한 것을 langgraph에 추가
    # check point, human in the loop
    # 별도의 그래프
    # 싫어요 배치

    # langchain_community.vectorstores

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

# DONE
@app.post("/get_price/")
async def get_price(request: RemarkRequest):
    print(f"입력된 잔소리: {request.remark}")
    state = {
        "remark": request.remark,
        "category": "",
        "suggested_price": 0,
        "explanation": "",
    }
     
    result = supervisor_executor.invoke(state)

    remark_store[request.remark] = result

    return {
        "result" : result,
        "message" : "분석이 완료되었습니다. 해당 가격이 어떤지 알려주세요."
    }

@app.post("/feedback/")
async def handle_feedback(request: FeedbackRequest):
    """FAISS에서 유사한 잔소리를 찾고, 존재하면 대체하고 없으면 추가"""

    # 🔍 remark_store에서 해당 잔소리 정보 가져오기
    remark_data = remark_store.get(request.remark)

    if not remark_data:
        return {"status": "error", "message": "잔소리 데이터가 존재하지 않습니다."}

    print("🔍 Stored remark result:", remark_data)

    # 🔥 category 필드에 안전하게 접근
    category = remark_data["category"]

    # 🔍 1️⃣ FAISS에서 유사한 remark 검색 (카테고리 필터링 가능)
    similar_results = search_similar_remark(request.remark, category)  # 🔥 같은 카테고리 내에서 검색

    print(similar_results)

    if similar_results:
        metadata = similar_results.metadata  # 🔥 Document 객체의 metadata 가져오기
        page_content = similar_results.page_content

        print(f"🔍 기존 remark 발견: {page_content} ({metadata})")

        # 🔥 2️⃣ 기존 remark의 피드백 업데이트
        if request.is_positive:
            metadata["positive_feedback"] = metadata.get("positive_feedback", 0) + 1
        else:
            metadata["negative_feedback"] = metadata.get("negative_feedback", 0) + 1

        print(f"✅ 업데이트된 피드백: {metadata}")

        # 🔥 3️⃣ 기존 remark를 새로운 remark로 대체
        replace_remark_in_faiss(original_remark=page_content, new_remark=request.remark, updated_metadata=metadata)

        return {
            "status": "success",
            "message": "유사한 잔소리를 찾아 대체했습니다.",
            "updated_remark": metadata
        }
    
    # 🔥 4️⃣ 유사한 remark가 없으면 새로운 remark 추가
    print("[Searcher] 유사한 잔소리를 찾지 못함. 새로운 remark 추가.")
    new_entry = {
        "remark": request.remark,
        "category": "일상 잔소리",  # 🔥 기본값 (명절 잔소리일 수도 있음, 필요 시 변경)
        "suggested_price": 10,  # 🔥 기본값 (LLM을 활용해 결정 가능)
        "explanation": "이 잔소리는 새로운 항목으로 추가되었습니다.",
        "repetition": 10,
        "mental_damage": 10,
        "avoidance_difficulty": 10,
        "replaceability": 10,
        "positive_feedback": 1 if request.is_positive else 0,
        "negative_feedback": 1 if not request.is_positive else 0
    }

    add_remarks_to_faiss([new_entry])  # 🔥 새로운 remark 추가

    return {
        "status": "success",
        "message": "새로운 잔소리를 추가했습니다.",
        "new_remark": new_entry
    }


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