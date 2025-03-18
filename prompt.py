from fastapi import FastAPI
from nagging_graph import llm, supervisor_executor
from models import RemarkRequest, FeedbackRequest, NewReasonRequest, PriceAnalysisOutput, PriceSuggestionRequest
from fastapi.middleware.cors import CORSMiddleware
from langchain.schema import SystemMessage, HumanMessage
from langchain.output_parsers import PydanticOutputParser
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

def generate_new_suggestion_prompt(base_explanation: str, positive_count: int, negative_count: int, original_price: int, suggested_price: int, reason: str) -> PriceSuggestionRequest:
    """잔소리에 대한 새로운 설명과 가격을 생성하는 함수"""
    # 설명 생성을 위한 출력 파서 스키마
    class ExplanationResponse(BaseModel):
        explanation: str = Field(description="잔소리에 대한 새로운 설명")
        fixed_price: int = Field(description="피드백을 반영한 가격")

    structured_llm = llm.with_structured_output(ExplanationResponse)
    try :
        messages = [
            SystemMessage(content="""
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
            """), 
            HumanMessage(content=f"""
                새로운 잔소리 설명과 가격을 생성해주세요.
                
                - 기존 설명: "{base_explanation}"
                - 긍정적 평가 횟수: {positive_count}
                - 부정적 평가 횟수: {negative_count}
                - 기존 가격: {original_price}만원
                - 제안된 가격: {suggested_price}만원
                - 가격 제안 이유: "{reason}"

                새로운 설명은 기본 설명의 의미를 유지하면서도, 사용자의 피드백을 자연스럽게 반영해주세요.
                모든 한국 사람의 공감을 살 수 있을만한 설명이어야 합니다.
                설명은 한 문장으로 작성되며, 가격은 1~15만원 범위에서 적절하게 책정해주세요.
            """)
        ]

        response: ExplanationResponse = structured_llm.invoke(messages)
        return response
    
    except Exception as e:
        print(f"새로운 잔소리 생성 오류: {e}")
        return {

        }
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

# DONE
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
        remark_store[request.remark] = metadata

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
async def suggest_price(request: NewReasonRequest):
    print(f"가격 제안 받음: {request}")

    original = remark_store.get(request.remark)
    
    # 기존 데이터 검색
    similar_results = search_similar_remark(request.remark, original["category"], top_k=1)

    if not similar_results:
        return {"status": "error", "message": "가격을 제안할 잔소리를 찾을 수 없습니다."}
    
    print(f"similar results: {similar_results}")
    result = similar_results.metadata

    print(f"suggest_price result: {result}")
    # 기존 피드백 데이터 가져오기
    positive_feedback = result.get("positive_feedback", 0)
    negative_feedback = result.get("negative_feedback", 0)
    
    # 가중치를 적용한 가격 계산
    weight_existing = positive_feedback / (positive_feedback + negative_feedback)
    weight_new = negative_feedback / (positive_feedback + negative_feedback)
    final_price = int(result["suggested_price"] * weight_existing + request.suggested_price * weight_new)
    
    # 기존 설명에서 핵심 내용 추출
    base_explanation = result['explanation']
    
    print(f"최종 가격: {final_price}")
    print(f"핵심 내용: {base_explanation}")

    # explanation, fixed_price 리턴
    new_explanation_and_price = generate_new_suggestion_prompt(
        base_explanation, positive_feedback, negative_feedback,
        original["suggested_price"], request.suggested_price, request.reason if request.reason != "" else "이유 없음")

    print(f"새로 바꾼 설명: {new_explanation_and_price}")
    
    result["suggested_price"] = new_explanation_and_price.fixed_price
    result["explanation"] = new_explanation_and_price.explanation
    
    print(f"업데이트 된 데이터: {result}")

    replace_remark_in_faiss(request.remark, result["remark"], result)