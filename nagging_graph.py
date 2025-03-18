from dotenv import load_dotenv
from faiss_db import add_remarks_to_faiss, search_similar_remark, fetch_all_remarks_from_faiss
from typing import Dict, Literal
from models import PriceSuggestionRequest, SupervisorState
import os
from langgraph.graph import StateGraph, END
from langchain.schema import SystemMessage, HumanMessage
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate

load_dotenv()

AOAI_ENDPOINT=os.getenv("AOAI_ENDPOINT")
AOAI_API_KEY=os.getenv("AOAI_API_KEY")
AOAI_DEPLOY_GPT4O=os.getenv("AOAI_DEPLOY_GPT4O")
AOAI_DEPLOY_GPT4O_MINI=os.getenv("AOAI_DEPLOY_GPT4O_MINI")
AOAI_DEPLOY_EMBED_3_LARGE=os.getenv("AOAI_DEPLOY_EMBED_3_LARGE")
AOAI_DEPLOY_EMBED_3_SMALL=os.getenv("AOAI_DEPLOY_EMBED_3_SMALL")
AOAI_DEPLOY_EMBED_ADA=os.getenv("AOAI_DEPLOY_EMBED_ADA")


llm = AzureChatOpenAI(
    openai_api_version="2024-10-21",
    azure_deployment=AOAI_DEPLOY_GPT4O, # 모델에 따라 적절히 선택
    azure_endpoint=AOAI_ENDPOINT,
    api_key=AOAI_API_KEY
)

# 임베딩 모델 설정
embeddings = AzureOpenAIEmbeddings(
    azure_deployment=AOAI_DEPLOY_EMBED_3_LARGE,  # 임베딩용 모델
    openai_api_version="2024-10-21",
    azure_endpoint=AOAI_ENDPOINT,
    api_key=AOAI_API_KEY
)

# categorizer
def categorize_remark(state: SupervisorState) -> Dict[str, str]:
    """잔소리를 '명절 잔소리' 또는 '일상 잔소리'로 분류하는 함수"""
    messages = [
        SystemMessage(content="""
        너는 잔소리 분석 AI야.
        사용자가 입력한 잔소리가 '명절 잔소리'인지 '일상 잔소리'인지 판단해야 해.
        '명절 잔소리'는 명절(설, 추석 등)과 관련된 것들이고, '일상 잔소리'는 평소에도 들을 수 있는 거야.
        답변은 반드시 '명절 잔소리' 또는 '일상 잔소리' 중 하나만 반환해야 해.
        """),
        HumanMessage(content=state["remark"])
    ]
    response = llm.invoke(messages)
    category = response.content.strip()

    return {"category": category if category in ["명절 잔소리", "일상 잔소리"] else "일상 잔소리"}

# searcher
def search_similar_remarks(state: SupervisorState) -> SupervisorState:
    """FAISS를 사용하여 가장 유사한 잔소리를 검색하는 함수"""
    print(f"[Searcher] 잔소리 검색 시작: {state['remark']}")

    similar_results = search_similar_remark(state["remark"], state["category"])

    # 명절 잔소리든 일반 잔소리든 비슷한게 있을 경우
    if similar_results:
        metadata = similar_results.metadata  # 🔥 Document 객체의 metadata 가져오기
        page_content = similar_results.page_content  # 🔥 Document 객체의 텍스트 가져오기

        print(f"[Searcher] 동일 카테고리 내 유사한 잔소리 발견: {metadata}")
        print(f"similar_results: {page_content}")

        state["similar_remark"] = True
        state["category"] = metadata["category"]
        state["suggested_price"] = metadata["suggested_price"]
        state["explanation"] = metadata["explanation"]

        return state

    # 🔥 비슷한 잔소리가 없을 경우 처리
    print("[Searcher] 유사한 잔소리를 찾지 못함.")

    # 비슷한 명절 잔소리가 없을 경우 바로 생성하고 VectorDB에 추가
    if state["category"] == "명절 잔소리":
        # 명절 잔소리는 새로 생성하고, 가격 예측 후 VectorDB에 추가
        new_entry = estimate_remark_price(state)  # 🔥 LLM을 호출하여 새로운 잔소리 데이터 생성
        print(f"[Searcher] 신규 명절 잔소리 추가됨: {new_entry}")

    # 비슷한 일반 잔소리가 없을 경우, 명절 잔소리의 가격 책정 기준을 기반으로 유사한 가격 찾기
    else:
        if "repetition" not in state:
            estimated_values = estimate_remark_price(state)
            state.update({
                "repetition": estimated_values["repetition"],
                "mental_damage": estimated_values["mental_damage"],
                "avoidance_difficulty": estimated_values["avoidance_difficulty"],
                "replaceability": estimated_values["replaceability"],
            })
        
        # 전체 잔소리 불러오기
        remark_metadata = fetch_all_remarks_from_faiss()
        filtered_holiday_remarks = [r for r in remark_metadata if r and r.get("category") == "명절 잔소리"]

        if not filtered_holiday_remarks:
            print("🚨 명절 잔소리가 존재하지 않음.")
            similar_holiday_remark = None  # 기본값 설정
        else:
            similar_holiday_remark = min(
                filtered_holiday_remarks,
                key=lambda x: abs(x.get("repetition", 10) - state.get("repetition", 10)) +
                    abs(x.get("mental_damage", 10) - state.get("mental_damage", 10)) +
                    abs(x.get("avoidance_difficulty", 10) - state.get("avoidance_difficulty", 10)) +
                    abs(x.get("replaceability", 10) - state.get("replaceability", 10)),
            default=None
        )

    # 3️⃣ 명절 잔소리를 기반으로 새로운 일반 잔소리 생성
    if similar_holiday_remark:
        print(f"[Searcher] 유사한 명절 잔소리를 기반으로 일반 잔소리 생성: {similar_holiday_remark}")

        new_entry = {
            "remark": state["remark"],
            "category": "일상 잔소리",
            "suggested_price": similar_holiday_remark["suggested_price"],            
            "explanation": estimated_values["explanation"],
            "repetition": state["repetition"],
            "mental_damage": state["mental_damage"],
            "avoidance_difficulty": state["avoidance_difficulty"],
            "replaceability": state["replaceability"],
            "positive_feedback": 0,
            "negative_feedback": 0,
        }

        print(new_entry)

        add_remarks_to_faiss([new_entry])

        state = new_entry

        return state

    return state



def estimate_remark_price(state: SupervisorState) -> Dict[str, int]:
    """잔소리 가격을 예측하는 함수 (새로운 잔소리를 생성하고 가격을 책정하여 반환)"""
    print(f"[Estimator] 가격 예측 시작: {state['remark']}")
    
    structured_llm = llm.with_structured_output(PriceSuggestionRequest)

    try:
        messages = [
            SystemMessage(content="""
            너는 잔소리 가격 책정 AI야.
            사용자가 입력한 잔소리에 대해 아래 기준을 예측해야 해.
            explanation은 최대한 설명을 1~2문장으로 적고, "가격을 책정했다"는 문구는 자제하는게 좋을 것 같아.
            부드러운 말투를 사용해주는게 좋을 것 같아.

            📌 가격 책정 기준:
            1. 반복 빈도 (1~20) - 자주 들을수록 높음
            2. 정신적 데미지 (1~20) - 듣기 싫을수록 높음
            3. 피할 수 있는 난이도 (1~20) - 회피 어려울수록 높음
            4. 대체 가능성 (1~20) - 영원히 사라지지 않을수록 높음

            📌 출력 형식:
            - `category`: 명절 잔소리 or 일상 잔소리
            - `suggested_price`: 예측된 최종 가격 (만원 단위, 1~15만 원)
            - `explanation`: 잔소리에 대한 AI의 최종 설명
            - `repetition`: 반복 빈도 점수 (1~20)
            - `mental_damage`: 정신적 데미지 점수 (1~20)
            - `avoidance_difficulty`: 피할 수 있는 난이도 점수 (1~20)
            - `replaceability`: 대체 가능성 점수 (1~20)
            """),
            HumanMessage(content=state["remark"])
        ]

        response: PriceSuggestionRequest = structured_llm.invoke(messages)

    except Exception as e:
        print(f"⚠️ LLM 응답 파싱 오류: {e}")
        return {
            "category": "일상 잔소리",
            "suggested_price": 5,
            "explanation": "기본 설명",
            "repetition": 10,
            "mental_damage": 10,
            "avoidance_difficulty": 10,
            "replaceability": 10
        }

    print(f"[Estimator] 예측된 가격 책정 기준: {response}")

    new_entry = {
        "remark": state["remark"],
        "category": response.category,
        "suggested_price": response.suggested_price,
        "explanation": response.explanation,
        "repetition": response.repetition,
        "mental_damage": response.mental_damage,
        "avoidance_difficulty": response.avoidance_difficulty,
        "replaceability": response.replaceability,
        "positive_feedback": 0,
        "negative_feedback": 0,
    }
    if state["category"] == "명절 잔소리":
        # 🔥 명절 잔소리 일 때만 vector db에 추가
        add_remarks_to_faiss([new_entry])

    return new_entry


graph = StateGraph(SupervisorState)
graph.add_node("categorizer", categorize_remark)
graph.add_node("searcher", search_similar_remarks)
graph.add_node("estimator", estimate_remark_price)

def route_search_edges(state: SupervisorState) -> Literal["estimator", "end"]:
    """Searcher에서 유사한 잔소리가 발견되었는지에 따라 흐름을 결정"""
    if state.get("similar_remark"):  # 🔥 유사한 잔소리가 있다면 종료
        print(f"🔥 [Supervisor] 유사한 잔소리 발견: {state['similar_remark']} → 검색 후 종료")
        return "end"  # 🔥 더 이상 진행하지 않고 종료
    
    return "estimator"  # 🔥 유사한 잔소리가 없으면 estimator 실행

graph.add_edge("categorizer", "searcher")
graph.add_conditional_edges("searcher", route_search_edges, {"end" : END, "estimator": END})
# graph.add_edge("estimator", END)

graph.set_entry_point("categorizer")
supervisor_executor = graph.compile()