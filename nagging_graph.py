from dotenv import load_dotenv
from faiss_db import add_remarks_to_faiss, search_similar_remark, fetch_all_remarks_from_faiss
from typing import Dict, Literal, List
from models import PriceSuggestionRequest, SupervisorState, UpdatedRemarkRequest, NaggingListResponse
import os
import json
from langgraph.graph import StateGraph, END
from langchain.schema import SystemMessage, HumanMessage
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from prompt_description import SYSTEM_MESSAGES

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

def focus_remark(state: SupervisorState) -> Dict[str, str]:
    """사용자가 말하고자 하는 의도를 정확하게 파악하는 함수"""
    structured_llm = llm.with_structured_output(UpdatedRemarkRequest)

    messages = [
        SystemMessage(content=SYSTEM_MESSAGES["updated_remark_script"]),
        HumanMessage(content=state["remark"])
    ]
    response = structured_llm.invoke(messages)
    print(response)

    state["updated_remark"] = response.updated_remark

    return {"remark" : response.remark, "updated_remark" : response.updated_remark}

# categorizer
def categorize_remark(state: SupervisorState) -> Dict[str, str]:
    """잔소리를 '명절 잔소리' 또는 '일상 잔소리'로 분류하는 함수"""
    messages = [
        SystemMessage(content=SYSTEM_MESSAGES["categorize_script"]),
        HumanMessage(content=state["updated_remark"])
    ]
    response = llm.invoke(messages)
    category = response.content.strip()

    return {"category": category if category in ["명절 잔소리", "일상 잔소리"] else "일상 잔소리"}

# searcher
def search_similar_remarks(state: SupervisorState) -> SupervisorState:
    """FAISS를 사용하여 가장 유사한 잔소리를 검색하는 함수"""
    print(f"[Searcher] 잔소리 검색 시작: {state['remark']}")

    similar_results = search_similar_remark(state["updated_remark"], state["tone"], state["category"])

    # 명절 잔소리든 일반 잔소리든 비슷한게 있을 경우
    if similar_results:
        metadata = similar_results.metadata  # 🔥 Document 객체의 metadata 가져오기
        page_content = similar_results.page_content  # 🔥 Document 객체의 텍스트 가져오기

        print(f"[Searcher] 동일 카테고리 내 유사한 잔소리 발견: {metadata}")
        print(f"similar_results: {page_content}")

        state["tone"] = metadata["tone"]
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
                key=lambda x: abs(x.get("repetition", 10) - state.get("repetition", 1)) +
                    abs(x.get("mental_damage", 10) - state.get("mental_damage", 1)) +
                    abs(x.get("avoidance_difficulty", 10) - state.get("avoidance_difficulty", 1)) +
                    abs(x.get("replaceability", 10) - state.get("replaceability", 1)),
            default=None
            ) 
    # 3️⃣ 명절 잔소리를 기반으로 새로운 일반 잔소리 생성
        if similar_holiday_remark:
            print(f"[Searcher] 유사한 명절 잔소리:  {similar_holiday_remark}")

            new_entry = {
                "remark": state["remark"],
                "updated_remark" : state["remark"],
                "tone": similar_holiday_remark["tone"],
                "category": "일상 잔소리",
                "suggested_price": str(round(int(similar_holiday_remark["suggested_price"]) / 3)),            
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

    system_message = f"""
        {SYSTEM_MESSAGES["nagging_script"]}

        {SYSTEM_MESSAGES["explanation_script"]}

        {SYSTEM_MESSAGES["updated_remark_script"]}

        {SYSTEM_MESSAGES["tone_script"]}
    """

    try:
        messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=state["updated_remark"]),
            HumanMessage(content=state["category"])
        ]

        response: PriceSuggestionRequest = structured_llm.invoke(messages)

    except Exception as e:
        print(f"⚠️ LLM 응답 파싱 오류: {e}")
        return {
            "category": "일상 잔소리",
            "suggested_price": 1,
            "explanation": "기본 설명",
            "repetition": 10,
            "mental_damage": 10,
            "avoidance_difficulty": 10,
            "replaceability": 10
        }

    print(f"[Estimator] 예측된 가격 책정 기준: {response}")

    new_entry = {
        "remark": state["remark"],
        "updated_remark" : state["updated_remark"],
        "tone" : response.tone,
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


def getNaggingList(state: NaggingListResponse) -> NaggingListResponse:
    structured_llm = llm.with_structured_output(NaggingListResponse)

    messages = [
        SystemMessage(content="""
                당신은 잔소리를 정리하는 AI입니다.  
                주어진 잔소리 목록에서 **완전히 동일한 문장은 최신 것만 유지**하고,  
                비슷한 의미를 가진 잔소리는 하나로 합쳐서 정리해주세요.  
        """),
        HumanMessage(content=json.dumps(state.model_dump(), ensure_ascii=False))
    ]
    response = structured_llm.invoke(messages)
    print("영수증이 성공적으로 생성되었습니다.")
    print(f"영수증 리스트: {response}")
    return response


graph = StateGraph(SupervisorState)
graph.add_node("focus", focus_remark)
graph.add_node("categorizer", categorize_remark)
graph.add_node("searcher", search_similar_remarks)
graph.add_node("estimator", estimate_remark_price)

def route_search_edges(state: SupervisorState) -> Literal["estimator", "end"]:
    """Searcher에서 유사한 잔소리가 발견되었는지에 따라 흐름을 결정"""
    if state.get("similar_remark"):  # 🔥 유사한 잔소리가 있다면 종료
        print(f"🔥 [Supervisor] 유사한 잔소리 발견: {state['similar_remark']} → 검색 후 종료")
        return "end"  # 🔥 더 이상 진행하지 않고 종료
    
    return "estimator"  # 🔥 유사한 잔소리가 없으면 estimator 실행

graph.add_edge("focus", "categorizer")
graph.add_edge("categorizer", "searcher")
graph.add_conditional_edges("searcher", route_search_edges, {"end" : END, "estimator": END})

graph.set_entry_point("focus")
supervisor_executor = graph.compile()