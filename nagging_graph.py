from typing import Literal, TypedDict, Dict, Annotated
from dotenv import load_dotenv
import os
import faiss
import json
import numpy as np
from langgraph.graph import StateGraph
from langchain.schema import SystemMessage, HumanMessage
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

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
    azure_deployment=AOAI_DEPLOY_GPT4O,  # 모델에 따라 적절히 선택
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

index_path = "faiss_index"
metadata_path = "remark_metadata.json"
dim = 3072
index = faiss.IndexFlatL2(dim)

# Supervisor를 위한 상태 정의
class SupervisorState(TypedDict):
    remark: Annotated[str, "single"]
    category: Literal["명절 잔소리", "일상 잔소리"]
    price: int

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
    print(f"[Categorizer] 분류 결과: {category}")

    return {"category": category if category in ["명절 잔소리", "일상 잔소리"] else "일상 잔소리"}

# searcher
def search_similar_remarks(state: SupervisorState) -> Dict[str, str]:
    """FAISS를 사용하여 가장 유사한 잔소리를 검색하는 함수"""
    print(f"[Searcher] 잔소리 검색 시작: {state['remark']}")

    if not os.path.exists(index_path) or not os.path.exists(metadata_path):
        print("[Searcher] FAISS 인덱스 파일이 존재하지 않음. 기본값 반환.")
        return {"similar_remark": ""}
    
    index = faiss.read_index(index_path)
    with open(metadata_path, "r", encoding="utf-8") as f:
        remark_metadata = json.load(f)
    
    query_embedding = embeddings.embed_query(state["remark"])
    distances, indices = index.search(np.array([query_embedding]).astype("float32"), 1)

    similar_remark = remark_metadata[indices[0][0]]["remark"] if 0 <= indices[0][0] < len(remark_metadata) else ""

    print(f"[Searcher] 검색된 유사 잔소리: {similar_remark}")
    
    return {"similar_remark": similar_remark}

# estimator
def estimate_remark_price(state: SupervisorState) -> Dict[str, int]:
    """잔소리 가격을 예측하는 함수 (유사한 잔소리의 가격을 참고하여 결정)"""
    print(f"[Estimator] 가격 예측 시작: {state['remark']}")

    if not os.path.exists(index_path) or not os.path.exists(metadata_path):
        print("[Estimator] FAISS 인덱스 파일이 없음. 기본 가격 5만원 설정.")
        return {"price": 5}
    
    messages = [
        SystemMessage(content="""
        너는 잔소리 가격 책정 AI야.
        사용자가 입력한 잔소리에 대해 아래 기준을 예측해야 해.
        
        가격 책정 기준:
        1. 반복 빈도 (1~20) - 자주 들을수록 높음
        2. 정신적 데미지 (1~20) - 듣기 싫을수록 높음
        3. 피할 수 있는 난이도 (1~20) - 회피 어려울수록 높음
        4. 대체 가능성 (1~20) - 영원히 사라지지 않을수록 높음
        
        참고 사항
        - 최저 가격은 1만원, 최대 가격은 15만원입니다.
        - 각 기준별로 점수와 이유를 상세히 설명해주세요.
        - 최종 가격은 각 기준의 점수를 종합적으로 고려하여 결정합니다.

        예시 분석
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
                      
        예시 출력:
        - 최종 설명: 결혼 관련 잔소리는 개인의 선택을 존중하지 않고 지속적인 정신적 압박을 주는 대표적인 잔소리입니다.
        - 최종 가격: 15만원
        - 반복 빈도: 17
        - 정신적 데미지: 15
        - 피할 수 있는 난이도: 12
        - 대체 가능성: 14
        """),
        HumanMessage(content=state["remark"])
    ]

    response = llm.invoke(messages)
    extracted_values = response.content.strip().split("\n")

    predicted_values = {
        "repetition": int(extracted_values[0].split(": ")[1]),
        "mental_damage": int(extracted_values[1].split(": ")[1]),
        "avoidance_difficulty": int(extracted_values[2].split(": ")[1]),
        "replaceability": int(extracted_values[3].split(": ")[1])
    }
    
    print(f"[Estimator] 예측된 가격 책정 기준: {predicted_values}")
    
    index = faiss.read_index(index_path)
    with open(metadata_path, "r", encoding="utf-8") as f:
        remark_metadata = json.load(f)
    
    query_vector = np.array([
        predicted_values["repetition"],
        predicted_values["mental_damage"],
        predicted_values["avoidance_difficulty"],
        predicted_values["replaceability"]
    ]).astype("float32").reshape(1, -1)

    distance, indices = index.search(query_vector, 3)

    similar_prices = [
        remark_metadata[idx]["price"] for idx in indices[0] if 0 <= idx < len(remark_metadata)
    ]

    predicted_price = int(np.mean(similar_prices)) if similar_prices else 5

    print(f"[Estimator] 예측된 잔소리 가격: {predicted_price}만원")

    return {"price": predicted_price}

graph = StateGraph(SupervisorState)
graph.add_node("categorizer", categorize_remark)
graph.add_node("searcher", search_similar_remarks)
graph.add_node("estimator", estimate_remark_price)

graph.add_edge("categorizer", "estimator")
graph.add_edge("categorizer", "searcher")
graph.add_edge("searcher", "estimator")

def route_edges(state: SupervisorState) -> str:
    """categorizer에서 다음 노드를 결정"""
    next_step = "estimator" if state["category"] == "명절 잔소리" else "searcher"
    print(f"[Supervisor] Categorizer 결과: {state['category']} -> 다음 단계: {next_step}")
    return next_step

def route_search_edges(state: SupervisorState) -> str:
    """searcher에서 다음 노드를 결정"""
    next_step = "estimator" if state.get("similar_remark") else "output"
    print(f"[Supervisor] Searcher 결과: {state.get('similar_remark')} -> 다음 단계: {next_step}")
    return next_step

graph.add_conditional_edges("categorizer", route_edges)
graph.add_conditional_edges("searcher", route_search_edges)

graph.set_entry_point("categorizer")
supervisor_executor = graph.compile()

# TODO: 카테고리 명절인지 아닌지도 구분해야 할까...?