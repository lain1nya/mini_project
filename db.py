import faiss
import numpy as np
from openai import AzureOpenAI
import os
import json
import pprint
from dotenv import load_dotenv
from typing import List
from models import PriceSuggestionRequest
from pydantic import Field, BaseModel
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema, PydanticOutputParser
from langchain.schema import SystemMessage, HumanMessage
from langchain.tools import tool
from langchain.agents import create_tool_calling_agent, AgentType
from langchain.agents import AgentExecutor, initialize_agent, OpenAIFunctionsAgent
from langchain.memory import ConversationBufferMemory

load_dotenv()

AOAI_ENDPOINT=os.getenv("AOAI_ENDPOINT")
AOAI_API_KEY=os.getenv("AOAI_API_KEY")
AOAI_DEPLOY_GPT4O=os.getenv("AOAI_DEPLOY_GPT4O")
AOAI_DEPLOY_GPT4O_MINI=os.getenv("AOAI_DEPLOY_GPT4O_MINI")
AOAI_DEPLOY_EMBED_3_LARGE=os.getenv("AOAI_DEPLOY_EMBED_3_LARGE")
AOAI_DEPLOY_EMBED_3_SMALL=os.getenv("AOAI_DEPLOY_EMBED_3_SMALL")
AOAI_DEPLOY_EMBED_ADA=os.getenv("AOAI_DEPLOY_EMBED_ADA")

# client 대신 llm만 사용
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

# 전체 카테고리별 잔소리 목록을 위한 모델
class HolidayRemarks(BaseModel):
    취업: List[PriceSuggestionRequest]
    결혼: List[PriceSuggestionRequest]
    자녀_출산: List[PriceSuggestionRequest] = Field(alias="자녀·출산")
    학업: List[PriceSuggestionRequest]
    외모_건강: List[PriceSuggestionRequest] = Field(alias="외모·건강")
    돈_재테크: List[PriceSuggestionRequest] = Field(alias="돈·재테크")
    집안일: List[PriceSuggestionRequest]

# 명절 잔소리 자동 생성 함수
def generate_holiday_remarks():
    parser = PydanticOutputParser(pydantic_object=HolidayRemarks)

    prompt = ChatPromptTemplate.from_template("""
        아래 카테고리별로 명절 잔소리 목록을 작성하고, 각 잔소리에 대한 설명과 가격을 책정해주세요.
        카테고리: 취업, 결혼, 자녀·출산, 학업, 외모·건강, 돈·재테크, 집안일
        
        가격 책정 기준:
        1. 반복 빈도 (1~20) - 자주 들을수록 높음
        2. 정신적 데미지 (1~20) - 듣기 싫을수록 높음
        3. 피할 수 있는 난이도 (1~20) - 회피 어려울수록 높음
        4. 대체 가능성 (1~20) - 영원히 사라지지 않을수록 높음
        
        가격은 만원 단위로 책정하며, 1~15만원 사이로 설정해주세요.
        최종 설명: 결혼 관련 잔소리는 개인의 선택을 존중하지 않고 지속적인 정신적 압박을 주는 대표적인 잔소리입니다.
        최종 가격: 15만원

        잔소리에 대해 위 예시와 같은 형식으로 분석해주세요.
        {format_instructions}
    """)

    chain = prompt.partial(format_instructions=parser.get_format_instructions()) | llm | parser
    
    try:
        # 응답 생성 및 파싱
        response = chain.invoke({})
        # 모든 카테고리의 잔소리를 하나의 리스트로 변환
        all_remarks = []
        for category, remarks_list in response.model_dump().items():
            pprint.pprint(f"category: {category}, remarks_list: {remarks_list}")
            for remark_obj in remarks_list:
                all_remarks.append({
                    "remark": remark_obj["remark"],
                    "explanation": remark_obj["reason"],
                    "price": remark_obj["suggested_price"]
                })
        
        # FAISS에 저장
        add_remarks_to_faiss(all_remarks)
        print("✅ 명절 잔소리 데이터 추가 완료!")
        
    except Exception as e:
        print(f"⚠️ 응답 파싱 오류: {e}")

index_path = "faiss_index"
metadata_path = "remark_metadata.json"
dim = 3072
index = faiss.IndexFlatL2(dim)

# 메타데이터 저장을 위한 전역 변수
remark_metadata = []

# AI가 분류하는 함수 => FIX
@tool
def categorize_remark(remark: str) -> str:
    """잔소리를 명절과 일상으로 AI가 분석"""
    
    # AI에게 잔소리 유형을 분류하도록 지시하는 프롬프트
    messages = [
        SystemMessage(content="""
        너는 잔소리 분석 AI야.
        사용자가 입력한 잔소리가 '명절 잔소리'인지 '일상 잔소리'인지 판단해야 해.
        '명절 잔소리'는 명절(설, 추석 등)과 관련된 것들이고, '일상 잔소리'는 평소에도 들을 수 있는 거야.
        답변은 반드시 '명절 잔소리' 또는 '일상 잔소리' 중 하나만 반환해야 해.
        """),
        HumanMessage(content=remark)
    ]

    # AI 호출
    response = llm.invoke(messages)

    # 응답에서 잔소리 유형 추출
    category = response.content.strip()

    # 유효성 검증
    if category not in ["명절 잔소리", "일상 잔소리"]:
        category = "일상 잔소리"  # 예외 처리: AI가 이상한 값을 반환하면 기본값 사용

    return category if category in ["명절 잔소리", "일상 잔소리"] else "일상 잔소리"

# AI가 유사한 잔소리를 찾는 함수
@tool
def search_similar_remarks(query: str, top_k: int = 1):
    """FAISS를 사용하여 가장 유사한 잔소리를 찾는 함수"""
    
    # 1. FAISS 인덱스 로드
    if not os.path.exists(index_path):
        print("⚠️ FAISS 인덱스 파일이 존재하지 않습니다.")
        return []
    index = faiss.read_index(index_path)
    
    # 2. 메타데이터 로드
    if not os.path.exists(metadata_path):
        print("⚠️ 메타데이터 파일이 존재하지 않습니다.")
        return []
    
    with open(metadata_path, "r", encoding="utf-8") as f:
        remark_metadata = json.load(f)
    
    # 3. 입력 query를 임베딩 변환
    query_embedding = embeddings.embed_query(query)
    query_embedding_array = np.array(query_embedding).astype("float32").reshape(1, -1)
    
    # 4. FAISS를 이용한 유사도 검색
    distances, indices = index.search(query_embedding_array, top_k)
    
    # 5. 검색 결과 반환
    results = []
    for i, idx in enumerate(indices[0]):
        if idx < len(remark_metadata) and idx >= 0:
            results.append({
                "index": int(idx),
                "distance": float(distances[0][i]),
                "remark": remark_metadata[idx]
            })
    
    return results


@tool
def estimate_remark_price(remark: str) -> int:
    """잔소리 가격을 예측하는 함수 (유사한 잔소리의 가격을 참고하여 결정)"""

    if not os.path.exists(index_path) or not os.path.exists(metadata_path) :
        print("⚠️ FAISS 인덱스 또는 메타데이터 파일이 없습니다. 기본 가격 설정을 사용합니다.")
        return 5
    
    with open(metadata_path, "r", encoding="utf-8") as f:
        remark_metadata = json.load(f)
    
    index = faiss.read_index(index_path)

    query_embedding = embeddings.embed_query(remark)
    query_embedding_array = np.array(query_embedding).astype("float32").reshape(1, -1)

    distance, indices = index.search(query_embedding_array, 3)
    similar_prices = [remark_metadata[i]["price"] for i in indices[0] if 0 <= i < len(remark_metadata)]
    return int(np.mean(similar_prices)) if similar_prices else 5

# 데이터 추가 함수 (FAISS 사용)
def add_remarks_to_faiss(remarks):
    global remark_metadata, index
    
    if os.path.exists(index_path):
        index = faiss.read_index(index_path)
    
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            remark_metadata = json.load(f)
    
    for remark_data in remarks:
        remark = remark_data["remark"]
        # llm의 embeddings 메서드 사용
        embedding = embeddings.embed_query(remark)
        embedding_array = np.array([embedding]).astype("float32")
        
        index.add(embedding_array)
        remark_metadata.append(remark_data)

    faiss.write_index(index, index_path)
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(remark_metadata, f, ensure_ascii=False, indent=2)

# 2️⃣ AI가 도구를 직접 호출할 수 있도록 설정
def process_remark_with_tool_calling(remark, top_k=1):
    """AI가 도구를 활용하여 잔소리 분석 및 가격 예측을 수행"""
    
    # Step 1: 잔소리 유형 판별 (명절 잔소리 vs. 일상 잔소리)
    category = categorize_remark.invoke(remark)
    
    # Step 2: 명절 잔소리라면 즉시 가격 예측 수행
    if category == "명절 잔소리":
        estimated_price = estimate_remark_price.invoke(remark)
        return {
            "remark": remark,
            "category": category,
            "estimated_price": estimated_price
        }
    
    # Step 3: 일상 잔소리라면 유사 잔소리 검색
    similar_results = search_similar_remarks.invoke({"query": remark, "top_k": top_k})
    
    if similar_results and len(similar_results) > 0:
        return {
            "remark": remark,
            "category": category,
            "similar_remarks": similar_results
        }
    
    # Step 4: 유사한 잔소리가 없으면 가격 예측 수행
    estimated_price = estimate_remark_price.invoke(remark)
    return {
        "remark": remark,
        "category": category,
        "estimated_price": estimated_price
    }


# 메인 실행 코드
if __name__ == "__main__":
    test_remark = "너 아직도 결혼 생각 없니?"
        # 🔥 invoke() 방식으로 실행
    print(categorize_remark.invoke(test_remark))
    print(search_similar_remarks.invoke(test_remark))
    print(estimate_remark_price.invoke(test_remark))
    
    print(process_remark_with_tool_calling(test_remark))
    