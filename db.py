import faiss
import time
import numpy as np
import os
import json
import pprint
from dotenv import load_dotenv
from typing import List
from models import PriceSuggestionRequest
from pydantic import Field, BaseModel
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser

load_dotenv()

AOAI_ENDPOINT=os.getenv("AOAI_ENDPOINT")
AOAI_API_KEY=os.getenv("AOAI_API_KEY")
AOAI_DEPLOY_GPT4O=os.getenv("AOAI_DEPLOY_GPT4O")
AOAI_DEPLOY_GPT4O_MINI=os.getenv("AOAI_DEPLOY_GPT4O_MINI")
AOAI_DEPLOY_EMBED_3_LARGE=os.getenv("AOAI_DEPLOY_EMBED_3_LARGE")
AOAI_DEPLOY_EMBED_3_SMALL=os.getenv("AOAI_DEPLOY_EMBED_3_SMALL")
AOAI_DEPLOY_EMBED_ADA=os.getenv("AOAI_DEPLOY_EMBED_ADA")

# llm 모델 설정
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
        아래 카테고리별로 1개씩 명절 잔소리 목록을 작성하고, 각 잔소리에 대한 설명과 가격을 책정해주세요.
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
                    "explanation": remark_obj["explanation"],
                    "price": remark_obj["suggested_price"],
                    "repetition": remark_obj["repetition"],
                    "mental_damage": remark_obj["mental_damage"],
                    "avoidance_difficulty": remark_obj["avoidance_difficulty"],
                    "replaceability": remark_obj["replaceability"]
                })
                time.sleep(2)
        
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



# 메인 실행 코드
if __name__ == "__main__":
    generate_holiday_remarks()