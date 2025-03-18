import os
import pprint
from dotenv import load_dotenv
from models import MultiPriceSuggestionRequests
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.schema import SystemMessage
from faiss_db import add_remarks_to_faiss

# 환경 변수 로드
load_dotenv()

AOAI_ENDPOINT = os.getenv("AOAI_ENDPOINT")
AOAI_API_KEY = os.getenv("AOAI_API_KEY")
AOAI_DEPLOY_GPT4O = os.getenv("AOAI_DEPLOY_GPT4O")
AOAI_DEPLOY_EMBED_3_LARGE = os.getenv("AOAI_DEPLOY_EMBED_3_LARGE")

# LLM 모델 설정
llm = AzureChatOpenAI(
    openai_api_version="2024-10-21",
    azure_deployment=AOAI_DEPLOY_GPT4O,
    azure_endpoint=AOAI_ENDPOINT,
    api_key=AOAI_API_KEY
)

# 임베딩 모델 설정
embeddings = AzureOpenAIEmbeddings(
    azure_deployment=AOAI_DEPLOY_EMBED_3_LARGE,
    openai_api_version="2024-10-21",
    azure_endpoint=AOAI_ENDPOINT,
    api_key=AOAI_API_KEY
)

# 명절 잔소리 자동 생성 함수
def generate_holiday_remarks():
    structured_llm = llm.with_structured_output(MultiPriceSuggestionRequests)
    
    try:
        message = [
            SystemMessage(content="""
                아래 종류 별로 정확히 '5개'씩 "명절 잔소리 목록"을 JSON 형식으로 작성해주세요.
                카테고리별로 배열로 구분해서 변환하세요.

                종류: 취업, 결혼, 자녀·출산, 학업, 외모·건강, 돈·재테크, 집안일
                
                가격 책정 기준:
                1. 반복 빈도 (1~20) - 자주 들을수록 높음
                2. 정신적 데미지 (1~20) - 듣기 싫을수록 높음
                3. 피할 수 있는 난이도 (1~20) - 회피 어려울수록 높음
                4. 대체 가능성 (1~20) - 영원히 사라지지 않을수록 높음
                
                가격은 1만원 단위로 책정하며, 1~15 사이로 설정해주세요.

                "category" 필드는 반드시 "명절 잔소리"로 고정해야 합니다.
                "explanation" 필드는 권유형으로 작성해주세요.
            """)
        ]

        # 응답 생성 및 파싱
        response = structured_llm.invoke(message)
        all_remarks = [remark.model_dump() for remark in response.remarks]

        pprint.pprint(all_remarks)

        # FAISS에 저장
        add_remarks_to_faiss(all_remarks)
        print("✅ 명절 잔소리 데이터 추가 완료!")

    except Exception as e:
        print(f"⚠️ 응답 파싱 오류: {e}")

# 메인 실행 코드
if __name__ == "__main__":

    generate_holiday_remarks()
