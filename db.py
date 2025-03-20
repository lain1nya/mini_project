import os
import pprint
from dotenv import load_dotenv
from models import MultiPriceSuggestionRequests
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain.schema import SystemMessage
from faiss_db import add_remarks_to_faiss
from prompt_description import SYSTEM_MESSAGES

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
    
    system_message = f"""
        {SYSTEM_MESSAGES["original_script"]}

        {SYSTEM_MESSAGES["explanation_script"]}

        {SYSTEM_MESSAGES["updated_remark_script"]}

        {SYSTEM_MESSAGES["tone_script"]}
    """

    try:
        message = [
            SystemMessage(content=system_message)
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
