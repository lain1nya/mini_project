# faiss_db.py

import os
import json
import pprint
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

# FAISS 인덱스 & 메타데이터 파일 경로
index_path = "faiss_index"
# index_file = os.path.join(index_path, "index.faiss")
metadata_path = "remark_metadata.json"

# Azure OpenAI 임베딩 모델 설정
embeddings = AzureOpenAIEmbeddings(
    azure_deployment=os.getenv("AOAI_DEPLOY_EMBED_3_LARGE"),
    openai_api_version="2024-10-21",
    azure_endpoint=os.getenv("AOAI_ENDPOINT"),
    api_key=os.getenv("AOAI_API_KEY")
)

# 글로벌 db 객체
db = None

# FAISS 인덱스 로드 또는 생성
def load_faiss_index():
    """FAISS 인덱스를 로드하고 없으면 새로 생성"""
    global db
    if os.path.exists(index_path):
        try:
            db = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
            print("✅ 기존 FAISS 인덱스 로드 성공")
        except Exception as e:
            print(f"⚠️ FAISS 인덱스 로드 실패: {e}")
            print("🚀 새로운 FAISS 인덱스 생성")
            db = FAISS.from_texts(["임시 데이터입니다."], embeddings)  # 기본값으로 초기화
            db.save_local(index_path)
    else:
        print("🚀 FAISS 인덱스가 존재하지 않음. 새로운 인덱스 생성")
        db = FAISS.from_texts(["임시 데이터입니다."], embeddings)  # 기본값으로 초기화
        db.save_local(index_path)

# FAISS 메타데이터 로드
def load_metadata():
    """메타데이터 로드"""
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

# 메타데이터 저장
def save_metadata(metadata):
    """메타데이터 저장"""
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)

# FAISS 데이터 추가 함수
def add_remarks_to_faiss(remarks):
    """FAISS 벡터DB에 데이터 추가"""
    global db
    metadata = load_metadata()

    # 메타데이터 업데이트
    metadata.extend(remarks)

    texts = [remark["remark"] for remark in remarks]  # remark 텍스트 리스트
    
    metadata_list = [
        {
            "remark": remark["remark"],
            "category": remark["category"],
            "explanation": remark["explanation"],
            "suggested_price": remark["suggested_price"],
            "repetition": remark["repetition"],
            "mental_damage": remark["mental_damage"],
            "avoidance_difficulty": remark["avoidance_difficulty"],
            "replaceability": remark["replaceability"],
            "positive_feedback": remark.get("positive_feedback", 0),
            "negative_feedback": remark.get("negative_feedback", 0)
        }
        for remark in remarks
    ]

    # ✅ FAISS 인덱스 업데이트 (벡터 추가)
    db.add_texts(texts=texts, metadatas=metadata_list)
    db.save_local(index_path)

    # 메타데이터 저장
    save_metadata(metadata)
    print(f"✅ {len(remarks)}개의 remark가 FAISS에 추가되었습니다!")

# FAISS에서 유사한 잔소리 검색
def search_similar_remark(query: str, category: str, top_k: int=1):
    if db is None:
        print("❌ FAISS DB가 비어 있습니다. 먼저 데이터를 추가하세요.")
        return None

    # 검색 수행
    docs = db.similarity_search(query, top_k)

    if not docs:
        print("❌ 유사한 잔소리를 찾을 수 없습니다.")
        return None
    
    filtered_docs = [doc for doc in docs if doc.metadata.get("category") == category]

    if not filtered_docs:
        print(f"⚠️ 같은 카테고리({category})에서 유사한 잔소리를 찾을 수 없습니다.")
        return None
    
    most_similar_doc = filtered_docs[0]  # 🔥 첫 번째 결과 선택
    return most_similar_doc

# 🔥 FAISS에서 모든 remark 데이터 가져오기
def fetch_all_remarks_from_faiss():
    """FAISS에서 모든 remark 데이터를 가져오는 함수"""
    if db is None:
        print("❌ FAISS DB가 비어 있습니다. 먼저 데이터를 추가하세요.")
        return []

    # 🔍 FAISS에서 모든 문서 가져오기
    all_docs = db.similarity_search("", k=1000)  # 최대 1000개 가져오기 (필요 시 조정 가능)

    # 🔍 각 문서에서 메타데이터 추출
    remarks = [doc.metadata for doc in all_docs]
    
    print(f"🔄 FAISS에서 {len(remarks)}개의 잔소리 데이터를 불러왔습니다.")
    pprint.pprint(remarks)
    return remarks

# 실행 시 FAISS 인덱스 로드
load_faiss_index()
