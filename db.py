import faiss
import numpy as np
from openai import AzureOpenAI
import os
import json
from dotenv import load_dotenv


load_dotenv()

AOAI_ENDPOINT=os.getenv("AOAI_ENDPOINT")
AOAI_API_KEY=os.getenv("AOAI_API_KEY")
AOAI_DEPLOY_GPT4O=os.getenv("AOAI_DEPLOY_GPT4O")
AOAI_DEPLOY_GPT4O_MINI=os.getenv("AOAI_DEPLOY_GPT4O_MINI")
AOAI_DEPLOY_EMBED_3_LARGE=os.getenv("AOAI_DEPLOY_EMBED_3_LARGE")
AOAI_DEPLOY_EMBED_3_SMALL=os.getenv("AOAI_DEPLOY_EMBED_3_SMALL")
AOAI_DEPLOY_EMBED_ADA=os.getenv("AOAI_DEPLOY_EMBED_ADA")

client = AzureOpenAI(
  azure_endpoint = AOAI_ENDPOINT, 
  api_key=AOAI_API_KEY,  
  api_version="2024-10-21"
)

def generate_prompt():
    return f"""
        아래 카테고리별로 잔소리의 목록을 작성하고, 각 잔소리에 대한 설명과 가격도 함께 제공해주세요.
        각 카테고리별로 적어도 5개 이상의 잔소리를 포함해주세요.
        카테고리: 취업, 결혼, 자녀·출산, 학업, 외모·건강, 돈·재테크, 집안일
        
        가격 책정 기준:
        1. 반복 빈도 (1~20) - 자주 들을수록 높음
        2. 정신적 데미지 (1~20) - 듣기 싫을수록 높음
        3. 피할 수 있는 난이도 (1~20) - 회피 어려울수록 높음
        4. 대체 가능성 (1~20) - 영원히 사라지지 않을수록 높음
        예시:
        - "너 언제 결혼하니?"  
            → 반복 빈도: 10, 정신적 데미지: 9, 피할 수 있는 난이도: 9, 대체 가능성: 10  
            → 가격: 15만 원
        - "공부 좀 해라."
            → 반복 빈도: 8, 정신적 데미지: 6, 피할 수 있는 난이도: 5, 대체 가능성: 7  
            → 가격: 3만 원

        먼저 각 기준을 분석한 후, 만원 단위로 최종 가격을 산출하세요.

        가격은 만원 단위로 책정하며, 분포는 최대한 세분화해서 최저 1만원에서 최대 15만원 사이로 설정해주세요.
        
        반드시 다음과 같은 JSON 형식으로 응답해주세요:
        
        ```json
        {{
            "취업": [
                {{
                    "remark": "잔소리 내용",
                    "explanation": "이 잔소리에 대한 설명",
                    "price": 10
                }},
                {{
                    "remark": "잔소리 내용2",
                    "explanation": "이 잔소리에 대한 설명2",
                    "price": 12
                }},
                ...
            ],
            "결혼": [
                {{
                    "remark": "잔소리 내용",
                    "explanation": "이 잔소리에 대한 설명",
                    "price": 15
                }},
                ...
            ],
            ...
        }}
        ```
        
        반드시 JSON 형식으로 응답해주세요.
    """

def get_ai_list(prompt: str):
    response = client.chat.completions.create(
        model=AOAI_DEPLOY_GPT4O,
        messages=[
            {"role": "system", "content" : "너는 명절 잔소리 리스트를 JSON 형식으로 출력해주는 AI야."},
            {"role": "user", 'content': prompt}
        ]
    )
    ai_response = response.choices[0].message.content
    
    # JSON 형식으로 파싱 시도
    try:
        import json
        import re
        
        # 코드 블록 내부의 JSON 추출 시도
        code_block_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", ai_response)
        if code_block_match:
            json_text = code_block_match.group(1).strip()
            try:
                remarks_data = json.loads(json_text)
                print("코드 블록 내용 파싱 성공!")
            except json.JSONDecodeError as e:
                print(f"코드 블록 내용 파싱 실패: {e}")
                # 실패 시 다른 방법 시도
                code_block_match = None
        
        # 코드 블록에서 찾지 못했거나 파싱에 실패한 경우, 전체 텍스트에서 중괄호로 둘러싸인 부분 찾기
        if not code_block_match:
            json_match = re.search(r"(\{[\s\S]*\})", ai_response)
            if json_match:
                json_text = json_match.group(1).strip()
                try:
                    remarks_data = json.loads(json_text)
                except json.JSONDecodeError as e:
                    print(f"중괄호 내용 파싱 실패: {e}")
                    return None
            else:
                print("JSON 형식을 찾을 수 없습니다.")
                print("응답 내용:", ai_response)
                return None
        
        # 모든 카테고리의 잔소리를 하나의 리스트로 변환
        all_remarks = []
        
        for category, remarks_list in remarks_data.items():
            for remark_obj in remarks_list:
                # 새로운 JSON 구조에서 데이터 추출
                remark_text = remark_obj["remark"]
                explanation = remark_obj["explanation"]
                price = remark_obj["price"]
                
                # 카테고리 정보 추가
                full_explanation = f"{category} 관련 잔소리: {explanation}"
                
                # (잔소리 텍스트, 설명, 가격) 형식으로 저장
                all_remarks.append((remark_text, full_explanation, price))
        # FAISS에 저장
        add_remarks_to_faiss(all_remarks)
        
    except Exception as e:
        print(f"JSON 파싱 오류: {e}")
        print("응답 내용:", ai_response)
        pass

index_path = "faiss_index"
metadata_path = "remark_metadata.json"

dim = 3072
index = faiss.IndexFlatL2(dim)

# 메타데이터 저장을 위한 전역 변수
remark_metadata = []

# 데이터 추가 함수 (FAISS 사용)
def add_remarks_to_faiss(remarks):
    """
    remarks: 리스트 형태로 (잔소리 텍스트, 설명, 가격) 데이터 추가
    """
    global remark_metadata

    for idx, (remark, explanation, price) in enumerate(remarks):
        response = client.embeddings.create(
            input=remark,
            model=AOAI_DEPLOY_EMBED_3_LARGE
        )
        embedding = response.data[0].embedding
        
        # 벡터를 FAISS 인덱스에 추가
        # 차원 확인 로그 추가
        if len(embedding) != dim:
            print(f"경고: 임베딩 차원({len(embedding)})이 인덱스 차원({dim})과 일치하지 않습니다.")
        
        # 벡터를 FAISS 인덱스에 추가
        embedding_array = np.array([embedding]).astype("float32")  # 배열 형태 명확히 지정
        
        # 인덱스에 추가
        index.add(embedding_array)
        
        # 메타데이터 저장 (잔소리, 설명, 가격)
        remark_metadata.append({
            "remark": remark,
            "explanation": explanation,
            "price": price
        })
        
        print(f"Added remark '{remark}' to FAISS index.")

    # 인덱스를 파일로 저장
    faiss.write_index(index, index_path)
    
    # 메타데이터를 JSON 파일로 저장
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(remark_metadata, f, ensure_ascii=False, indent=2)
    
    print("✅ 잔소리 데이터가 FAISS 인덱스에 저장되었습니다.")

# FAISS 인덱스를 사용하여 검색하는 함수
def search_similar_remarks(query, top_k=5):
    """
    query: 검색할 텍스트
    top_k: 반환할 결과 수
    """
    global remark_metadata
    
    # 인덱스 로드
    if os.path.exists(index_path):
        index = faiss.read_index(index_path)
    else:
        print("⚠️ FAISS 인덱스 파일이 존재하지 않습니다.")
        return []
    
    # 메타데이터 로드
    if os.path.exists(metadata_path):
        with open(metadata_path, "r", encoding="utf-8") as f:
            remark_metadata = json.load(f)
    else:
        print("⚠️ 메타데이터 파일이 존재하지 않습니다.")
        return []
    
    # 쿼리 텍스트 임베딩
    response = client.embeddings.create(
        input=query,
        model=AOAI_DEPLOY_EMBED_3_LARGE
    )
    query_embedding = response.data[0].embedding
    
    # 벡터 변환
    query_embedding = np.array(query_embedding).astype("float32").reshape(1, -1)
    
    # 유사도 검색
    distances, indices = index.search(query_embedding, top_k)
    
    # 결과 반환 (인덱스와 메타데이터)
    results = []
    for i, idx in enumerate(indices[0]):
        if idx < len(remark_metadata) and idx >= 0:
            results.append({
                "index": int(idx),
                "distance": float(distances[0][i]),
                "metadata": remark_metadata[idx]
            })
    
    return results

# 예시 데이터
example_remarks = [
    # 취업 관련 잔소리 (10~12만원)
    ("요즘 취업 힘들다던데, 어디 지원해봤어?", "취업 준비 중인 사람에게 부담이 되는 전형적인 질문. 스트레스 강도가 높고 피하기 어려움.", 12),
    ("네 친구 누구는 어디 들어갔다더라.", "사회적 비교를 유도하는 말로, 상대방에게 자괴감을 줄 가능성이 높음.", 11),
    ("너는 대기업 안 들어가고 뭐하니?", "대기업 취업에 대한 압박을 주는 질문으로, 듣는 사람에게 부담이 큼.", 10),
]

# 메인 실행 코드
if __name__ == "__main__":
    # AI로부터 잔소리 목록 생성 및 FAISS에 저장
    get_ai_list(generate_prompt())
    
    # # 생성된 데이터가 없으면 예시 데이터 사용
    # if not remarks_data:
    #     print("AI 생성 데이터를 사용할 수 없어 예시 데이터를 사용합니다.")
    #     add_remarks_to_faiss(example_remarks)