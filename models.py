from pydantic import BaseModel, Field
from typing import List, Annotated, Literal, TypedDict

class RemarkRequest(BaseModel):
    remark: str

class UpdatedRemarkRequest(BaseModel):
    remark: str = Field(description="기존에 입력된 잔소리")
    updated_remark: str = Field(description="사용자가 말하고자 하는 의도를 기반으로 재해석된 잔소리")

class Explanation(BaseModel):
    explanation: str = Field(description="잔소리에 대한 AI의 최종 설명")

class FeedbackRequest(BaseModel):
    remark: str
    category: str
    is_positive: bool

class NewReasonRequest(BaseModel):
    remark: str
    reason: str
    suggested_price: int

# Supervisor를 위한 상태 정의
class SupervisorState(TypedDict):
    remark: Annotated[str, "single"]
    updated_remark: Annotated[str, "single"]
    category: Literal["명절 잔소리", "일상 잔소리"]
    suggested_price: int
    explanation: str
    similar_remark: bool

class PriceSuggestionRequest(BaseModel):
    remark: str = Field(description="입력된 잔소리")
    updated_remark: str = Field(description="사용자가 말하고자 하는 의도를 기반으로 재해석된 잔소리")
    category: Literal["명절 잔소리", "일상 잔소리"] = Field(description="입력된 잔소리의 종류")
    suggested_price: int = Field(description="예측된 최종 가격 (1~15)")
    explanation: str = Field(description="잔소리에 대한 AI의 최종 설명")
    repetition: int = Field(description="반복 빈도 점수 (1~20)")
    mental_damage: int = Field(description="정신적 데미지 점수 (1~20)")
    avoidance_difficulty: int = Field(description="피할 수 있는 난이도 점수 (1~20)")
    replaceability: int = Field(description="대체 가능성 점수 (1~20)")

class MultiPriceSuggestionRequests(BaseModel):
    remarks: List[PriceSuggestionRequest]

class AnalysisItem(BaseModel):
    criteria: str = Field(description="평가 기준 (반복 빈도/정신적 데미지/피할 수 있는 난이도/대체 가능성)")
    score: int = Field(description="1-10 사이의 점수")
    reason: str = Field(description="점수 부여 이유")

class PriceAnalysisOutput(BaseModel):
    thinking_steps: List[str] = Field(
        description="잔소리 가격 측정을 위한 사고 과정"
    )
    analysis: List[AnalysisItem] = Field(
        description="각 평가 기준별 분석 결과"
    )
    final_explanation: str = Field(
        description="최종 설명"
    )
    price: int = Field(
        description="최종 가격 (만원 단위)"
    )