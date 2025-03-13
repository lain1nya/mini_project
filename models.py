from pydantic import BaseModel, Field
from typing import List

class RemarkRequest(BaseModel):
    remark: str

class FeedbackRequest(BaseModel):
    remark: str
    is_positive: bool

class PriceSuggestionRequest(BaseModel):
    remark: str
    suggested_price: int
    reason: str

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