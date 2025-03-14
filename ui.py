import streamlit as st
import requests

st.title("🗣 명절 잔소리 가격 측정기")

# 세션 상태 초기화
if 'positive_feedback_submitted' not in st.session_state:
    st.session_state.positive_feedback_submitted = False
if 'negative_feedback_submitted' not in st.session_state:
    st.session_state.negative_feedback_submitted = False
if 'show_negative_feedback_options' not in st.session_state:
    st.session_state.show_negative_feedback_options = False
if 'show_negative_feedback_form' not in st.session_state:
    st.session_state.show_negative_feedback_form = False
if 'current_remark' not in st.session_state:
    st.session_state.current_remark = ""
if 'current_price' not in st.session_state:
    st.session_state.current_price = ""

# 사용자 입력 받기
remark = st.text_input("💬 잔소리를 입력하세요:")

def submit_positive_feedback():
    feedback_response = requests.post(
        "http://127.0.0.1:8000/feedback/",
        json={
            "remark": st.session_state.current_remark,
            "is_positive": True,
            "suggested_price": None,
            "reason": None
        }
    )
    if feedback_response.status_code == 200:
        st.session_state.positive_feedback_submitted = True

def submit_negative_feedback():
    feedback_response = requests.post(
        "http://127.0.0.1:8000/feedback/",
        json={
            "remark": st.session_state.current_remark,
            "is_positive": False,
            "suggested_price": None,
            "reason": "가격이 적절하지 않습니다."
        }
    )
    if feedback_response.status_code == 200:
        show_negative_feedback_options()

def submit_detailed_negative_feedback(suggested_price, reason):
    feedback_response = requests.post(
        "http://127.0.0.1:8000/feedback/",
        json={
            "remark": st.session_state.current_remark,
            "is_positive": False,
            "suggested_price": suggested_price,
            "reason": reason
        }
    )
    if feedback_response.status_code == 200:
        st.session_state.negative_feedback_submitted = True

def show_negative_feedback_options():
    st.session_state.show_negative_feedback_options = True

def show_negative_feedback_form():
    st.session_state.show_negative_feedback_form = True

if st.button("💰 가격 측정하기"):
    if remark:
        # 새로운 가격 측정시 피드백 상태 초기화
        st.session_state.positive_feedback_submitted = False
        st.session_state.negative_feedback_submitted = False
        st.session_state.show_negative_feedback_options = False
        st.session_state.show_negative_feedback_form = False
        st.session_state.current_remark = remark
        
        response = requests.post("http://127.0.0.1:8000/get_price/", json={"remark": remark})
        if response.status_code == 200:
            result = response.json()
            
            if "new" in result:
                is_new = result["new"]
                if is_new:
                    st.info("📋 이 잔소리는 처음 분석되었습니다!")
                else:
                    st.info("🔄 이 잔소리와 유사한 잔소리가 이미 분석되었습니다.")
            
            st.success(f"📢 '{remark}' 잔소리의 가격은 **{result['price']}** 입니다!")
            st.markdown(f"**근거는 다음과 같습니다:**  \n\n {result['explanation']}")
            st.session_state.current_price = result['price']

            # 피드백 섹션
            st.markdown("---")
            st.markdown("### 💭 이 가격에 대해 어떻게 생각하시나요?")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if not st.session_state.positive_feedback_submitted:
                    if st.button("👍 좋아요", key="positive_feedback"):
                        submit_positive_feedback()
                if st.session_state.positive_feedback_submitted:
                    st.success("피드백 감사합니다! 👍")
            
            with col2:
                if not st.session_state.negative_feedback_submitted:
                    if st.button("👎 나빠요", key="negative_feedback"):
                        show_negative_feedback_options()
                if st.session_state.negative_feedback_submitted:
                    st.success("소중한 의견 감사합니다! 🙏")

            # 부정적 피드백 옵션 표시
            if st.session_state.show_negative_feedback_options and not st.session_state.negative_feedback_submitted:
                st.markdown("### 💭 어떤 의견을 남기시겠습니까?")
                col3, col4 = st.columns(2)
                
                with col3:
                    if st.button("단순히 가격이 안 맞아요"):
                        submit_simple_negative_feedback()
                
                with col4:
                    if st.button("더 나은 가격을 제안할게요"):
                        show_negative_feedback_form()

            # 상세 피드백 폼 표시
            if st.session_state.show_negative_feedback_form and not st.session_state.negative_feedback_submitted:
                st.markdown("### 💡 더 나은 가격을 제안해주세요")
                suggested_price = st.number_input("제안하는 가격 (만원)", min_value=1, max_value=15, value=5)
                reason = st.text_area("왜 이 가격이 더 적절하다고 생각하시나요?")
                
                if st.button("제안하기", key="submit_suggestion"):
                    submit_detailed_negative_feedback(suggested_price, reason)
        else:
            st.error("⚠️ 가격을 측정하는 중 오류가 발생했습니다.")
    else:
        st.warning("⚠️ 잔소리를 입력해주세요!")
