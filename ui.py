import streamlit as st
import requests

st.title("🗣 명절 잔소리 가격 측정기")

# 사용자 입력 받기
remark = st.text_input("💬 잔소리를 입력하세요:")

if st.button("💰 가격 측정하기"):
    if remark:
        response = requests.post("http://127.0.0.1:8000/get_price/", json={"remark": remark})
        print(f"response {response}")
        if response.status_code == 200:
            result = response.json()
            #디버깅을 위해 전체 응답 출력
            # st.success(result)
            if "new" in result:  # new 키가 있는지 확인
                is_new = result["new"]
                if is_new:
                    st.info("📋 이 잔소리는 처음 분석되었습니다!")
                else:
                    st.info("🔄 이 잔소리와 유사한 잔소리가 이미 분석되었습니다.")
            
            st.success(f"📢 '{remark}' 잔소리의 가격은 **{result['price']}** 입니다!")
            st.markdown(f"**근거는 다음과 같습니다:**  \n\n {result['explanation']}")
        else:
            st.error("⚠️ 가격을 측정하는 중 오류가 발생했습니다.")
    else:
        st.warning("⚠️ 잔소리를 입력해주세요!")
