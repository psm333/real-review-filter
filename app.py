"""진짜 리뷰 필터 — Streamlit 앱.  실행: streamlit run app.py"""
import os

import openai
import streamlit as st
from dotenv import load_dotenv

from analyzer import MODELS, PROMPT_LABELS, analyze
from fetchers import (
    FetchError,
    extract_video_id,
    get_blog_text,
    get_youtube_comments,
    get_youtube_transcript,
)

load_dotenv()


def get_secret(name: str) -> str:
    try:
        return st.secrets[name]
    except Exception:
        return os.getenv(name, "")


OPENAI_KEY = get_secret("OPENAI_API_KEY")
YOUTUBE_KEY = get_secret("YOUTUBE_API_KEY")

st.set_page_config(page_title="진짜 리뷰 필터", page_icon="🔍", layout="centered")
st.title("🔍 진짜 리뷰 필터")
st.caption("광고 멘트는 걸러내고, 리뷰 속 **진짜 단점 3개 · 장점 3개**만 보여드려요.")

# ---------------- 사이드바: 설정 ----------------
with st.sidebar:
    st.header("⚙️ 설정")
    model_label = st.selectbox("AI 모델", list(MODELS.keys()))
    prompt_version = st.radio(
        "프롬프트 버전 (실험용)",
        list(PROMPT_LABELS.keys()),
        index=3,
        format_func=lambda k: PROMPT_LABELS[k],
    )
    use_comments = st.checkbox("💬 댓글도 분석하기", value=True)
    st.divider()
    st.caption("🛡️ 이메일·전화번호는 AI에 보내기 전에 자동으로 가려져요. 분석 내용은 저장하지 않아요.")
    if not OPENAI_KEY:
        st.error("OPENAI_API_KEY 가 설정되지 않았어요. README를 확인하세요.")

# ---------------- 입력 ----------------
mode = st.radio("입력 방법", ["▶️ 유튜브 링크", "📝 블로그 링크", "✍️ 직접 붙여넣기"], horizontal=True)

url, pasted_body, pasted_comments = "", "", ""
if mode == "▶️ 유튜브 링크":
    url = st.text_input("유튜브 리뷰 영상 링크", placeholder="https://www.youtube.com/watch?v=...")
    if use_comments and not YOUTUBE_KEY:
        st.info("YOUTUBE_API_KEY 가 없어서 댓글은 건너뛰어요. 자막 분석은 그대로 돼요.")
elif mode == "📝 블로그 링크":
    url = st.text_input("블로그 리뷰 링크 (네이버 블로그, 티스토리 등)", placeholder="https://blog.naver.com/...")
    if use_comments:
        pasted_comments = st.text_area("블로그 댓글 (선택 · 한 줄에 하나씩 붙여넣기)", height=100)
else:
    pasted_body = st.text_area("리뷰 본문 또는 영상 자막을 붙여넣으세요", height=220)
    if use_comments:
        pasted_comments = st.text_area("댓글 (선택 · 한 줄에 하나씩)", height=100)

go = st.button("🔎 진짜 장단점 분석하기", type="primary", use_container_width=True)


# ---------------- 결과 화면 ----------------
def show_result(result: dict, usage: dict, source_text: str, comments: list[str]):
    st.success(f"**한 줄 총평** · {result.get('summary', '')}")
    multi = result.get("review_type") == "여러 제품 비교"
    if multi:
        names = ", ".join(result.get("products", [])[:6])
        st.info(f"📦 **여러 제품을 비교한 리뷰예요.** 장단점마다 어떤 제품인지 표시했어요.\n\n등장 제품: {names}")

    ad = int(result.get("ad_score", 0) or 0)
    level = "🟢 낮음" if ad < 30 else ("🟡 보통" if ad < 70 else "🔴 높음")
    st.subheader(f"📢 광고 의심 지수: {ad}점 ({level})")
    st.progress(min(max(ad, 0), 100) / 100)
    st.caption(result.get("ad_reason", ""))

    col_bad, col_good = st.columns(2)
    with col_bad:
        st.markdown("### 👎 치명적 단점")
        for i, c in enumerate(result.get("cons", [])[:3], 1):
            with st.container(border=True):
                if multi and c.get("product"):
                    st.caption(f"🏷️ {c['product']}")
                st.markdown(f"**{i}. {c.get('point', '')}**")
                st.caption(f"📌 근거: “{c.get('evidence', '')}”")
        if not result.get("cons"):
            st.write("원문에서 근거 있는 단점을 찾지 못했어요.")
    with col_good:
        st.markdown("### 👍 핵심 장점")
        for i, p in enumerate(result.get("pros", [])[:3], 1):
            with st.container(border=True):
                if multi and p.get("product"):
                    st.caption(f"🏷️ {p['product']}")
                st.markdown(f"**{i}. {p.get('point', '')}**")
                st.caption(f"📌 근거: “{p.get('evidence', '')}”")

    ca = result.get("comment_analysis")
    if ca:
        st.markdown("### 💬 댓글 감지")
        m1, m2, m3 = st.columns(3)
        m1.metric("🗣️ 광고라고 지적", f"{ca.get('ad_suspicion_count', 0)}개")
        m2.metric("🤖 홍보성(바이럴) 의심", f"{ca.get('viral_count', 0)}개")
        m3.metric("🙋 실사용자 의견", f"{len(ca.get('real_user_opinions', []))}개")
        if ca.get("mismatch"):
            st.warning(f"⚠️ **리뷰와 댓글이 다른 점**: {ca['mismatch']}")
        for op in ca.get("real_user_opinions", [])[:5]:
            icon = "👎" if op.get("type") == "단점" else "👍"
            st.write(f"{icon} {op.get('opinion', '')}")
        if ca.get("ad_suspicion_examples"):
            with st.expander("🗣️ 광고라고 지적하는 댓글 보기"):
                for ex in ca["ad_suspicion_examples"]:
                    st.write(f"- {ex}")
        if ca.get("viral_examples"):
            with st.expander("🤖 홍보성(바이럴) 의심 댓글 보기"):
                for ex in ca["viral_examples"]:
                    st.write(f"- {ex}")

    st.markdown("### 🔎 AI가 본 자료 확인하기")
    with st.expander(f"🎬 영상·글 본문에서 걸러낸 광고성 문장 ({len(result.get('ad_sentences', []))}개)"):
        st.caption("리뷰어가 직접 한 말(자막·본문) 중에서 홍보처럼 보여서 장단점 판단에서 뺀 문장이에요.")
        for s in result.get("ad_sentences", []):
            st.write(f"- ~~{s}~~")
    with st.expander(f"💬 가져온 댓글 전체 ({len(comments)}개, 답글 포함)"):
        if comments:
            for c in comments:
                st.write(f"- {c}")
        else:
            st.write("분석에 쓴 댓글이 없어요.")
    with st.expander("📄 분석한 원문(자막·본문) 일부"):
        st.write(source_text[:1500] + ("..." if len(source_text) > 1500 else ""))

    st.caption(
        f"🤖 AI 분석 결과는 틀릴 수 있어요. 구매 전에 근거 문장을 꼭 직접 확인하세요. "
        f"(사용 토큰: 입력 {usage['input_tokens']:,} / 출력 {usage['output_tokens']:,})"
    )


if go:
    if not OPENAI_KEY:
        st.error("API 키가 없어서 분석할 수 없어요.")
        st.stop()
    try:
        with st.spinner("리뷰를 가져오는 중..."):
            comments: list[str] = []
            if mode == "▶️ 유튜브 링크":
                if not url.strip():
                    st.warning("링크를 입력해 주세요.")
                    st.stop()
                vid = extract_video_id(url)
                text = get_youtube_transcript(vid)
                if use_comments and YOUTUBE_KEY:
                    try:
                        comments = get_youtube_comments(vid, YOUTUBE_KEY)
                    except FetchError as e:
                        st.info(f"{e} 본문만 분석할게요.")
            elif mode == "📝 블로그 링크":
                if not url.strip():
                    st.warning("링크를 입력해 주세요.")
                    st.stop()
                text = get_blog_text(url)
            else:
                text = pasted_body.strip()
                if len(text) < 50:
                    st.warning("내용이 너무 짧아요. 50자 이상 붙여넣어 주세요.")
                    st.stop()
            if use_comments and pasted_comments.strip():
                comments += [c.strip() for c in pasted_comments.splitlines() if c.strip()]

        with st.spinner(f"AI가 분석 중... (댓글 {len(comments)}개 포함)"):
            result, usage = analyze(
                text, comments, prompt_version, MODELS[model_label], api_key=OPENAI_KEY
            )
        show_result(result, usage, text, comments)

    except FetchError as e:
        st.error(f"😥 {e}")
    except openai.AuthenticationError:
        st.error("😥 OpenAI API 키가 올바르지 않아요.")
    except openai.RateLimitError:
        st.error("😥 요청이 너무 많거나 크레딧이 부족해요. 잠시 후 다시 시도하거나 결제 잔액을 확인하세요.")
    except openai.APIError as e:
        st.error(f"😥 AI 서버 오류가 났어요. 잠시 후 다시 시도해 주세요. ({type(e).__name__})")
    except ValueError as e:
        st.error(f"😥 AI 응답을 읽지 못했어요. 다시 눌러 주세요. ({e})")
