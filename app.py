"""Real-pick — Streamlit 앱.  실행: py -m streamlit run app.py"""
import base64
import html
import os
import re
from pathlib import Path

import openai
import streamlit as st
from dotenv import load_dotenv

from analyzer import MODELS, PROMPT_LABELS, analyze
from fetchers import (
    FetchError,
    extract_video_id,
    get_blog_post,
    get_youtube_comments,
    get_youtube_meta,
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

ASSETS = Path(__file__).parent / "assets"


def img_b64(name: str) -> str:
    """assets 폴더의 그림을 HTML에 바로 넣을 수 있는 글자(base64)로 바꾼다. 없으면 빈 글자."""
    try:
        return "data:image/png;base64," + base64.b64encode((ASSETS / name).read_bytes()).decode()
    except OSError:
        return ""


LOGO = img_b64("logo.png")
ICON_PATH = ASSETS / "icon.png"

st.set_page_config(page_title="RealPick AI · 진짜 리뷰만 골라보기", page_icon=str(ICON_PATH) if ICON_PATH.exists() else "🔍", layout="centered", initial_sidebar_state="expanded")


def e(text) -> str:
    """AI·댓글 글자를 화면에 넣기 전에 HTML 특수문자를 막는다."""
    return html.escape(str(text or "")).replace("\r", "").replace("\n", "<br>")


# ======================= 디자인 (CSS) =======================
CSS = """
<link rel="stylesheet" href="https://cdn.jsdelivr.net/gh/orioncactus/pretendard/dist/web/static/pretendard.css">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Noto+Serif+KR:wght@600;700;900&display=swap">
<style>
:root { --ink:#1F1F1C; --sub:#6B6B66; --line:#E7E5DF; --bg:#FBFAF7; --side:#F3F2EE; --hl:#FFF0A0; }
html, body, [class*="css"], .stMarkdown, .stTextInput, .stChatInput textarea, button {
  font-family: 'Pretendard', -apple-system, 'Apple SD Gothic Neo', sans-serif !important;
}
.stApp { background: var(--bg); color: var(--ink); }
header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }
.block-container { max-width: 780px; padding-top: 3.2rem; padding-bottom: 7rem; }

/* 사이드바 */
section[data-testid="stSidebar"] { background: var(--side); border-right: 1px solid var(--line); }
section[data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }
.brand { font-weight: 800; font-size: 1.05rem; letter-spacing: -0.02em; margin-bottom: 1.1rem; }
.pill { display:inline-flex; gap:4px; background:#fff; border:1px solid var(--line); border-radius:999px; padding:3px; margin-bottom:1rem; }
.pill span { font-size:.8rem; padding:4px 12px; border-radius:999px; color:var(--sub); }
.pill span.on { background:var(--ink); color:#fff; font-weight:600; }
.toc-title { font-size:.88rem; font-weight:700; line-height:1.4; margin:.2rem 0 .8rem; }
.toc a { display:flex; gap:10px; padding:6px 4px; font-size:.86rem; color:var(--sub) !important; text-decoration:none !important; border-radius:6px; }
.toc a:hover { background:#E9E8E3; color:var(--ink) !important; }
.toc a span { color:#A9A8A2; font-variant-numeric: tabular-nums; }
.side-note { font-size:.78rem; color:var(--sub); line-height:1.6; }

/* 첫 화면: 로고 + 가운데 입력창 */
.hero { padding: 13vh 0 1.4rem; text-align:center; }
.hero img.logo { width: 340px; max-width: 80%; }
.hero .tag { margin-top: .9rem; font-size: 1.05rem; color: var(--sub); }
.hero .tag b { color: var(--ink); }
[data-testid="stForm"] { background:#fff; border:1px solid var(--line) !important; border-radius:18px; padding: 14px 16px 10px; box-shadow: 0 10px 30px rgba(0,0,0,.06); }
[data-testid="stForm"] textarea { border:none !important; background:transparent !important; font-size:1rem !important; box-shadow:none !important; }
[data-testid="stForm"] [data-baseweb="textarea"] { border:none !important; background:transparent !important; }
[data-testid="stForm"] button[kind="primaryFormSubmit"], [data-testid="stForm"] button[kind="primary"] { background:#1F1F1C; border:none; border-radius:10px; }
.hint { font-size:.78rem; color:#A9A8A2; padding-top:.55rem; }
.steps { display:flex; gap:10px; margin-top:2.2rem; flex-wrap:wrap; }
.step { flex:1; min-width:180px; background:transparent; border:1px solid var(--line); border-radius:12px; padding:14px 16px; }
.step b { display:block; font-size:.92rem; margin-bottom:4px; }
.step small { color:var(--sub); font-size:.8rem; line-height:1.5; }
.serif { font-family: 'Noto Serif KR', serif !important; letter-spacing:-0.02em; }
.brand img { height: 30px; }
/* 결과 문서 */
.doc-title { font-size: 2.15rem; line-height:1.3; font-weight:700; margin: .3rem 0 1.2rem; color:var(--ink); }
.src { display:flex; gap:14px; align-items:center; background:#fff; border:1px solid var(--line); border-radius:12px; padding:10px; text-decoration:none !important; }
.src img { width:120px; height:68px; object-fit:cover; border-radius:8px; flex-shrink:0; background:#eee; }
.src .t { font-weight:600; font-size:.92rem; color:var(--ink); line-height:1.4; display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical; overflow:hidden; }
.src .m { font-size:.78rem; color:var(--sub); margin-top:4px; }
.src .go { margin-left:auto; font-size:.8rem; color:var(--sub); white-space:nowrap; padding-right:6px; }
.chips { display:flex; gap:6px; flex-wrap:wrap; margin: 12px 0 2.2rem; }
.chip { font-size:.78rem; border:1px solid var(--line); background:#fff; border-radius:8px; padding:4px 10px; color:var(--sub); }
.chip b { color:var(--ink); }
.chip.red b { color:#D2453B; } .chip.amber b { color:#C98A00; } .chip.green b { color:#2E8B57; }

h2.sec { font-size:1.15rem; font-weight:700; margin: 2.4rem 0 .9rem; padding-bottom:.55rem; border-bottom: 2px solid var(--ink); scroll-margin-top: 70px; }
h2.sec small { font-weight:500; color:var(--sub); font-size:.82rem; margin-left:6px; }
.lead { font-size:1.02rem; line-height:1.8; }
mark { background: linear-gradient(transparent 55%, var(--hl) 55%); color:inherit; padding:0 2px; }
.gauge { height:6px; background:#ECEAE4; border-radius:99px; overflow:hidden; margin:.7rem 0 .4rem; }
.gauge div { height:100%; border-radius:99px; }
.note { font-size:.84rem; color:var(--sub); line-height:1.6; }
.callout { background:#fff; border:1px solid var(--line); border-left:3px solid #E0A800; border-radius:8px; padding:12px 14px; font-size:.9rem; line-height:1.65; margin: .8rem 0; }

ol.pts { padding-left: 1.3rem; margin:0; }
ol.pts > li { margin: 0 0 1.05rem; font-size:1rem; line-height:1.6; }
ol.pts > li::marker { font-weight:700; }
ol.pts .tag { font-size:.72rem; background:#EFEDE7; color:var(--sub); border-radius:5px; padding:1px 7px; margin-left:6px; font-weight:500; }
ol.pts ul { margin:.35rem 0 0; padding-left:1.1rem; }
ol.pts ul li { font-size:.88rem; color:var(--sub); line-height:1.65; list-style: disc; }

.stats { display:grid; grid-template-columns: repeat(4, 1fr); gap:8px; margin: .4rem 0 .6rem; }
.stat { background:#fff; border:1px solid var(--line); border-radius:10px; padding:10px 12px; }
.stat small { font-size:.75rem; color:var(--sub); }
.stat div { font-size:1.35rem; font-weight:800; margin-top:2px; }
ul.plain { padding-left:1.1rem; } ul.plain li { font-size:.9rem; line-height:1.65; margin-bottom:.25rem; }

details { background:#fff; border:1px solid var(--line); border-radius:10px; padding: 10px 14px; margin: .5rem 0; }
details summary { cursor:pointer; font-size:.9rem; font-weight:600; }
details ul { margin:.6rem 0 0; padding-left:1.1rem; }
details li { font-size:.86rem; color:var(--sub); line-height:1.6; margin-bottom:.25rem; }
details .strike li { text-decoration: line-through; }
details p { font-size:.86rem; color:var(--sub); line-height:1.7; white-space:pre-wrap; }

.foot { text-align:center; font-size:.76rem; color:#A9A8A2; margin-top:3rem; }

/* 아래 입력창 */
[data-testid="stBottom"] > div { background: transparent; }
[data-testid="stChatInput"] { border-radius: 14px !important; border:1px solid var(--line) !important; background:#fff !important; box-shadow: 0 6px 24px rgba(0,0,0,.06); }
</style>
"""
# Streamlit은 HTML 안에 빈 줄이 있으면 거기서 HTML이 끝난 걸로 봐서, 빈 줄을 모두 없앤다
st.markdown("\n".join(line for line in CSS.splitlines() if line.strip()), unsafe_allow_html=True)

SECTIONS = [
    ("summary", "한눈에 보기"),
    ("cons", "치명적 단점"),
    ("pros", "핵심 장점"),
    ("comments", "댓글 감지"),
    ("ads", "걸러낸 광고 문장"),
    ("source", "원문 확인"),
]

# ======================= 사이드바 =======================
data = st.session_state.get("data")
with st.sidebar:
    brand = f'<img src="{LOGO}" alt="RealPick AI">' if LOGO else "🔍 RealPick AI"
    st.markdown(f'<div class="brand">{brand}</div>', unsafe_allow_html=True)
    if data:
        links = "".join(
            f'<a href="#{sid}"><span>{i:02d}</span>{name}</a>' for i, (sid, name) in enumerate(SECTIONS, 1)
        )
        st.markdown(
            f'<div class="toc-title">{e(data["meta"]["title"])}</div><nav class="toc">{links}</nav>',
            unsafe_allow_html=True,
        )
        st.write("")
        if st.button("＋ 새 리뷰 분석하기", use_container_width=True):
            st.session_state.pop("data", None)
            st.rerun()
    else:
        st.markdown(
            '<div class="side-note">가운데 입력창에 <b>유튜브 링크</b>, <b>블로그 링크</b>, '
            "또는 <b>리뷰 글</b>을 붙여넣으면 분석이 시작돼요.</div>",
            unsafe_allow_html=True,
        )
    st.write("")
    with st.expander("⚙️ 분석 설정"):
        model_label = st.selectbox("AI 모델", list(MODELS.keys()))
        prompt_version = st.radio(
            "프롬프트 (실험용)", list(PROMPT_LABELS.keys()), index=3, format_func=lambda k: PROMPT_LABELS[k]
        )
        use_comments = st.checkbox("댓글도 분석하기", value=True)
    with st.expander("💬 댓글 직접 넣기"):
        pasted_comments = st.text_area(
            "블로그나 붙여넣은 글의 댓글을 한 줄에 하나씩", height=120, label_visibility="visible"
        )
    st.markdown(
        '<div class="side-note" style="margin-top:1rem">🛡️ 이메일·전화번호는 AI에 보내기 전에 가려지고, '
        "분석 내용은 저장하지 않아요.</div>",
        unsafe_allow_html=True,
    )
    if not OPENAI_KEY:
        st.error("OPENAI_API_KEY 가 설정되지 않았어요. README를 확인하세요.")


# ======================= 입력 처리 =======================
def detect_kind(q: str) -> str:
    if re.search(r"(youtube\.com|youtu\.be)/", q):
        return "youtube"
    if re.match(r"https?://", q):
        return "blog"
    return "text"


def run_analysis(query: str):
    kind = detect_kind(query)
    comments: list[str] = []
    with st.status("리뷰를 분석하고 있어요...", expanded=True) as status:
        if kind == "youtube":
            st.write("▶️ 유튜브 자막을 가져오는 중")
            vid = extract_video_id(query)
            meta = get_youtube_meta(vid)
            text = get_youtube_transcript(vid)
            if use_comments and YOUTUBE_KEY:
                st.write("💬 댓글을 가져오는 중")
                try:
                    comments = get_youtube_comments(vid, YOUTUBE_KEY)
                except FetchError as err:
                    st.write(f"⚠️ {err} 본문만 분석할게요.")
        elif kind == "blog":
            st.write("📝 블로그 글을 가져오는 중")
            meta, text = get_blog_post(query)
        else:
            if len(query) < 50:
                raise FetchError("글이 너무 짧아요. 리뷰 링크나 50자 이상의 리뷰 글을 넣어 주세요.")
            text = query
            first = text.strip().splitlines()[0][:40]
            meta = {"title": first + ("…" if len(first) >= 40 else ""), "channel": "직접 붙여넣은 글", "thumb": "", "url": ""}
        if use_comments and pasted_comments.strip():
            comments += [c.strip() for c in pasted_comments.splitlines() if c.strip()]

        st.write(f"🤖 AI가 광고를 걸러내고 장단점을 고르는 중 (댓글 {len(comments)}개)")
        result, usage = analyze(text, comments, prompt_version, MODELS[model_label], api_key=OPENAI_KEY)
        status.update(label="분석 완료!", state="complete", expanded=False)

    st.session_state["data"] = {
        "result": result,
        "usage": usage,
        "meta": meta,
        "text": text,
        "comments": comments,
        "kind": kind,
        "setting": f"{model_label.split(' ')[0]} · 프롬프트 {prompt_version}",
    }


def handle(query: str):
    if not query or not query.strip():
        return
    if not OPENAI_KEY:
        st.error("API 키가 없어서 분석할 수 없어요.")
        return
    try:
        run_analysis(query.strip())
        st.rerun()
    except FetchError as err:
        st.error(f"😥 {err}")
    except openai.AuthenticationError:
        st.error("😥 OpenAI API 키가 올바르지 않아요.")
    except openai.RateLimitError:
        st.error("😥 요청이 너무 많거나 크레딧이 부족해요. 잠시 후 다시 시도하거나 결제 잔액을 확인하세요.")
    except openai.APIError as err:
        st.error(f"😥 AI 서버 오류가 났어요. 잠시 후 다시 시도해 주세요. ({type(err).__name__})")
    except ValueError as err:
        st.error(f"😥 AI 응답을 읽지 못했어요. 다시 시도해 주세요. ({err})")


# ======================= 화면 그리기 =======================
def points_html(items: list, multi: bool) -> str:
    if not items:
        return '<p class="note">원문에서 근거 있는 항목을 찾지 못했어요.</p>'
    rows = []
    for it in items[:3]:
        tag = f'<span class="tag">{e(it.get("product"))}</span>' if multi and it.get("product") else ""
        ev = f'<ul><li>“{e(it.get("evidence"))}”</li></ul>' if it.get("evidence") else ""
        rows.append(f'<li><b><mark>{e(it.get("point"))}</mark></b>{tag}{ev}</li>')
    return '<ol class="pts">' + "".join(rows) + "</ol>"


def list_details(title: str, items: list, strike: bool = False) -> str:
    if not items:
        return ""
    lis = "".join(f"<li>{e(x)}</li>" for x in items)
    cls = ' class="strike"' if strike else ""
    return f"<details><summary>{e(title)}</summary><ul{cls}>{lis}</ul></details>"


def section(sid: str, title: str, sub: str = "") -> str:
    small = f"<small>{e(sub)}</small>" if sub else ""
    return f'<h2 class="sec" id="{sid}">{title}{small}</h2>'


def render(d: dict):
    r, meta = d["result"], d["meta"]
    out = []

    # 제목 + 출처 카드
    out.append(f'<h1 class="doc-title serif">{e(meta["title"])}</h1>')
    img = f'<img src="{e(meta["thumb"])}">' if meta.get("thumb") else ""
    go = '<span class="go">원본 보기 →</span>' if meta.get("url") else ""
    src_kind = {"youtube": "YouTube", "blog": "블로그", "text": "붙여넣은 글"}[d["kind"]]
    card = (
        f'{img}<div><div class="t">{e(meta["title"])}</div>'
        f'<div class="m">{e(meta.get("channel"))} · {src_kind}</div></div>{go}'
    )
    if meta.get("url"):
        out.append(f'<a class="src" href="{e(meta["url"])}" target="_blank">{card}</a>')
    else:
        out.append(f'<div class="src">{card}</div>')

    # 정보 칩
    ad = int(r.get("ad_score", 0) or 0)
    level, color = ("낮음", "green") if ad < 30 else (("보통", "amber") if ad < 70 else ("높음", "red"))
    multi = r.get("review_type") == "여러 제품 비교"
    ca = r.get("comment_analysis") if isinstance(r.get("comment_analysis"), dict) else None
    chips = [
        f'<span class="chip {color}">📢 광고 의심 <b>{ad}점 · {level}</b></span>',
        f'<span class="chip">📦 <b>{"여러 제품 비교" if multi else "단일 제품"}</b></span>',
        f'<span class="chip">💬 댓글 <b>{len(d["comments"])}개</b></span>',
        f'<span class="chip">🤖 {e(d["setting"])}</span>',
    ]
    out.append('<div class="chips">' + "".join(chips) + "</div>")

    # 01 한눈에 보기
    out.append(section("summary", "한눈에 보기"))
    out.append(f'<p class="lead">{e(r.get("summary"))}</p>')
    bar = {"green": "#3BA272", "amber": "#E0A800", "red": "#D2453B"}[color]
    out.append(f'<div class="gauge"><div style="width:{min(max(ad, 2), 100)}%;background:{bar}"></div></div>')
    out.append(f'<p class="note">📢 광고 의심 지수 {ad}점 — {e(r.get("ad_reason"))}</p>')
    if multi and r.get("products"):
        names = ", ".join(e(p) for p in r["products"][:6])
        out.append(f'<div class="callout">📦 여러 제품을 비교한 리뷰예요. 장단점마다 어떤 제품인지 표시했어요.<br><b>등장 제품</b> · {names}</div>')

    # 02 단점 / 03 장점
    out.append(section("cons", "👎 치명적 단점", "구매 결정에 영향이 큰 순서"))
    out.append(points_html(r.get("cons", []), multi))
    out.append(section("pros", "👍 핵심 장점"))
    out.append(points_html(r.get("pros", []), multi))

    # 04 댓글 감지
    out.append(section("comments", "💬 댓글 감지", "AI가 댓글마다 이름표를 붙이고, 개수는 코드가 셌어요"))
    if ca and d["comments"]:
        stats = [
            ("🗣️ 광고라고 지적", ca.get("ad_suspicion_count", 0)),
            ("🤖 바이럴 의심", ca.get("viral_count", 0)),
            ("🙋 실사용 경험", ca.get("experience_count", 0)),
            ("💬 기타", ca.get("other_count", 0)),
        ]
        out.append('<div class="stats">' + "".join(f"<div class='stat'><small>{a}</small><div>{b}</div></div>" for a, b in stats) + "</div>")
        if ca.get("unlabeled"):
            out.append(f'<p class="note">※ AI가 이름표를 빠뜨린 {ca["unlabeled"]}개는 기타로 셌어요.</p>')
        if ca.get("mismatch"):
            out.append(f'<div class="callout">⚠️ <b>리뷰와 댓글이 다른 점</b><br>{e(ca["mismatch"])}</div>')
        ops = ca.get("real_user_opinions") or []
        if ops:
            lis = "".join(
                f'<li>{"👎" if o.get("type") == "단점" else "👍"} {e(o.get("opinion"))}</li>' for o in ops[:5]
            )
            out.append(f'<p class="note" style="margin-top:1rem"><b>실사용자 경험 요약</b></p><ul class="plain">{lis}</ul>')
        out.append(list_details(f"🗣️ 광고라고 지적한 댓글 ({ca.get('ad_suspicion_count', 0)})", ca.get("ad_suspicion_examples", [])))
        out.append(list_details(f"🤖 바이럴 의심 댓글 ({ca.get('viral_count', 0)})", ca.get("viral_examples", [])))
        out.append(list_details(f"🙋 실사용 경험 댓글 ({ca.get('experience_count', 0)})", ca.get("experience_comments", [])))
    else:
        out.append('<p class="note">분석한 댓글이 없어요. 블로그나 붙여넣은 글은 왼쪽 <b>💬 댓글 직접 넣기</b>에 댓글을 넣으면 함께 분석해요.</p>')

    # 05 걸러낸 광고 문장
    ads = r.get("ad_sentences", [])
    out.append(section("ads", "🚫 걸러낸 광고 문장", f"{len(ads)}개 · 리뷰어가 직접 한 말 중 장단점 판단에서 뺀 문장"))
    if ads:
        out.append("<ul class='plain'>" + "".join(f"<li><s style='color:#8E8D88'>{e(a)}</s></li>" for a in ads) + "</ul>")
    else:
        out.append('<p class="note">걸러낸 광고 문장이 없어요.</p>')

    # 06 원문 확인
    out.append(section("source", "🔎 원문 확인", "AI가 실제로 읽은 자료"))
    out.append(list_details(f"💬 가져온 댓글 전체 ({len(d['comments'])}개, 답글 포함)", d["comments"]))
    txt = d["text"]
    out.append(f"<details><summary>📄 분석한 원문 (자막·본문) 앞부분</summary><p>{e(txt[:2000])}{'…' if len(txt) > 2000 else ''}</p></details>")

    u = d["usage"]
    out.append(
        f'<div class="foot">AI 분석 결과는 틀릴 수 있어요. 구매 전에 근거 문장을 꼭 직접 확인하세요.<br>'
        f'사용 토큰 · 입력 {u["input_tokens"]:,} / 출력 {u["output_tokens"]:,}</div>'
    )
    body = "\n".join(x for x in out if x)
    body = "\n".join(line for line in body.splitlines() if line.strip())
    st.markdown(body, unsafe_allow_html=True)


if data:
    render(data)
    handle(st.chat_input("다른 리뷰 링크나 글을 붙여넣으세요"))
else:
    logo_html = f'<img class="logo" src="{LOGO}" alt="RealPick AI">' if LOGO else '<h1 class="serif">RealPick AI</h1>'
    st.markdown(
        f'<div class="hero">{logo_html}<div class="tag">광고는 걸러내고, <b>진짜 장단점만</b> 골라 드려요.</div></div>',
        unsafe_allow_html=True,
    )
    with st.form("main_input", border=False):
        q = st.text_area(
            "리뷰 입력",
            placeholder="유튜브·블로그 리뷰 링크를 붙여넣거나, 리뷰 글을 그대로 붙여넣으세요",
            height=96,
            label_visibility="collapsed",
        )
        c1, c2 = st.columns([4, 1])
        c1.markdown('<div class="hint">예) https://youtu.be/... · Ctrl+Enter로 바로 분석</div>', unsafe_allow_html=True)
        submitted = c2.form_submit_button("분석하기 →", type="primary", use_container_width=True)
    if submitted:
        handle(q)
    st.markdown(
        '<div class="steps">'
        '<div class="step"><b>① 붙여넣기</b><small>유튜브 링크, 블로그 링크, 리뷰 글 아무거나</small></div>'
        '<div class="step"><b>② 광고 걸러내기</b><small>협찬 고지·할인 링크·과장 표현 제외</small></div>'
        '<div class="step"><b>③ 댓글 교차 확인</b><small>실사용 경험과 바이럴 댓글 구분</small></div>'
        "</div>",
        unsafe_allow_html=True,
    )
