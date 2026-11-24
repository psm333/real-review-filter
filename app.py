"""RealPick AI — Streamlit 앱.  실행: py -m streamlit run app.py

화면은 세 가지 모드로 나뉜다.
  1) 리뷰 하나 분석   : 링크/글 하나 → 광고 필터링, 장단점, 신뢰 점수, 댓글 감지, 질문하기, 공유 카드
  2) 여러 리뷰 종합   : 링크 여러 개 또는 제품 이름 → 공통 단점 순위
  3) 제품 비교        : 제품 A vs B → 상황별 추천
"""
import base64
import hashlib
import html
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import openai
import streamlit as st
from dotenv import load_dotenv

from analyzer import MODELS, PROMPT_LABELS, aggregate, analyze, ask, compare, locate_time, trust_score
from fetchers import (
    FetchError,
    extract_video_id,
    get_blog_post,
    get_youtube_comments,
    get_youtube_meta,
    get_youtube_segments,
    search_youtube_reviews,
)
from share import make_card

load_dotenv()

ROOT = Path(__file__).parent
ASSETS = ROOT / "assets"
CACHE_DIR = ROOT / ".cache"  # 한 번 분석한 결과를 저장해 두는 곳 (GitHub에는 안 올라감)
MAX_LINKS = 5


def get_secret(name: str, default: str = "") -> str:
    try:
        return st.secrets[name]
    except Exception:
        return os.getenv(name, default)


OPENAI_KEY = get_secret("OPENAI_API_KEY")
YOUTUBE_KEY = get_secret("YOUTUBE_API_KEY")
APP_URL = get_secret("APP_URL", "http://localhost:8501").rstrip("/")


def img_b64(name: str) -> str:
    try:
        return "data:image/png;base64," + base64.b64encode((ASSETS / name).read_bytes()).decode()
    except OSError:
        return ""


LOGO = img_b64("logo.png")
ICON_PATH = ASSETS / "icon.png"
st.set_page_config(
    page_title="RealPick AI · 진짜 리뷰만 골라보기",
    page_icon=str(ICON_PATH) if ICON_PATH.exists() else "🔍",
    layout="centered",
    initial_sidebar_state="expanded",
)


def e(text) -> str:
    """AI·댓글 글자를 화면에 넣기 전에 HTML 특수문자를 막고, 줄바꿈은 <br>로 바꾼다."""
    return html.escape(str(text or "")).replace("\r", "").replace("\n", "<br>")


def show_html(parts) -> None:
    """HTML 조각을 화면에 그린다. Streamlit은 빈 줄에서 HTML이 끊기므로 빈 줄을 없앤다."""
    body = "\n".join(p for p in parts if p)
    st.markdown("\n".join(line for line in body.splitlines() if line.strip()), unsafe_allow_html=True)


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
.block-container { max-width: 800px; padding-top: 3.2rem; padding-bottom: 7rem; }
/* 사이드바 */
section[data-testid="stSidebar"] { background: var(--side); border-right: 1px solid var(--line); }
section[data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }
.brand { margin-bottom: 1.1rem; } .brand img { height: 30px; }
.toc-title { font-size:.88rem; font-weight:700; line-height:1.4; margin:.2rem 0 .8rem; }
.toc a { display:flex; gap:10px; padding:6px 4px; font-size:.86rem; color:var(--sub) !important; text-decoration:none !important; border-radius:6px; }
.toc a:hover { background:#E9E8E3; color:var(--ink) !important; }
.toc a span { color:#A9A8A2; font-variant-numeric: tabular-nums; }
.side-note { font-size:.78rem; color:var(--sub); line-height:1.6; }
/* 첫 화면 */
.hero { padding: 10vh 0 1.2rem; text-align:center; }
.hero img.logo { width: 340px; max-width: 80%; }
.hero .tag { margin-top: .9rem; font-size: 1.05rem; color: var(--sub); }
.hero .tag b { color: var(--ink); }
[data-testid="stMain"] [data-testid="stRadio"] > div[role="radiogroup"], .main [data-testid="stRadio"] > div[role="radiogroup"] { justify-content:center; gap:6px; flex-wrap:wrap; }
[data-testid="stMain"] [data-testid="stRadio"] label, .main [data-testid="stRadio"] label { background:#fff; border:1px solid var(--line); border-radius:999px; padding:4px 14px 4px 8px; }
[data-testid="stForm"] { background:#fff; border:1px solid var(--line) !important; border-radius:18px; padding: 14px 16px 10px; box-shadow: 0 10px 30px rgba(0,0,0,.06); }
[data-testid="stForm"] textarea { font-size:1rem !important; }
[data-testid="stForm"] button[kind="primaryFormSubmit"], [data-testid="stForm"] button[kind="primary"] { background:#1F1F1C; border:none; border-radius:10px; }
.hint { font-size:.78rem; color:#A9A8A2; padding-top:.55rem; }
.steps { display:flex; gap:10px; margin-top:2rem; flex-wrap:wrap; }
.step { flex:1; min-width:180px; border:1px solid var(--line); border-radius:12px; padding:14px 16px; }
.step b { display:block; font-size:.92rem; margin-bottom:4px; }
.step small { color:var(--sub); font-size:.8rem; line-height:1.5; }
.serif { font-family: 'Noto Serif KR', serif !important; letter-spacing:-0.02em; }
/* 결과 문서 */
.doc-title { font-size: 2.1rem; line-height:1.3; font-weight:700; margin: .3rem 0 1.2rem; color:var(--ink); }
.src { display:flex; gap:14px; align-items:center; background:#fff; border:1px solid var(--line); border-radius:12px; padding:10px; text-decoration:none !important; margin-bottom:8px; }
.src img { width:120px; height:68px; object-fit:cover; border-radius:8px; flex-shrink:0; background:#eee; }
.src .t { font-weight:600; font-size:.92rem; color:var(--ink); line-height:1.4; display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical; overflow:hidden; }
.src .m { font-size:.78rem; color:var(--sub); margin-top:4px; }
.src .go { margin-left:auto; font-size:.8rem; color:var(--sub); white-space:nowrap; padding-right:6px; }
.src .no { font-weight:800; color:#A9A8A2; padding:0 4px 0 6px; }
.chips { display:flex; gap:6px; flex-wrap:wrap; margin: 12px 0 2rem; }
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
.trust { display:flex; gap:18px; align-items:center; background:#fff; border:1px solid var(--line); border-radius:12px; padding:14px 18px; margin:.8rem 0; }
.trust .num { font-size:2.2rem; font-weight:800; line-height:1; } .trust .num small { font-size:.9rem; color:var(--sub); font-weight:600; }
.trust .lab { font-size:.75rem; color:var(--sub); margin-top:4px; }
.trust ul { margin:0; padding-left:1rem; font-size:.82rem; color:var(--sub); line-height:1.6; }
ol.pts { padding-left: 1.3rem; margin:0; }
ol.pts > li { margin: 0 0 1.05rem; font-size:1rem; line-height:1.6; }
ol.pts > li::marker { font-weight:700; }
ol.pts .tag { font-size:.72rem; background:#EFEDE7; color:var(--sub); border-radius:5px; padding:1px 7px; margin-left:6px; font-weight:500; }
ol.pts ul { margin:.35rem 0 0; padding-left:1.1rem; }
ol.pts ul li { font-size:.88rem; color:var(--sub); line-height:1.65; list-style: disc; }
.fit { font-size:.72rem; border-radius:5px; padding:1px 7px; margin-left:6px; font-weight:700; }
.fit.high { background:#FBE3E1; color:#B0362D; } .fit.mid { background:#FFF3CC; color:#8A6400; } .fit.low { background:#EFEDE7; color:#6B6B66; }
a.ts { font-size:.74rem; margin-left:6px; padding:1px 8px; border-radius:99px; background:#1F1F1C; color:#fff !important; text-decoration:none !important; white-space:nowrap; }
.cnt { display:inline-flex; align-items:center; gap:6px; font-size:.76rem; color:var(--sub); margin-left:6px; }
.cnt .bar { width:70px; height:6px; background:#ECEAE4; border-radius:99px; overflow:hidden; display:inline-block; }
.cnt .bar div { height:100%; background:#D2453B; }
.cnt.good .bar div { background:#3BA272; }
ul.check { padding-left:0; } ul.check li { list-style:none; font-size:.95rem; line-height:1.7; margin-bottom:.35rem; }
.stats { display:grid; grid-template-columns: repeat(4, 1fr); gap:8px; margin: .4rem 0 .6rem; }
.stat { background:#fff; border:1px solid var(--line); border-radius:10px; padding:10px 12px; }
.stat small { font-size:.75rem; color:var(--sub); }
.stat div { font-size:1.35rem; font-weight:800; margin-top:2px; }
ul.plain { padding-left:1.1rem; } ul.plain li { font-size:.9rem; line-height:1.65; margin-bottom:.25rem; }
.picks { display:grid; grid-template-columns: repeat(3, 1fr); gap:8px; }
.pick { background:#fff; border:1px solid var(--line); border-radius:12px; padding:12px 14px; font-size:.85rem; line-height:1.55; color:var(--sub); }
.pick .who { color:var(--ink); font-weight:600; }
.pick .p { display:block; font-size:1.15rem; font-weight:800; color:var(--ink); margin:.3rem 0; }
.vs { display:grid; grid-template-columns: 1fr 1fr; gap:12px; }
.vs .col { background:#fff; border:1px solid var(--line); border-radius:12px; padding:12px 16px; }
.vs h4 { margin:.2rem 0 .6rem; font-size:1rem; }
.vs .k { font-size:.78rem; color:var(--sub); font-weight:700; margin:.6rem 0 .2rem; }
.vs li { font-size:.86rem; line-height:1.55; margin-bottom:.3rem; }
table.diff { width:100%; border-collapse:collapse; font-size:.88rem; background:#fff; }
table.diff th, table.diff td { border-bottom:1px solid var(--line); padding:9px 8px; text-align:left; vertical-align:top; line-height:1.55; }
table.diff th { color:var(--sub); font-weight:600; font-size:.8rem; }
.qa-q { font-weight:700; margin-top:1rem; } .qa-q::before { content:"Q. "; color:#A9A8A2; }
.qa-a { background:#fff; border:1px solid var(--line); border-radius:10px; padding:10px 14px; font-size:.92rem; line-height:1.7; margin-top:.4rem; }
details { background:#fff; border:1px solid var(--line); border-radius:10px; padding: 10px 14px; margin: .5rem 0; }
details summary { cursor:pointer; font-size:.9rem; font-weight:600; }
details ul, details ol { margin:.6rem 0 0; padding-left:1.1rem; }
details li { font-size:.86rem; color:var(--sub); line-height:1.6; margin-bottom:.25rem; }
details p { font-size:.86rem; color:var(--sub); line-height:1.7; }
.guide { background:#fff; border:1px solid var(--line); border-left:3px solid #3B7DD8; border-radius:10px; padding: 10px 14px; margin: 0 0 1.4rem; }
.guide summary { cursor:pointer; font-size:.9rem; font-weight:700; }
.guide ul, .guide ol { margin:.6rem 0 .2rem; padding-left:1.2rem; }
.guide li { font-size:.86rem; color:var(--ink); line-height:1.7; margin-bottom:.2rem; }
.guide .eg { display:inline-block; font-size:.72rem; padding:1px 8px; border-radius:99px; background:#1F1F1C; color:#fff; white-space:nowrap; }
.guide-land { margin-top:1.4rem; }
.down { text-align:center; font-size:.85rem; color:#3B7DD8; font-weight:700; margin-top:.6rem; }
.foot { text-align:center; font-size:.76rem; color:#A9A8A2; margin-top:3rem; }
[data-testid="stBottom"] > div { background: transparent; }
[data-testid="stChatInput"] { border-radius: 14px !important; border:1px solid var(--line) !important; background:#fff !important; box-shadow: 0 6px 24px rgba(0,0,0,.06); }
@media (max-width: 640px) { .picks, .vs, .stats { grid-template-columns: 1fr 1fr; } .doc-title { font-size:1.6rem; } }
</style>
"""
st.markdown("\n".join(line for line in CSS.splitlines() if line.strip()), unsafe_allow_html=True)

PURPOSES = ["선택 안 함", "게임용", "사무·공부용", "출퇴근·이동용", "운동용", "선물용", "가성비가 중요함"]
MODES = ["🔎 리뷰 하나 분석", "📚 여러 리뷰 종합", "⚖️ 제품 비교"]


# ======================= 저장(캐시): 같은 영상은 바로 =======================
def _key(*parts) -> str:
    return hashlib.sha1(json.dumps(parts, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def cache_get(kind: str, key: str):
    path = CACHE_DIR / f"{kind}-{key}.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def cache_put(kind: str, key: str, value) -> None:
    try:
        CACHE_DIR.mkdir(exist_ok=True)
        (CACHE_DIR / f"{kind}-{key}.json").write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass  # 저장을 못 해도 분석은 계속한다


# ======================= 자료 모으기 (화면을 건드리지 않음 → 여러 개 동시에 가능) =======================
def detect_kind(q: str) -> str:
    if re.search(r"(youtube\.com|youtu\.be)/", q):
        return "youtube"
    if re.match(r"https?://", q):
        return "blog"
    return "text"


def collect(query: str, with_comments: bool) -> dict:
    """링크나 글에서 분석할 자료를 모은다. 링크는 저장해 두고 다음에 바로 쓴다."""
    kind = detect_kind(query)
    if kind == "text":
        if len(query) < 50:
            raise FetchError("글이 너무 짧아요. 리뷰 링크나 50자 이상의 리뷰 글을 넣어 주세요.")
        first = query.strip().splitlines()[0][:40]
        meta = {"title": first + ("…" if len(first) >= 40 else ""), "channel": "직접 붙여넣은 글", "thumb": "", "url": ""}
        return {"kind": kind, "meta": meta, "text": query, "comments": [], "segments": [], "vid": ""}

    key = _key(query, with_comments and bool(YOUTUBE_KEY))
    hit = cache_get("src", key)
    if hit:
        return hit
    if kind == "youtube":
        vid = extract_video_id(query)
        meta = get_youtube_meta(vid)
        segments = get_youtube_segments(vid)
        comments = []
        if with_comments and YOUTUBE_KEY:
            try:
                comments = get_youtube_comments(vid, YOUTUBE_KEY)
            except FetchError:
                comments = []
        src = {"kind": kind, "meta": meta, "text": " ".join(s["text"] for s in segments),
               "comments": comments, "segments": segments, "vid": vid}
    else:
        meta, text = get_blog_post(query)
        src = {"kind": kind, "meta": meta, "text": text, "comments": [], "segments": [], "vid": ""}
    cache_put("src", key, src)
    return src


def analyze_cached(src: dict, s: dict) -> tuple[dict, dict, bool]:
    """같은 자료·같은 설정이면 저장된 결과를 바로 돌려준다 (기다림 0초, 비용 0원)."""
    key = _key(src["text"], src["comments"], s["version"], s["model"], s["purpose"])
    hit = cache_get("ana", key)
    if hit:
        return hit["result"], {"input_tokens": 0, "output_tokens": 0}, True
    result, usage = analyze(src["text"], src["comments"], s["version"], s["model"], api_key=OPENAI_KEY, purpose=s["purpose"])
    cache_put("ana", key, {"result": result})
    return result, usage, False


def friendly_error(err: Exception) -> str:
    if isinstance(err, FetchError):
        return str(err)
    if isinstance(err, openai.AuthenticationError):
        return "OpenAI API 키가 올바르지 않아요."
    if isinstance(err, openai.RateLimitError):
        return "요청이 너무 많거나 크레딧이 부족해요. 잠시 후 다시 시도하거나 결제 잔액을 확인하세요."
    if isinstance(err, openai.APIError):
        return f"AI 서버 오류가 났어요. 잠시 후 다시 시도해 주세요. ({type(err).__name__})"
    if isinstance(err, ValueError):
        return f"AI 응답을 읽지 못했어요. 다시 시도해 주세요. ({err})"
    return f"알 수 없는 오류가 났어요. ({type(err).__name__})"


def _try_collect(link: str, s: dict) -> dict:
    try:
        return {"ok": True, "link": link, "src": collect(link, s["comments"])}
    except Exception as err:
        return {"ok": False, "link": link, "error": friendly_error(err)}


def _try_analyze(item: dict, s: dict) -> dict:
    try:
        src = item["src"]
        result, usage, cached = analyze_cached(src, s)
        score, reasons = trust_score(result, src["text"], src["comments"])
        return dict(item, result=result, usage=usage, cached=cached, trust=score, trust_reasons=reasons)
    except Exception as err:
        return {"ok": False, "link": item["link"], "error": friendly_error(err)}


def run_many(links: list[str], s: dict, need: int) -> tuple[list[dict], list[dict]]:
    """여러 리뷰를 동시에 처리한다. 먼저 자료를 모으고(자막 없는 영상은 여기서 빠짐),
    성공한 것 중 앞에서부터 need개만 AI로 분석한다 → 필요 없는 분석 비용을 쓰지 않는다."""
    with ThreadPoolExecutor(max_workers=5) as pool:
        collected = list(pool.map(lambda lk: _try_collect(lk, s), links))
    good = [c for c in collected if c["ok"]][:need]
    fails = [c for c in collected if not c["ok"]]
    with ThreadPoolExecutor(max_workers=5) as pool:
        analyzed = list(pool.map(lambda c: _try_analyze(c, s), good))
    oks = [a for a in analyzed if a.get("ok")]
    fails += [a for a in analyzed if not a.get("ok")]
    return oks, fails


def resolve_links(query: str) -> tuple[list[str], str]:
    """입력이 링크 여러 개면 그대로, 제품 이름이면 유튜브에서 리뷰를 찾는다."""
    links = re.findall(r"https?://\S+", query)
    if links:
        return list(dict.fromkeys(links))[:MAX_LINKS], ""
    name = query.strip()
    if len(name) < 2:
        raise FetchError("제품 이름이나 리뷰 링크를 넣어 주세요.")
    return search_youtube_reviews(name, YOUTUBE_KEY, MAX_LINKS)[: MAX_LINKS + 3], name


def sum_usage(*usages) -> dict:
    return {
        "input_tokens": sum(u.get("input_tokens", 0) for u in usages),
        "output_tokens": sum(u.get("output_tokens", 0) for u in usages),
    }


# ======================= 모드별 실행 =======================
def run_single(query: str, s: dict, extra_comments: list[str]):
    with st.status("리뷰를 분석하고 있어요...", expanded=True) as status:
        st.write("📥 자료를 가져오는 중 (자막·댓글·본문)")
        src = collect(query, s["comments"])
        if extra_comments:
            src = dict(src, comments=src["comments"] + extra_comments)
        st.write(f"🤖 AI가 광고를 걸러내고 장단점을 고르는 중 (댓글 {len(src['comments'])}개)")
        result, usage, cached = analyze_cached(src, s)
        score, reasons = trust_score(result, src["text"], src["comments"])
        status.update(label="분석 완료!", state="complete", expanded=False)
    st.session_state["data"] = {
        "mode": "single", "src": src, "result": result, "usage": usage, "cached": cached,
        "trust": score, "trust_reasons": reasons, "settings": s, "qa": [],
    }


def run_multi(query: str, s: dict):
    with st.status("여러 리뷰를 모으고 있어요...", expanded=True) as status:
        links, name = resolve_links(query)
        if name:
            st.write(f"🔍 유튜브에서 '{name}' 리뷰를 찾았어요")
        st.write(f"🤖 리뷰 {min(len(links), MAX_LINKS)}개를 동시에 분석하는 중")
        oks, fails = run_many(links, s, MAX_LINKS)
        if not oks:
            raise FetchError("분석할 수 있는 리뷰가 없었어요. " + (fails[0]["error"] if fails else ""))
        st.write("🧩 공통 단점·장점을 묶는 중")
        agg, agg_usage = aggregate([{"meta": o["src"]["meta"], "result": o["result"]} for o in oks],
                                   s["purpose"], s["model"], OPENAI_KEY)
        status.update(label="종합 완료!", state="complete", expanded=False)
    st.session_state["data"] = {
        "mode": "multi", "name": name, "reviews": oks, "fails": fails, "agg": agg, "settings": s,
        "usage": sum_usage(agg_usage, *[o["usage"] for o in oks]),
    }


def run_compare(qa: str, qb: str, s: dict):
    if not qa.strip() or not qb.strip():
        raise FetchError("제품 A와 제품 B를 모두 넣어 주세요.")
    with st.status("두 제품의 리뷰를 모으고 있어요...", expanded=True) as status:
        sides = []
        for label, q in (("A", qa), ("B", qb)):
            links, name = resolve_links(q)
            st.write(f"🤖 제품 {label}: 리뷰 {min(len(links), MAX_LINKS)}개 분석 중")
            oks, fails = run_many(links, s, MAX_LINKS)
            if not oks:
                raise FetchError(f"제품 {label}의 리뷰를 분석하지 못했어요. " + (fails[0]["error"] if fails else ""))
            agg, u = aggregate([{"meta": o["src"]["meta"], "result": o["result"]} for o in oks],
                               s["purpose"], s["model"], OPENAI_KEY)
            sides.append({"label": label, "name": name or f"제품 {label}", "reviews": oks, "fails": fails,
                          "agg": agg, "usage": sum_usage(u, *[o["usage"] for o in oks])})
        st.write("⚖️ 두 제품을 비교하는 중")
        cmp, cu = compare(sides[0]["name"], sides[0]["agg"], sides[1]["name"], sides[1]["agg"],
                          s["purpose"], s["model"], OPENAI_KEY)
        status.update(label="비교 완료!", state="complete", expanded=False)
    st.session_state["data"] = {"mode": "compare", "sides": sides, "cmp": cmp, "settings": s,
                                "usage": sum_usage(cu, sides[0]["usage"], sides[1]["usage"])}


def handle(fn, *args) -> None:
    if not OPENAI_KEY:
        st.error("OPENAI_API_KEY 가 없어서 분석할 수 없어요. README를 확인하세요.")
        return
    try:
        fn(*args)
    except Exception as err:
        st.error(f"😥 {friendly_error(err)}")
        return
    st.rerun()


# ======================= 화면 조각 =======================
def section(sid: str, title: str, sub: str = "") -> str:
    small = f"<small>{e(sub)}</small>" if sub else ""
    return f'<h2 class="sec" id="{sid}">{title}{small}</h2>'


def mmss(sec: float) -> str:
    sec = int(sec)
    return f"{sec // 3600}:{sec % 3600 // 60:02d}:{sec % 60:02d}" if sec >= 3600 else f"{sec // 60}:{sec % 60:02d}"


def fit_badge(it: dict) -> str:
    fit = it.get("fit")
    cls = {"높음": "high", "보통": "mid", "낮음": "low"}.get(fit)
    if not cls:
        return ""
    title = e(it.get("fit_reason"))
    return f'<span class="fit {cls}" title="{title}">나에게 {e(fit)}</span>'


def points_html(items: list, multi: bool, src: dict | None = None, limit: int = 3) -> str:
    if not items:
        return '<p class="note">원문에서 근거 있는 항목을 찾지 못했어요.</p>'
    rows = []
    for it in items[:limit]:
        tag = f'<span class="tag">{e(it.get("product"))}</span>' if multi and it.get("product") else ""
        sub = []
        ev = it.get("evidence")
        if ev:
            ts = ""
            if src and src.get("vid") and src.get("segments"):
                t = locate_time(ev, src["segments"])
                if t is not None:
                    url = f"https://www.youtube.com/watch?v={src['vid']}&t={int(t)}s"
                    ts = f'<a class="ts" href="{e(url)}" target="_blank" title="유튜브에서 {mmss(t)} 장면부터 새 탭으로 재생해요">▶ {mmss(t)} 장면 보기</a>'
            sub.append(f"<li>“{e(ev)}”{ts}</li>")
        if it.get("fit_reason"):
            sub.append(f"<li>나에게 {e(it.get('fit'))}: {e(it['fit_reason'])}</li>")
        subs = "<ul>" + "".join(sub) + "</ul>" if sub else ""
        rows.append(f'<li><b><mark>{e(it.get("point"))}</mark></b>{tag}{fit_badge(it)}{subs}</li>')
    return '<ol class="pts">' + "".join(rows) + "</ol>"


def counted_html(items: list, total: int, good: bool, limit: int = 5) -> str:
    """여러 리뷰 종합: '5개 중 4개 리뷰'를 막대와 함께 보여준다."""
    if not items:
        return '<p class="note">여러 리뷰에서 공통으로 나온 항목이 없어요.</p>'
    rows = []
    for it in items[:limit]:
        n = it.get("count", 0)
        cls = "cnt good" if good else "cnt"
        bar = f'<span class="{cls}"><span class="bar"><div style="width:{round(n / max(total, 1) * 100)}%"></div></span>리뷰 {n}/{total}개 · {", ".join("#" + str(x) for x in it.get("sources", []))}</span>'
        sub = []
        if it.get("detail"):
            sub.append(f"<li>{e(it['detail'])}</li>")
        if it.get("fit_reason"):
            sub.append(f"<li>나에게 {e(it.get('fit'))}: {e(it['fit_reason'])}</li>")
        if it.get("caution"):
            sub.append(f"<li>⚠️ {e(it['caution'])}</li>")
        subs = "<ul>" + "".join(sub) + "</ul>" if sub else ""
        rows.append(f'<li><b><mark>{e(it.get("point"))}</mark></b>{fit_badge(it)}<br>{bar}{subs}</li>')
    return '<ol class="pts">' + "".join(rows) + "</ol>"


def checklist_html(items: list) -> str:
    if not items:
        return '<p class="note">체크리스트를 만들지 못했어요.</p>'
    return '<ul class="check">' + "".join(f"<li>☐ {e(c)}</li>" for c in items[:5]) + "</ul>"


def trust_html(score: int, reasons: list) -> str:
    color = "#2E8B57" if score >= 70 else ("#C98A00" if score >= 40 else "#D2453B")
    word = "믿을 만해요" if score >= 70 else ("가려서 보세요" if score >= 40 else "조심하세요")
    if reasons:
        lis = "".join(f"<li>{e(r)} ({d:+d})</li>" for r, d in reasons)
    else:
        lis = "<li>깎인 점수가 없어요</li>"
    return (
        f'<div class="trust"><div><div class="num" style="color:{color}">{score}<small>/100</small></div>'
        f'<div class="lab">신뢰 점수 · {word}</div></div>'
        f'<ul>{lis}<li style="list-style:none;margin-left:-1rem;color:#A9A8A2">100점에서 시작해 이유마다 깎는 공식이에요 (AI가 정한 숫자 아님)</li></ul></div>'
    )


def src_card(src: dict, no: int | None = None, extra: str = "") -> str:
    meta = src["meta"]
    img = f'<img src="{e(meta["thumb"])}">' if meta.get("thumb") else ""
    num = f'<span class="no">#{no}</span>' if no else ""
    kind = {"youtube": "YouTube", "blog": "블로그", "text": "붙여넣은 글"}[src["kind"]]
    go = '<span class="go">원본 보기 →</span>' if meta.get("url") else ""
    inner = f'{num}{img}<div><div class="t">{e(meta["title"])}</div><div class="m">{e(meta.get("channel"))} · {kind}{extra}</div></div>{go}'
    if meta.get("url"):
        return f'<a class="src" href="{e(meta["url"])}" target="_blank" id="rv{no or 0}">{inner}</a>'
    return f'<div class="src" id="rv{no or 0}">{inner}</div>'


def ad_chip(ad: int) -> str:
    level, color = ("낮음", "green") if ad < 30 else (("보통", "amber") if ad < 70 else ("높음", "red"))
    return f'<span class="chip {color}">📢 광고 의심 <b>{ad}점 · {level}</b></span>'


def list_details(title: str, items: list) -> str:
    if not items:
        return ""
    return f"<details><summary>{e(title)}</summary><ul>" + "".join(f"<li>{e(x)}</li>" for x in items) + "</ul></details>"


def foot(usage: dict, cached: bool = False) -> str:
    note = "⚡ 저장된 결과를 바로 보여줬어요 (추가 비용 0원)" if cached else f'사용 토큰 · 입력 {usage["input_tokens"]:,} / 출력 {usage["output_tokens"]:,}'
    return f'<div class="foot">AI 분석 결과는 틀릴 수 있어요. 구매 전에 근거 문장을 꼭 직접 확인하세요.<br>{note}</div>'


# ======================= 처음 쓰는 사람을 위한 안내 =======================
def guide(kind: str, has_video: bool = False) -> str:
    if kind == "single":
        items = []
        if has_video:
            items.append('<li><span class="eg">▶ 1:35 장면 보기</span> 를 누르면 유튜브가 <b>새 탭</b>에서 리뷰어가 <b>그 말을 한 장면부터</b> 재생돼요.</li>')
        items += [
            '<li><mark>노란 형광펜</mark>이 핵심이고, 아래 회색 “따옴표 글”이 리뷰 원문에서 그대로 가져온 <b>근거</b>예요.</li>',
            '<li><span class="fit high">나에게 높음</span> 은 내가 고른 <b>사용 목적</b>에 치명적인 단점이에요. 목적을 고르지 않으면 안 보여요.</li>',
            '<li><b>신뢰 점수</b>는 AI가 정한 게 아니라 공식으로 계산한 점수예요. 옆에 깎인 이유가 적혀 있어요.</li>',
            '<li>왼쪽 <b>목차</b>를 누르면 그 부분으로 바로 이동해요.</li>',
            '<li>궁금한 게 있으면 <b>맨 아래 입력창</b>에 물어보세요. 리뷰에 있는 내용으로만 답해요.</li>',
            '<li><b>📤 공유하기</b>에서 결과를 카드 이미지로 저장해 카톡으로 보낼 수 있어요.</li>',
        ]
        title = "💡 처음이라면: 이 화면 읽는 법"
    elif kind == "multi":
        items = [
            '<li>막대와 <b>리뷰 3/5개</b>는 “분석한 리뷰 5개 중 3개가 이 단점을 말했다”는 뜻이에요. 많을수록 믿을 만해요.</li>',
            '<li><b>#1, #2</b> 는 아래 <b>📄 분석한 리뷰</b>의 번호예요. 어느 리뷰에서 나온 말인지 확인할 수 있어요.</li>',
            '<li>리뷰마다 <b>신뢰 점수</b>와 <b>광고 의심 점수</b>가 붙어 있어요. 점수가 낮은 리뷰의 칭찬은 걸러서 보세요.</li>',
            '<li>리뷰 하나를 자세히 보고 싶으면 그 링크를 <b>🔎 리뷰 하나 분석</b>에 넣으면 돼요 (▶ 장면 보기, 질문하기 가능).</li>',
        ]
        title = "💡 처음이라면: 이 화면 읽는 법"
    else:
        items = [
            '<li><b>🎯 상황별 추천</b>에서 내 상황에 맞는 카드를 찾아보세요. 한쪽이 무조건 낫다고 하지 않아요.</li>',
            '<li><b>2/5</b> 같은 숫자는 “그 제품 리뷰 5개 중 2개가 말했다”는 뜻이에요.</li>',
            '<li>리뷰 수가 적은 쪽은 결과가 치우칠 수 있어요. 위의 ⚠️ 주의 문구를 꼭 확인하세요.</li>',
        ]
        title = "💡 처음이라면: 이 화면 읽는 법"
    return f'<details class="guide" open><summary>{title}</summary><ul>' + "".join(items) + "</ul></details>"


LANDING_GUIDE = (
    '<details class="guide guide-land"><summary>❓ 처음이신가요? 1분 사용법</summary><ol>'
    '<li><b>모드 고르기</b> — 🔎 리뷰 하나 / 📚 여러 리뷰 종합 / ⚖️ 제품 비교 중 하나를 눌러요.</li>'
    '<li><b>넣기</b> — 유튜브·블로그 <b>링크</b>를 붙여넣거나, 리뷰 <b>글</b>을 통째로 붙여넣어요. 여러 리뷰 종합·제품 비교는 <b>제품 이름</b>만 넣어도 돼요.</li>'
    '<li><b>🎯 사용 목적</b>(선택) — “게임용”, “출퇴근용”을 고르거나 “손이 작은 편”처럼 직접 적으면, 나에게 치명적인 단점이 위로 와요.</li>'
    '<li><b>분석하기 →</b> 를 누르면 10~20초 뒤 결과가 나와요 (여러 리뷰는 1~2분). 한 번 본 영상은 다음에 바로 나와요.</li>'
    '<li><b>결과 보기</b> — <span class="eg">▶ 1:35 장면 보기</span> 를 누르면 유튜브에서 그 장면부터 재생되고, 맨 아래 입력창에 리뷰에 대해 질문할 수 있어요.</li>'
    '<li><b>유튜브에서 바로</b> — 왼쪽 <b>🧩 유튜브에서 바로 분석하기</b>의 북마크를 만들어 두면, 영상을 보다가 한 번 눌러 분석할 수 있어요.</li>'
    "</ol></details>"
)


# ======================= 화면: 리뷰 하나 =======================
SINGLE_TOC = [("summary", "한눈에 보기"), ("cons", "치명적 단점"), ("pros", "핵심 장점"), ("check", "사기 전에 확인할 것"),
              ("comments", "댓글 감지"), ("ads", "걸러낸 광고 문장"), ("share", "공유하기"), ("qa", "리뷰에 물어보기"),
              ("source", "원문 확인")]


def render_single(d: dict):
    r, src, s = d["result"], d["src"], d["settings"]
    multi = r.get("review_type") == "여러 제품 비교"
    ca = r.get("comment_analysis") if isinstance(r.get("comment_analysis"), dict) else None
    ad = int(r.get("ad_score", 0) or 0)
    out = [f'<h1 class="doc-title serif">{e(src["meta"]["title"])}</h1>', src_card(src)]
    chips = [ad_chip(ad), f'<span class="chip">📦 <b>{"여러 제품 비교" if multi else "단일 제품"}</b></span>',
             f'<span class="chip">💬 댓글 <b>{len(src["comments"])}개</b></span>']
    if s["purpose"]:
        chips.append(f'<span class="chip">🎯 내 목적 <b>{e(s["purpose"])}</b></span>')
    if d.get("cached"):
        chips.append('<span class="chip green">⚡ <b>저장된 결과</b></span>')
    out.append('<div class="chips">' + "".join(chips) + "</div>")
    out.append(guide("single", bool(src.get("segments") and src.get("vid"))))

    out.append(section("summary", "한눈에 보기"))
    out.append(f'<p class="lead">{e(r.get("summary"))}</p>')
    out.append(trust_html(d["trust"], d["trust_reasons"]))
    out.append(f'<p class="note">📢 광고 의심 지수 {ad}점 — {e(r.get("ad_reason"))}</p>')
    if multi and r.get("products"):
        out.append(f'<div class="callout">📦 여러 제품을 비교한 리뷰예요. 장단점마다 어떤 제품인지 표시했어요.<br><b>등장 제품</b> · {", ".join(e(p) for p in r["products"][:6])}</div>')

    sub = "내 목적에 치명적인 순서" if s["purpose"] else "구매 결정에 영향이 큰 순서"
    if src.get("segments"):
        sub += " · ▶ 장면 보기를 누르면 유튜브에서 그 장면부터 재생"
    out.append(section("cons", "👎 치명적 단점", sub))
    out.append(points_html(r.get("cons", []), multi, src))
    out.append(section("pros", "👍 핵심 장점", "▶ 장면 보기를 누르면 그 장면부터 재생" if src.get("segments") else ""))
    out.append(points_html(r.get("pros", []), multi, src))
    out.append(section("check", "✅ 사기 전에 확인할 것", f"'{s['purpose']}' 기준" if s["purpose"] else ""))
    out.append(checklist_html(r.get("checklist", [])))

    out.append(section("comments", "💬 댓글 감지", "AI가 댓글마다 이름표를 붙이고, 개수는 코드가 셌어요"))
    if ca and src["comments"]:
        stats = [("🗣️ 광고라고 지적", ca.get("ad_suspicion_count", 0)), ("🤖 바이럴 의심", ca.get("viral_count", 0)),
                 ("🙋 실사용 경험", ca.get("experience_count", 0)), ("💬 기타", ca.get("other_count", 0))]
        out.append('<div class="stats">' + "".join(f"<div class='stat'><small>{a}</small><div>{b}</div></div>" for a, b in stats) + "</div>")
        if ca.get("mismatch"):
            out.append(f'<div class="callout">⚠️ <b>리뷰와 댓글이 다른 점</b><br>{e(ca["mismatch"])}</div>')
        ops = ca.get("real_user_opinions") or []
        if ops:
            lis = "".join(f'<li>{"👎" if o.get("type") == "단점" else "👍"} {e(o.get("opinion"))}</li>' for o in ops[:5])
            out.append(f'<p class="note" style="margin-top:1rem"><b>실사용자 경험 요약</b></p><ul class="plain">{lis}</ul>')
        out.append(list_details(f"🗣️ 광고라고 지적한 댓글 ({ca.get('ad_suspicion_count', 0)})", ca.get("ad_suspicion_examples", [])))
        out.append(list_details(f"🤖 바이럴 의심 댓글 ({ca.get('viral_count', 0)})", ca.get("viral_examples", [])))
        out.append(list_details(f"🙋 실사용 경험 댓글 ({ca.get('experience_count', 0)})", ca.get("experience_comments", [])))
    else:
        out.append('<p class="note">분석한 댓글이 없어요. 블로그나 붙여넣은 글은 왼쪽 <b>💬 댓글 직접 넣기</b>에 댓글을 넣으면 함께 분석해요.</p>')

    ads = r.get("ad_sentences", [])
    out.append(section("ads", "🚫 걸러낸 광고 문장", f"{len(ads)}개 · 리뷰어가 직접 한 말 중 장단점 판단에서 뺀 문장"))
    out.append("<ul class='plain'>" + "".join(f"<li><s style='color:#8E8D88'>{e(a)}</s></li>" for a in ads) + "</ul>" if ads else '<p class="note">걸러낸 광고 문장이 없어요.</p>')
    out.append(section("share", "📤 공유하기", "친구에게 보내기 좋은 카드 이미지"))
    show_html(out)

    # 공유: 이미지 카드 + 복사용 글
    png = make_card(src["meta"]["title"], f'{src["meta"].get("channel", "")} · RealPick AI 분석', d["trust"], ad,
                    r.get("cons", []), r.get("pros", []), r.get("checklist", []))
    lines = [f"[RealPick AI] {src['meta']['title']}", f"신뢰 점수 {d['trust']}/100 · 광고 의심 {ad}점", "", "치명적 단점"]
    lines += [f"{i}. {c.get('point')}" for i, c in enumerate(r.get("cons", [])[:3], 1)]
    lines += ["", "핵심 장점"] + [f"{i}. {p.get('point')}" for i, p in enumerate(r.get("pros", [])[:3], 1)]
    if src["meta"].get("url"):
        lines += ["", src["meta"]["url"]]
    c1, c2 = st.columns([1, 1])
    if png:
        c1.download_button("🖼️ 카드 이미지 저장", png, file_name="realpick.png", mime="image/png", use_container_width=True)
        with c2.popover("👀 카드 미리보기", use_container_width=True):
            st.image(png)
    else:
        c1.caption("이 컴퓨터에서 한글 글꼴을 찾지 못해 이미지 카드는 만들 수 없어요. 아래 글을 복사해 주세요.")
    st.caption("카톡에 붙여넣을 글 (오른쪽 위 버튼으로 복사)")
    st.code("\n".join(lines), language=None)

    qa_html = [section("qa", "🙋 이 리뷰에 물어보기", "리뷰와 댓글에 있는 내용으로만 답해요")]
    if d["qa"]:
        for h in d["qa"]:
            qa_html.append(f'<div class="qa-q">{e(h["q"])}</div><div class="qa-a">{e(h["a"])}</div>')
    else:
        qa_html.append('<p class="note">화면 <b>맨 아래 입력창</b>에 궁금한 걸 적고 엔터를 누르세요. 예) "손이 작은 사람한테도 괜찮대?", "배터리는 실제로 몇 시간이래?"</p><div class="down">⬇ 화면 맨 아래 입력창에 질문하기</div>')
    qa_html.append(section("source", "🔎 원문 확인", "AI가 실제로 읽은 자료"))
    qa_html.append(list_details(f"💬 가져온 댓글 전체 ({len(src['comments'])}개, 답글 포함)", src["comments"]))
    txt = src["text"]
    qa_html.append(f"<details><summary>📄 분석한 원문 (자막·본문) 앞부분</summary><p>{e(txt[:2000])}{'…' if len(txt) > 2000 else ''}</p></details>")
    qa_html.append(foot(d["usage"], d.get("cached")))
    show_html(qa_html)


# ======================= 화면: 여러 리뷰 종합 =======================
MULTI_TOC = [("summary", "한눈에 보기"), ("cons", "공통 단점 순위"), ("pros", "공통 장점"), ("split", "평가가 갈린 부분"),
             ("check", "사기 전에 확인할 것"), ("reviews", "분석한 리뷰")]


def reviews_html(reviews: list, fails: list) -> list:
    out = []
    for i, o in enumerate(reviews, 1):
        ad = int(o["result"].get("ad_score", 0) or 0)
        extra = f" · 신뢰 {o['trust']}점 · 광고 의심 {ad}점" + (" · ⚡저장됨" if o.get("cached") else "")
        out.append(src_card(o["src"], i, extra))
        cons = "".join(f"<li>👎 {e(c.get('point'))}</li>" for c in o["result"].get("cons", [])[:3])
        pros = "".join(f"<li>👍 {e(p.get('point'))}</li>" for p in o["result"].get("pros", [])[:3])
        out.append(f"<details><summary>#{i} 리뷰의 장단점 보기</summary><ul>{cons}{pros}</ul></details>")
    if fails:
        out.append(list_details(f"분석하지 못한 링크 {len(fails)}개", [f"{f['link']} — {f['error']}" for f in fails]))
    return out


def render_multi(d: dict):
    agg, reviews, s = d["agg"], d["reviews"], d["settings"]
    n = len(reviews)
    title = f"{d['name']} 리뷰 {n}개 종합" if d.get("name") else f"리뷰 {n}개 종합"
    avg_trust = round(sum(o["trust"] for o in reviews) / n)
    out = [f'<h1 class="doc-title serif">{e(title)}</h1>']
    chips = [f'<span class="chip">📚 분석한 리뷰 <b>{n}개</b></span>', f'<span class="chip">🛡️ 평균 신뢰 점수 <b>{avg_trust}</b></span>']
    if d["fails"]:
        chips.append(f'<span class="chip amber">⚠️ 분석 실패 <b>{len(d["fails"])}개</b></span>')
    if s["purpose"]:
        chips.append(f'<span class="chip">🎯 내 목적 <b>{e(s["purpose"])}</b></span>')
    out.append('<div class="chips">' + "".join(chips) + "</div>")
    out.append(guide("multi"))
    out.append(section("summary", "한눈에 보기"))
    out.append(f'<p class="lead">{e(agg.get("summary"))}</p>')
    out.append('<p class="note">몇 개 리뷰에서 나왔는지는 AI가 적은 리뷰 번호를 코드가 직접 세서 보여줘요. 어느 리뷰에서 나왔는지 모르는 항목은 버렸어요.</p>')
    out.append(section("cons", "👎 공통 단점 순위", "여러 리뷰가 함께 말할수록 위로"))
    out.append(counted_html(agg.get("common_cons", []), n, good=False))
    out.append(section("pros", "👍 공통 장점"))
    out.append(counted_html(agg.get("common_pros", []), n, good=True))
    out.append(section("split", "🤔 평가가 갈린 부분"))
    dis = agg.get("disagreements") or []
    out.append("<ul class='plain'>" + "".join(f"<li><b>{e(x.get('topic'))}</b> — {e(x.get('detail'))}</li>" for x in dis) + "</ul>" if dis else '<p class="note">리뷰끼리 크게 엇갈린 부분은 없어요.</p>')
    out.append(section("check", "✅ 사기 전에 확인할 것"))
    out.append(checklist_html(agg.get("checklist", [])))
    out.append(section("reviews", "📄 분석한 리뷰", "번호(#)는 위 순위의 출처 번호예요"))
    out += reviews_html(reviews, d["fails"])
    out.append(foot(d["usage"]))
    show_html(out)


# ======================= 화면: 제품 비교 =======================
COMPARE_TOC = [("summary", "한눈에 보기"), ("picks", "상황별 추천"), ("diff", "차이점"), ("vs", "단점·장점 나란히"),
               ("reviews", "분석한 리뷰")]


def side_col(side: dict) -> str:
    agg, n = side["agg"], len(side["reviews"])
    cons = "".join(f"<li>{e(c.get('point'))} <span class='cnt'>{c.get('count')}/{n}</span></li>" for c in agg.get("common_cons", [])[:4])
    pros = "".join(f"<li>{e(p.get('point'))} <span class='cnt good'>{p.get('count')}/{n}</span></li>" for p in agg.get("common_pros", [])[:4])
    return (f'<div class="col"><h4>{side["label"]} · {e(side["name"])}</h4><div class="note">리뷰 {n}개 · {e(agg.get("summary"))}</div>'
            f'<div class="k">👎 공통 단점</div><ol>{cons or "<li>없음</li>"}</ol><div class="k">👍 공통 장점</div><ol>{pros or "<li>없음</li>"}</ol></div>')


def render_compare(d: dict):
    a, b, cmp, s = d["sides"][0], d["sides"][1], d["cmp"], d["settings"]
    out = [f'<h1 class="doc-title serif">{e(a["name"])} vs {e(b["name"])}</h1>']
    chips = [f'<span class="chip">A 리뷰 <b>{len(a["reviews"])}개</b></span>', f'<span class="chip">B 리뷰 <b>{len(b["reviews"])}개</b></span>']
    if s["purpose"]:
        chips.append(f'<span class="chip">🎯 내 목적 <b>{e(s["purpose"])}</b></span>')
    out.append('<div class="chips">' + "".join(chips) + "</div>")
    out.append(guide("compare"))
    out.append(section("summary", "한눈에 보기"))
    out.append(f'<p class="lead">{e(cmp.get("summary"))}</p>')
    if cmp.get("caution"):
        out.append(f'<div class="callout">⚠️ {e(cmp["caution"])}</div>')
    out.append(section("picks", "🎯 상황별 추천", "한쪽이 무조건 낫다고 하지 않아요"))
    names = {"A": a["name"], "B": b["name"]}
    picks = "".join(
        f'<div class="pick"><span class="who">{e(p.get("situation"))}</span><span class="p">{e(names.get(p.get("pick"), p.get("pick")))}</span>{e(p.get("reason"))}</div>'
        for p in (cmp.get("picks") or [])[:3]
    )
    out.append(f'<div class="picks">{picks}</div>' if picks else '<p class="note">추천을 만들지 못했어요.</p>')
    out.append(section("diff", "↔️ 차이점"))
    rows = "".join(f"<tr><td><b>{e(x.get('topic'))}</b></td><td>{e(x.get('a'))}</td><td>{e(x.get('b'))}</td></tr>" for x in (cmp.get("differences") or []))
    out.append(f'<table class="diff"><tr><th>항목</th><th>{e(a["name"])}</th><th>{e(b["name"])}</th></tr>{rows}</table>' if rows else '<p class="note">뚜렷한 차이점을 찾지 못했어요.</p>')
    out.append(section("vs", "📊 단점·장점 나란히", "숫자는 '몇 개 리뷰 중 몇 개가 말했나'"))
    out.append(f'<div class="vs">{side_col(a)}{side_col(b)}</div>')
    out.append(section("reviews", "📄 분석한 리뷰"))
    for side in (a, b):
        out.append(f'<p class="note"><b>{side["label"]} · {e(side["name"])}</b></p>')
        out += reviews_html(side["reviews"], side["fails"])
    out.append(foot(d["usage"]))
    show_html(out)


# ======================= 사이드바 =======================
data = st.session_state.get("data")
with st.sidebar:
    brand = f'<img src="{LOGO}" alt="RealPick AI">' if LOGO else "<b>🔍 RealPick AI</b>"
    st.markdown(f'<div class="brand">{brand}</div>', unsafe_allow_html=True)
    if data:
        toc = {"single": SINGLE_TOC, "multi": MULTI_TOC, "compare": COMPARE_TOC}[data["mode"]]
        links = "".join(f'<a href="#{sid}"><span>{i:02d}</span>{name}</a>' for i, (sid, name) in enumerate(toc, 1))
        st.markdown(f'<nav class="toc">{links}</nav>', unsafe_allow_html=True)
        st.write("")
        if st.button("＋ 새로 분석하기", use_container_width=True):
            st.session_state.pop("data", None)
            st.rerun()
    else:
        st.markdown('<div class="side-note">가운데에서 <b>모드</b>를 고르고 링크나 제품 이름을 넣으면 분석이 시작돼요.</div>', unsafe_allow_html=True)
    st.write("")
    with st.expander("❓ 사용법"):
        st.markdown(
            '<div class="side-note">'
            "<b>1.</b> 가운데에서 모드를 고르고 링크·글·제품 이름을 넣어요.<br>"
            "<b>2.</b> 🎯 사용 목적을 고르면 나에게 맞춘 단점 순서가 나와요.<br>"
            "<b>3.</b> 결과의 <b>▶ 장면 보기</b>는 유튜브 그 장면으로 이동해요.<br>"
            "<b>4.</b> 맨 아래 입력창에 리뷰에 대해 질문할 수 있어요.<br>"
            "<b>5.</b> 📤 공유하기에서 카드 이미지를 저장해요.</div>",
            unsafe_allow_html=True,
        )
    with st.expander("⚙️ 분석 설정"):
        model_label = st.selectbox("AI 모델", list(MODELS.keys()))
        prompt_version = st.radio("프롬프트 (실험용)", list(PROMPT_LABELS.keys()), index=3, format_func=lambda k: PROMPT_LABELS[k])
        use_comments = st.checkbox("유튜브 댓글도 분석하기", value=True)
    with st.expander("💬 댓글 직접 넣기"):
        pasted_comments = st.text_area("블로그나 붙여넣은 글의 댓글을 한 줄에 하나씩 (리뷰 하나 분석용)", height=110)
    with st.expander("🧩 유튜브에서 바로 분석하기"):
        st.markdown('<div class="side-note"><b>북마크 버튼</b>: 아래 글을 복사해 브라우저 북마크의 <b>URL</b> 칸에 붙여넣으세요. 유튜브를 보다가 그 북마크를 누르면 RealPick이 그 영상을 바로 분석해요.</div>', unsafe_allow_html=True)
        st.code(f"javascript:window.open('{APP_URL}/?url='+encodeURIComponent(location.href))", language=None)
        st.markdown('<div class="side-note"><b>크롬 확장 프로그램</b>: 저장소의 <code>extension</code> 폴더를 크롬에 불러오면 툴바 버튼 하나로 분석해요. 방법은 README에 있어요.</div>', unsafe_allow_html=True)
    st.markdown('<div class="side-note" style="margin-top:1rem">🛡️ 이메일·전화번호는 AI에 보내기 전에 가려져요. 결과는 이 컴퓨터의 <code>.cache</code> 폴더에만 저장되고, 구매 링크·광고는 넣지 않아요.</div>', unsafe_allow_html=True)
    if not OPENAI_KEY:
        st.error("OPENAI_API_KEY 가 설정되지 않았어요. README를 확인하세요.")


def settings(purpose: str) -> dict:
    return {"version": prompt_version, "model": MODELS[model_label], "comments": use_comments, "purpose": purpose}


def pick_purpose(choice: str, custom: str) -> str:
    if custom.strip():
        return custom.strip()[:60]
    return "" if choice == "선택 안 함" else choice


extra_comments = [c.strip() for c in pasted_comments.splitlines() if c.strip()]

# 북마크·확장 프로그램에서 ?url=... 로 들어오면 바로 분석
incoming = st.query_params.get("url")
if incoming and not st.session_state.get("handled_url") == incoming:
    st.session_state["handled_url"] = incoming
    st.query_params.clear()
    handle(run_single, incoming, settings(""), [])


# ======================= 본문 =======================
if data:
    if data["mode"] == "single":
        render_single(data)
        question = st.chat_input("이 리뷰에 물어보세요 (예: 손 작은 사람한테도 괜찮대?)")
        if question:
            answer = None
            try:
                with st.spinner("리뷰에서 답을 찾는 중..."):
                    answer, u = ask(question, data["src"]["text"], data["src"]["comments"], data["qa"],
                                    data["settings"]["model"], OPENAI_KEY)
            except Exception as err:
                st.error(f"😥 {friendly_error(err)}")
            if answer is not None:
                data["qa"].append({"q": question, "a": answer})
                st.rerun()
    elif data["mode"] == "multi":
        render_multi(data)
    else:
        render_compare(data)
else:
    logo_html = f'<img class="logo" src="{LOGO}" alt="RealPick AI">' if LOGO else '<h1 class="serif">RealPick AI</h1>'
    show_html([f'<div class="hero">{logo_html}<div class="tag">광고는 걸러내고, <b>진짜 장단점만</b> 골라 드려요.</div></div>'])
    mode = st.radio("모드", MODES, horizontal=True, label_visibility="collapsed")
    with st.form("main_input", border=False):
        if mode == MODES[0]:
            q = st.text_area("리뷰", placeholder="유튜브·블로그 리뷰 링크를 붙여넣거나, 리뷰 글을 그대로 붙여넣으세요",
                             height=96, label_visibility="collapsed")
        elif mode == MODES[1]:
            q = st.text_area("리뷰들", placeholder="제품 이름(예: 에어팟 프로 2) 또는 리뷰 링크 여러 개 (한 줄에 하나, 최대 5개)",
                             height=96, label_visibility="collapsed")
        else:
            ca_, cb_ = st.columns(2)
            qa_in = ca_.text_input("제품 A", placeholder="제품 A 이름 또는 링크")
            qb_in = cb_.text_input("제품 B", placeholder="제품 B 이름 또는 링크")
        p1, p2 = st.columns([1, 1])
        p_choice = p1.selectbox("🎯 내 사용 목적 (선택)", PURPOSES)
        p_custom = p2.text_input("직접 적기 (선택)", placeholder="예: 손이 작은 편, 회의 통화 많음")
        h1, h2 = st.columns([4, 1])
        hints = ["예) https://youtu.be/... · Ctrl+Enter로 바로 분석",
                 "제품 이름만 넣으면 유튜브에서 리뷰를 찾아요 (YouTube 키 필요)",
                 "제품 이름(유튜브 검색) 또는 리뷰 링크 · 여러 링크는 띄어쓰기로"]
        h1.markdown(f'<div class="hint">{hints[MODES.index(mode)]}</div>', unsafe_allow_html=True)
        submitted = h2.form_submit_button("분석하기 →", type="primary", use_container_width=True)
    if submitted:
        s = settings(pick_purpose(p_choice, p_custom))
        if mode == MODES[0]:
            if q.strip():
                handle(run_single, q.strip(), s, extra_comments)
        elif mode == MODES[1]:
            if q.strip():
                handle(run_multi, q.strip(), s)
        else:
            handle(run_compare, qa_in, qb_in, s)
    show_html([
        '<div class="steps">'
        '<div class="step"><b>🔎 리뷰 하나</b><small>광고 걸러내기, 신뢰 점수, ▶ 근거 장면 바로가기, 리뷰에 질문하기</small></div>'
        '<div class="step"><b>📚 여러 리뷰 종합</b><small>"5개 중 4개가 말한 단점" 순위로 정리</small></div>'
        '<div class="step"><b>⚖️ 제품 비교</b><small>두 제품을 나란히, 내 상황에 맞는 쪽 추천</small></div>'
        "</div>",
        LANDING_GUIDE,
    ])
