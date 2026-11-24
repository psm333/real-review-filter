"""OpenAI(GPT) API로 리뷰를 분석하는 부분.

- analyze   : 리뷰 하나 분석 (프롬프트 A/B/C/D 비교 실험 가능)
- aggregate : 여러 리뷰 결과를 종합해 공통 단점 순위
- compare   : 두 제품의 종합 결과 비교
- ask       : 분석한 리뷰에 대해 질문하기
- trust_score / locate_time : 코드로 계산하는 신뢰 점수, 근거 문장의 영상 시간
"""
import json
import re

import os

from openai import OpenAI

# .env 의 OPENAI_MODEL 로 기본 모델을 바꿀 수 있다.
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
MODELS = {
    f"{DEFAULT_MODEL} (기본)": DEFAULT_MODEL,
    "gpt-4o-mini (비교용 · 저렴)": "gpt-4o-mini",
}

MAX_CHARS = 15000  # 비용 관리: 본문은 앞부분 15,000자까지만
MAX_COMMENTS = 50
MAX_COMMENT_CHARS = 300

JSON_FORMAT = """
반드시 아래 JSON 형식으로만 답하세요. 다른 말은 쓰지 마세요.
{
  "summary": "한 줄 총평",
  "review_type": "단일 제품" 또는 "여러 제품 비교",
  "products": ["리뷰에 나온 제품 이름", ...],
  "ad_score": 0~100 사이 정수 (광고/협찬 성격이 강할수록 높음),
  "ad_reason": "광고 점수의 이유 한두 문장",
  "ad_sentences": ["리뷰 본문에서 걸러낸 광고성 문장 (댓글 제외)", ...],
  "cons": [{"product": "해당 제품 이름", "point": "단점", "evidence": "원문 근거 문장"}, ... 최대 3개],
  "pros": [{"product": "해당 제품 이름", "point": "장점", "evidence": "원문 근거 문장"}, ... 최대 3개],
  "checklist": ["단점을 바탕으로 사기 전에 확인할 것 (구체적인 행동으로)", ... 3개],
  "comment_analysis": {
    "comment_labels": [{"i": 댓글 번호, "label": "지적" 또는 "바이럴" 또는 "경험" 또는 "기타"}, ... 모든 댓글 빠짐없이],
    "real_user_opinions": [{"type": "장점" 또는 "단점", "opinion": "실사용자 경험담 요약"}, ... 최대 5개],
    "mismatch": "리뷰 본문과 댓글 반응이 다른 점 (없으면 빈 문자열)"
  }
}
comment_labels 기준:
  "지적" = 리뷰가 광고·협찬이라고 의심하거나 지적 (예: "광고네", "협찬 표시 안 함")
  "바이럴" = 댓글 자체가 홍보·구매 유도 (할인 링크, 공구, 근거 없는 과한 칭찬)
  "경험" = 제품을 직접 써 본 경험담 (예: "저도 샀는데 배터리 4시간 가요")
  "기타" = 질문, 감탄, 잡담, 리뷰어 칭찬 등 위에 해당하지 않는 것
댓글이 없으면 comment_analysis는 null로 두세요.
"""

PROMPTS = {
    # A: 가장 단순한 프롬프트 (기준점)
    "A": "다음 리뷰의 장점과 단점을 각각 3개씩 알려주세요.\n" + JSON_FORMAT,
    # B: 광고 필터링 지시 추가
    "B": (
        "다음 리뷰에서 광고, 협찬, 홍보성 멘트와 불필요한 서론은 무시하고 "
        "제품의 장점과 단점을 각각 3개씩 알려주세요.\n" + JSON_FORMAT
    ),
    # C: 역할 + 판단 기준 + 근거 요구 + 댓글 교차 검증 (최종 채택안)
    "C": """당신은 광고에 속지 않는 깐깐한 소비자 리뷰 분석가입니다.

[해야 할 일]
1. 광고 걸러내기: 다음에 해당하는 문장은 장단점 판단에서 제외하고 ad_sentences에 모으세요.
   - 협찬/제공/유료광고 표시, 할인코드·링크·구매 유도
   - 근거 없는 과장 ("인생템", "무조건 사세요", "단점이 없어요")
   - 브랜드 홍보 문구를 그대로 읽은 듯한 문장
2. 진짜 단점 3개: 실제 사용 경험에서 나온 불편함만 고르세요. 리뷰어가 "사소하지만", "굳이 꼽자면"처럼
   돌려 말한 단점도 놓치지 마세요. 구매 결정에 영향이 큰 순서로 정렬하세요.
3. 핵심 장점 3개: 구체적 경험이나 수치가 있는 장점만 고르세요.
4. 모든 장단점에는 원문에서 그대로 가져온 근거 문장을 evidence에 적으세요.
   원문에 근거가 없으면 지어내지 말고 항목 수를 줄이세요.
5. 댓글 분석: 광고를 의심/지적하는 댓글을 세고, 실제 사용자 경험담을 뽑고,
   리뷰 본문과 댓글 반응이 어긋나는 점을 mismatch에 적으세요.
   댓글은 세 종류로 나누세요: (가) 리뷰가 광고라고 지적하는 댓글 → ad_suspicion,
   (나) 댓글 자체가 홍보·구매 유도인 댓글 → viral, (다) 실제 사용 경험담 → real_user_opinions.
   댓글 문장은 ad_sentences에 넣지 마세요 (ad_sentences는 리뷰 본문 전용).
6. 여러 제품을 비교하거나 추천하는 리뷰라면 review_type을 "여러 제품 비교"로 하고,
   모든 장단점의 product에 어떤 제품 이야기인지 반드시 적으세요.
7. 의학적·법률적 판단은 하지 마세요.
""" + JSON_FORMAT,
}

# D: 1차 실험에서 C가 리뷰어의 의견·단점까지 광고로 걸러낸 문제를 고친 버전
PROMPTS["D"] = """당신은 광고에 속지 않으면서도 공정한 소비자 리뷰 분석가입니다.

[광고 문장의 정의 — 아래에 해당할 때만 ad_sentences에 넣으세요]
- 협찬·제품 제공·유료광고·파트너스 수수료 고지
- 구매 링크, 할인 코드, 할인 행사, 한정 수량 등 구매를 재촉하는 말
- 구독·댓글 이벤트, 경품 추첨 홍보
- 근거 없는 과장 ("인생템", "무조건 사세요", "단점이 하나도 없어요")

[광고가 아닌 것 — 절대 ad_sentences에 넣지 마세요]
- 리뷰어의 개인 의견, 비교, 아쉬움, 비판 (부정적인 문장은 광고가 아닙니다)
- 제품 사양·기능 설명, 가격 정보의 단순 언급
- 인사, 영상 소개, 전작 이야기

[해야 할 일]
1. ad_sentences에는 리뷰 본문에 실제로 있는 문장을 그대로 복사하세요. 요약하거나 지어내지 마세요.
2. 진짜 단점 3개: 리뷰어가 아쉽다고 말한 것, "다만", "굳이 꼽자면"처럼 돌려 말한 것을 꼭 찾으세요.
   리뷰어가 조금이라도 아쉬움을 말했다면 단점을 비워 두지 마세요. 구매 결정에 영향이 큰 순서로.
   단점은 리뷰 본문에서 찾고, 댓글에서만 나온 불만은 real_user_opinions에 넣으세요.
3. 핵심 장점 3개: 구체적 경험이나 수치가 있는 장점을 고르세요.
4. 모든 장단점의 evidence에는 리뷰 본문 문장을 그대로 복사하세요. 근거가 없는 항목은 만들지 마세요.
5. 댓글 분류: 번호가 붙은 모든 댓글에 comment_labels 이름표를 하나씩 붙이세요 (빠짐없이).
   "경험" 댓글은 real_user_opinions에 요약하고, 리뷰와 댓글 반응이 어긋나는 점은 mismatch에.
6. 여러 제품을 비교하는 리뷰면 review_type을 "여러 제품 비교"로 하고 각 장단점의 product를 적으세요.
7. 의학적·법률적 판단은 하지 마세요.
""" + JSON_FORMAT

PROMPT_LABELS = {
    "A": "A · 단순 요청",
    "B": "B · 광고 제외 지시",
    "C": "C · 기준+근거+댓글검증",
    "D": "D · C 개선: 광고 정의 좁힘 (최종)",
}


def mask_personal_info(text: str) -> str:
    """AI에 보내기 전에 이메일, 전화번호를 가린다 (책임 있는 AI)."""
    text = re.sub(r"[\w.+-]+@[\w-]+\.[\w.]+", "[이메일]", text)
    text = re.sub(r"01[016789][-\s.]?\d{3,4}[-\s.]?\d{4}", "[전화번호]", text)
    return text


def _parse_json(raw: str) -> dict:
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("AI 응답에서 JSON을 찾지 못했어요.")
    return json.loads(raw[start : end + 1])


def analyze(
    text: str,
    comments: list[str] | None = None,
    prompt_version: str = "D",
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
    purpose: str = "",
) -> tuple[dict, dict]:
    """리뷰를 분석해서 (결과 dict, 사용량 dict)를 돌려준다. purpose = 사용자의 사용 목적 (선택)."""
    client = OpenAI(api_key=api_key)
    system = PROMPTS[prompt_version] + (PURPOSE_BLOCK.format(purpose=purpose) if purpose else "")

    body = mask_personal_info(text[:MAX_CHARS])
    comments = [mask_personal_info(c[:MAX_COMMENT_CHARS]) for c in (comments or [])[:MAX_COMMENTS]]

    user_msg = f"<review>\n{body}\n</review>\n"
    if comments:
        user_msg += "<comments>\n" + "\n".join(f"[{i}] {c}" for i, c in enumerate(comments, 1)) + "\n</comments>\n"
    else:
        user_msg += "(댓글 없음)\n"
    user_msg += "JSON만 출력하세요."

    resp = client.chat.completions.create(
        model=model,
        max_completion_tokens=4000,  # 댓글 이름표까지 담을 수 있게
        response_format={"type": "json_object"},  # JSON으로만 답하게 강제
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
        ],
    )
    raw = resp.choices[0].message.content or ""
    usage = {"input_tokens": resp.usage.prompt_tokens, "output_tokens": resp.usage.completion_tokens}
    result = _parse_json(raw)
    _count_comments(result, comments)
    if purpose:
        _sort_by_fit(result)
    return result, usage


PURPOSE_BLOCK = """
[사용자 맞춤]
사용자의 사용 목적: "{purpose}"
- cons의 각 항목에 "fit": "높음" 또는 "보통" 또는 "낮음" (이 사용자에게 얼마나 치명적인지)과
  "fit_reason": "왜 그런지 한 문장"을 추가하세요.
- checklist는 이 사용 목적에 맞춰 작성하세요.
"""

FIT_ORDER = {"높음": 0, "보통": 1, "낮음": 2}


def _sort_by_fit(result: dict) -> None:
    """사용 목적에 치명적인 단점이 위로 오게 코드가 정렬한다."""
    cons = result.get("cons") or []
    result["cons"] = sorted(cons, key=lambda c: FIT_ORDER.get(c.get("fit"), 1))


def _chat_json(system: str, user: str, model: str, api_key: str | None, max_tokens: int = 3000):
    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        max_completion_tokens=max_tokens,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
    )
    usage = {"input_tokens": resp.usage.prompt_tokens, "output_tokens": resp.usage.completion_tokens}
    return _parse_json(resp.choices[0].message.content or ""), usage


# ======================= 여러 리뷰 종합 =======================
AGGREGATE_PROMPT = """당신은 같은 제품의 여러 리뷰 분석 결과를 종합하는 공정한 소비자 분석가입니다.
입력은 리뷰마다 번호, 제목, 광고 의심 지수, 단점·장점(근거 포함), 댓글 속 실사용 경험입니다.

[해야 할 일]
1. 뜻이 같은 단점을 하나로 묶고, 그 단점이 나온 리뷰 번호를 sources에 빠짐없이 적으세요.
2. 장점도 같은 방식으로 묶으세요. 광고 의심 지수가 70 이상인 리뷰에서만 나온 장점은 caution에 그 사실을 적으세요.
3. 리뷰끼리 평가가 갈리는 부분은 disagreements에 적으세요.
4. 입력에 없는 내용은 절대 만들지 마세요.{purpose_part}

반드시 아래 JSON 형식으로만 답하세요.
{{
  "summary": "종합 한 줄 평",
  "common_cons": [{{"point": "단점", "sources": [리뷰 번호, ...], "detail": "리뷰들이 말한 내용 한 문장"{fit_fields}}}, ... 최대 5개],
  "common_pros": [{{"point": "장점", "sources": [리뷰 번호, ...], "detail": "한 문장", "caution": "주의할 점 (없으면 빈 문자열)"}}, ... 최대 5개],
  "disagreements": [{{"topic": "주제", "detail": "어떤 리뷰는 ~, 다른 리뷰는 ~"}}, ...],
  "checklist": ["사기 전에 확인할 것", ... 3개]
}}
"""


def _review_digest(i: int, rv: dict) -> str:
    r = rv["result"]
    lines = [f"[리뷰 {i}] {rv['meta'].get('title', '')} (광고 의심 지수 {r.get('ad_score', 0)})"]
    for c in r.get("cons", []):
        lines.append(f"  단점: {c.get('point')} / 근거: {c.get('evidence', '')}")
    for p in r.get("pros", []):
        lines.append(f"  장점: {p.get('point')} / 근거: {p.get('evidence', '')}")
    ca = r.get("comment_analysis") or {}
    for o in (ca.get("real_user_opinions") or [])[:5]:
        lines.append(f"  댓글 실사용 {o.get('type')}: {o.get('opinion')}")
    return "\n".join(lines)


def aggregate(reviews: list[dict], purpose: str = "", model: str = DEFAULT_MODEL, api_key: str | None = None):
    """여러 리뷰 분석 결과를 종합한다. 몇 개 리뷰에서 나왔는지는 코드가 sources로 직접 센다."""
    purpose_part = f'\n5. 사용자의 사용 목적은 "{purpose}"입니다. 각 단점에 fit과 fit_reason을 적으세요.' if purpose else ""
    fit_fields = ', "fit": "높음|보통|낮음", "fit_reason": "한 문장"' if purpose else ""
    system = AGGREGATE_PROMPT.format(purpose_part=purpose_part, fit_fields=fit_fields)
    user = "\n\n".join(_review_digest(i, rv) for i, rv in enumerate(reviews, 1))
    result, usage = _chat_json(system, user, model, api_key)
    n = len(reviews)
    for key in ("common_cons", "common_pros"):
        items = []
        for it in result.get(key) or []:
            srcs = sorted({int(x) for x in (it.get("sources") or []) if str(x).isdigit() and 1 <= int(x) <= n})
            if not srcs:
                continue  # 어느 리뷰에서 나왔는지 모르는 항목은 버린다 (지어낸 내용 방지)
            it["sources"] = srcs
            it["count"] = len(srcs)
            items.append(it)
        items.sort(key=lambda it: (-it["count"], FIT_ORDER.get(it.get("fit"), 1)))
        result[key] = items
    result["total"] = n
    return result, usage


# ======================= 두 제품 비교 =======================
COMPARE_PROMPT = """당신은 두 제품의 리뷰 종합 결과를 비교하는 공정한 소비자 분석가입니다.
입력에 있는 내용만 쓰고, 한쪽이 무조건 낫다고 하지 마세요. 상황에 따라 어느 쪽이 맞는지 알려 주세요.{purpose_part}

반드시 아래 JSON 형식으로만 답하세요.
{{
  "summary": "비교 한 줄 평",
  "differences": [{{"topic": "비교 항목", "a": "제품 A는 ~", "b": "제품 B는 ~"}}, ... 최대 5개],
  "picks": [{{"situation": "이런 사람이라면", "pick": "A" 또는 "B" 또는 "비슷함", "reason": "이유 한 문장"}}, ... 3개],
  "criteria": [{{"name": "비교 기준 (예: 배터리, 휴대성, 통화 품질)", "weight": 1~5 정수 (이 사용자에게 중요한 정도), "a": 0~10 정수 (제품 A 점수), "b": 0~10 정수 (제품 B 점수), "reason": "점수 근거 한 문장"}}, ... 3~6개],
  "caution": "비교할 때 주의할 점 (예: 리뷰 수가 적음)"
}}
criteria 규칙:
- 두 제품의 리뷰에 실제로 나온 내용으로만 기준과 점수를 정하세요. 근거가 없는 기준은 넣지 마세요.
- 한쪽 리뷰에만 정보가 있는 기준은 넣지 마세요 (공정한 비교가 안 됨).
- weight는 사용 목적이 있으면 그 목적에 중요한 기준일수록 높게, 없으면 일반 소비자 기준으로 정하세요.
- 최종 비율은 코드가 계산하니 비율을 직접 쓰지 마세요.
"""


def _agg_digest(name: str, agg: dict) -> str:
    n = agg.get("total", 0)
    lines = [f"{name} (리뷰 {n}개 종합): {agg.get('summary', '')}"]
    for c in agg.get("common_cons", []):
        lines.append(f"  단점 ({c['count']}/{n}개 리뷰): {c.get('point')} - {c.get('detail', '')}")
    for p in agg.get("common_pros", []):
        lines.append(f"  장점 ({p['count']}/{n}개 리뷰): {p.get('point')} - {p.get('detail', '')}")
    return "\n".join(lines)


def compare(name_a: str, agg_a: dict, name_b: str, agg_b: dict, purpose: str = "",
            model: str = DEFAULT_MODEL, api_key: str | None = None):
    purpose_part = f'\n사용자의 사용 목적은 "{purpose}"입니다. picks의 첫 항목은 이 목적에 대한 추천으로 하세요.' if purpose else ""
    system = COMPARE_PROMPT.format(purpose_part=purpose_part)
    user = f"[제품 A] {_agg_digest(name_a, agg_a)}\n\n[제품 B] {_agg_digest(name_b, agg_b)}"
    result, usage = _chat_json(system, user, model, api_key, max_tokens=2500)
    result["split"] = fit_split(result.get("criteria") or [])
    return result, usage


def _clamp(v, lo: int, hi: int, default: int) -> int:
    try:
        return max(lo, min(hi, int(round(float(v)))))
    except (TypeError, ValueError):
        return default


def fit_split(criteria: list[dict]) -> dict | None:
    """기준별 점수 × 중요도를 더해 '나에게 맞는 정도'를 A:B = 100으로 나눈다 (코드가 계산).
    차이가 10%p 미만이면 '비슷함'으로 본다."""
    sum_a = sum_b = 0.0
    used = []
    for c in criteria:
        if not c.get("name"):
            continue
        w = _clamp(c.get("weight"), 1, 5, 3)
        a = _clamp(c.get("a"), 0, 10, 5)
        b = _clamp(c.get("b"), 0, 10, 5)
        c.update(weight=w, a=a, b=b)
        sum_a += w * a
        sum_b += w * b
        used.append(c)
    if not used or sum_a + sum_b == 0:
        return None
    pa = round(sum_a / (sum_a + sum_b) * 100)
    pb = 100 - pa
    winner = "비슷함" if abs(pa - pb) < 10 else ("A" if pa > pb else "B")
    return {"a": pa, "b": pb, "winner": winner, "sum_a": round(sum_a), "sum_b": round(sum_b), "n": len(used)}


# ======================= 리뷰에 질문하기 =======================
ASK_PROMPT = """당신은 사용자가 방금 분석한 리뷰에 대한 질문에 답하는 도우미입니다.
- 아래 리뷰 원문과 댓글에 있는 내용으로만 답하세요.
- 답에는 근거가 된 원문 문장을 짧게 인용하세요.
- 리뷰와 댓글에 없는 내용이면 "이 리뷰에는 나오지 않아요"라고 말하고, 추측하지 마세요.
- 3~4문장 이내로, 쉬운 말로 답하세요. 의학적·법률적 판단은 하지 마세요.
"""


def ask(question: str, text: str, comments: list[str], history: list[dict],
        model: str = DEFAULT_MODEL, api_key: str | None = None) -> tuple[str, dict]:
    client = OpenAI(api_key=api_key)
    context = f"<review>\n{mask_personal_info(text[:MAX_CHARS])}\n</review>\n"
    if comments:
        context += "<comments>\n" + "\n".join(f"- {mask_personal_info(c[:MAX_COMMENT_CHARS])}" for c in comments[:MAX_COMMENTS]) + "\n</comments>"
    messages = [{"role": "system", "content": ASK_PROMPT + "\n" + context}]
    for h in history[-6:]:  # 최근 대화 3번까지만 (비용 관리)
        messages.append({"role": "user", "content": h["q"]})
        messages.append({"role": "assistant", "content": h["a"]})
    messages.append({"role": "user", "content": mask_personal_info(question[:500])})
    resp = client.chat.completions.create(model=model, max_completion_tokens=600, messages=messages)
    usage = {"input_tokens": resp.usage.prompt_tokens, "output_tokens": resp.usage.completion_tokens}
    return (resp.choices[0].message.content or "").strip(), usage


# ======================= 코드로 계산하는 것들 =======================
def norm(t: str) -> str:
    """띄어쓰기·문장부호를 없애서 비교하기 쉽게 만든다."""
    return re.sub(r"[\s\W_]+", "", t or "")


def in_source(snippet: str, source_norm: str) -> bool:
    """AI가 적은 문장이 원문에 실제로 있는지. '...'로 잘린 부분은 조각마다 확인한다."""
    parts = [norm(p) for p in re.split(r"\.{2,}|…", snippet or "")]
    parts = [p for p in parts if len(p) >= 6]
    return bool(parts) and all(p in source_norm for p in parts)


def locate_time(evidence: str, segments: list[dict]) -> float | None:
    """근거 문장이 자막의 몇 초에 나오는지 찾는다. 못 찾으면 None."""
    if not segments or not evidence:
        return None
    full, owner = [], []
    for idx, seg in enumerate(segments):
        piece = norm(seg["text"])
        full.append(piece)
        owner.extend([idx] * len(piece))
    full = "".join(full)
    first = next((norm(p) for p in re.split(r"\.{2,}|…", evidence) if len(norm(p)) >= 6), "")
    for size in (30, 20, 12, 8):
        key = first[:size]
        if len(key) < 6:
            continue
        pos = full.find(key)
        if pos >= 0:
            return segments[owner[pos]]["start"]
    return None


def trust_score(result: dict, text: str, comments: list[str]) -> tuple[int, list[tuple[str, int]]]:
    """리뷰를 얼마나 믿을 만한지 공식으로 계산한다 (AI가 정한 숫자가 아님).
    100점에서 시작해 이유마다 깎고, 깎은 이유를 함께 돌려준다."""
    reasons = []
    ad = int(result.get("ad_score", 0) or 0)
    if ad:
        reasons.append((f"광고 의심 지수 {ad}점", -round(ad * 0.4)))
    ca = result.get("comment_analysis") if isinstance(result.get("comment_analysis"), dict) else None
    if ca and ca.get("total"):
        ratio = ca.get("ad_suspicion_count", 0) / ca["total"]
        if ratio:
            reasons.append((f"광고라고 지적한 댓글 {ca.get('ad_suspicion_count', 0)}/{ca['total']}개", -min(20, round(ratio * 100))))
        if ca.get("mismatch"):
            reasons.append(("리뷰와 댓글 반응이 다름", -10))
    n_cons = len(result.get("cons") or [])
    if n_cons < 2:
        reasons.append((f"리뷰어가 말한 단점이 {n_cons}개뿐", -15))
    src = norm(text + " ".join(comments or []))
    ev = [it.get("evidence", "") for it in (result.get("cons") or []) + (result.get("pros") or []) if it.get("evidence")]
    if ev:
        miss = sum(not in_source(e, src) for e in ev)
        if miss:
            reasons.append((f"원문에서 확인 안 된 근거 {miss}/{len(ev)}개", -round(miss / len(ev) * 20)))
    score = max(0, min(100, 100 + sum(d for _, d in reasons)))
    return score, reasons


LABELS = {"지적": "ad_suspicion", "바이럴": "viral", "경험": "experience", "기타": "other"}


def _count_comments(result: dict, comments: list[str]) -> None:
    """AI가 댓글마다 붙인 이름표를 코드가 직접 센다 (AI가 말한 숫자를 믿지 않음)."""
    ca = result.get("comment_analysis")
    if not comments or not isinstance(ca, dict):
        return
    labeled = {}
    for item in ca.get("comment_labels", []) or []:
        try:
            i = int(item.get("i"))
        except (TypeError, ValueError):
            continue
        if 1 <= i <= len(comments) and item.get("label") in LABELS:
            labeled[i] = item["label"]
    groups = {key: [] for key in LABELS.values()}
    for i, c in enumerate(comments, 1):
        groups[LABELS[labeled.get(i, "기타")]].append(c)
    ca["total"] = len(comments)
    ca["unlabeled"] = len(comments) - len(labeled)
    ca["ad_suspicion_count"] = len(groups["ad_suspicion"])
    ca["ad_suspicion_examples"] = groups["ad_suspicion"]
    ca["viral_count"] = len(groups["viral"])
    ca["viral_examples"] = groups["viral"]
    ca["experience_count"] = len(groups["experience"])
    ca["experience_comments"] = groups["experience"]
    ca["other_count"] = len(groups["other"])
