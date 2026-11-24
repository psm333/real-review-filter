"""OpenAI(GPT) API로 리뷰를 분석하는 부분. 프롬프트 버전 A/B/C를 비교 실험할 수 있다."""
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
  "comment_analysis": {
    "ad_suspicion_count": 리뷰가 광고라고 의심하거나 지적하는 댓글 수 (예: "광고네", "협찬 표시 안 함"),
    "ad_suspicion_examples": ["예시 댓글", ...],
    "viral_count": 댓글 자체가 홍보·구매 유도로 보이는 댓글 수 (할인 링크, 공구, 과한 칭찬),
    "viral_examples": ["예시 댓글", ...],
    "real_user_opinions": [{"type": "장점" 또는 "단점", "opinion": "실사용자 의견 요약"}, ...],
    "mismatch": "리뷰 본문과 댓글 반응이 다른 점 (없으면 빈 문자열)"
  }
}
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
5. 댓글 분류: (가) 리뷰가 광고라고 지적 → ad_suspicion, (나) 댓글 자체가 홍보·구매 유도 → viral,
   (다) 실제 사용 경험담 → real_user_opinions. 리뷰와 댓글 반응이 어긋나는 점은 mismatch에.
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
) -> tuple[dict, dict]:
    """리뷰를 분석해서 (결과 dict, 사용량 dict)를 돌려준다."""
    client = OpenAI(api_key=api_key)

    body = mask_personal_info(text[:MAX_CHARS])
    comments = [mask_personal_info(c[:MAX_COMMENT_CHARS]) for c in (comments or [])[:MAX_COMMENTS]]

    user_msg = f"<review>\n{body}\n</review>\n"
    if comments:
        user_msg += "<comments>\n" + "\n".join(f"- {c}" for c in comments) + "\n</comments>\n"
    else:
        user_msg += "(댓글 없음)\n"
    user_msg += "JSON만 출력하세요."

    resp = client.chat.completions.create(
        model=model,
        max_completion_tokens=2000,
        response_format={"type": "json_object"},  # JSON으로만 답하게 강제
        messages=[
            {"role": "system", "content": PROMPTS[prompt_version]},
            {"role": "user", "content": user_msg},
        ],
    )
    raw = resp.choices[0].message.content or ""
    usage = {"input_tokens": resp.usage.prompt_tokens, "output_tokens": resp.usage.completion_tokens}
    return _parse_json(raw), usage
