"""프롬프트 A/B/C + 모델 비교 실험.

사용법:
  1) samples/links.txt 에 유튜브 리뷰 링크를 한 줄에 하나씩 적는다 (5개 정도).
     또는 samples/ 폴더에 리뷰를 .txt 로 넣는다
     (댓글도 넣으려면 본문 아래에 '===댓글===' 줄을 쓰고 한 줄에 하나씩).
  2) py experiment.py
  3) results/experiment_results.md 를 열어서 사람이 직접 점수를 매긴다.

실험 조건:
  - 프롬프트 A, B, C  × 기본 모델
  - 프롬프트 C        × 비교 모델 (모델 비교)
"""
import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from analyzer import DEFAULT_MODEL, analyze
from fetchers import FetchError, extract_video_id, get_youtube_comments, get_youtube_transcript

load_dotenv()
KEY = os.getenv("OPENAI_API_KEY")
YT_KEY = os.getenv("YOUTUBE_API_KEY", "")
COMPARE_MODEL = "gpt-4o-mini"

CONDITIONS = [
    ("A", DEFAULT_MODEL),
    ("B", DEFAULT_MODEL),
    ("C", DEFAULT_MODEL),
    ("D", DEFAULT_MODEL),
    ("D", COMPARE_MODEL),
]

SAMPLE_DIR = Path("samples")
# py experiment.py 2  처럼 숫자를 주면 experiment_results_2.md 로 따로 저장
SUFFIX = f"_{sys.argv[1]}" if len(sys.argv) > 1 else ""
OUT = Path(f"results/experiment_results{SUFFIX}.md")


# ---------- 자동 채점 ----------
def _norm(t: str) -> str:
    """띄어쓰기·문장부호를 없애서 비교하기 쉽게 만든다."""
    return re.sub(r"[\s\W_]+", "", t or "")


def in_source(snippet: str, source_norm: str) -> bool:
    """AI가 적은 문장이 원문(자막+댓글)에 실제로 있는지 확인. '...'로 잘린 부분은 조각마다 확인."""
    parts = [_norm(p) for p in re.split(r"\.{2,}|…", snippet or "")]
    parts = [p for p in parts if len(p) >= 6]
    return bool(parts) and all(p in source_norm for p in parts)


def auto_check(result: dict, body: str, comments: list[str]) -> dict:
    src = _norm(body + " ".join(comments))
    items = result.get("cons", []) + result.get("pros", [])
    ev = [it.get("evidence", "") for it in items if it.get("evidence")]
    ads = result.get("ad_sentences", [])
    ev_ok = sum(in_source(e, src) for e in ev)
    ad_ok = sum(in_source(a, src) for a in ads)
    return {
        "cons_n": len(result.get("cons", [])),
        "evidence": f"{ev_ok}/{len(ev)}",
        "ad_real": f"{ad_ok}/{len(ads)}",
        "ev_ok": ev_ok, "ev_n": len(ev), "ad_ok": ad_ok, "ad_n": len(ads),
    }


def load_samples():
    """(이름, 본문, 댓글목록) 리스트를 만든다."""
    samples = []
    links = SAMPLE_DIR / "links.txt"
    if links.exists():
        for i, line in enumerate(links.read_text(encoding="utf-8").splitlines(), 1):
            url = line.strip()
            if not url or url.startswith("#"):
                continue
            try:
                vid = extract_video_id(url)
                body = get_youtube_transcript(vid)
                comments = get_youtube_comments(vid, YT_KEY) if YT_KEY else []
                samples.append((f"유튜브{i} ({vid})", body, comments))
                print(f"✅ 가져옴: {url} (댓글 {len(comments)}개)")
            except FetchError as e:
                print(f"⚠️ 건너뜀: {url} → {e}")
    for path in sorted(SAMPLE_DIR.glob("*.txt")):
        if path.name == "links.txt":
            continue
        raw = path.read_text(encoding="utf-8")
        if "===댓글===" in raw:
            body, cm = raw.split("===댓글===", 1)
            samples.append((path.name, body.strip(), [c.strip() for c in cm.splitlines() if c.strip()]))
        else:
            samples.append((path.name, raw.strip(), []))
    return samples


def main():
    samples = load_samples()
    if not samples:
        print("samples/links.txt 에 링크를 적거나 samples/ 에 .txt 파일을 넣어 주세요.")
        return
    OUT.parent.mkdir(exist_ok=True)

    lines = [
        "# 프롬프트·모델 비교 실험 결과\n",
        f"- 실험 날짜: {time.strftime('%Y-%m-%d %H:%M')}",
        f"- 샘플 수: {len(samples)}개",
        "- 조건: " + ", ".join(f"프롬프트 {v} × {m}" for v, m in CONDITIONS),
        "\n## ✍️ 채점 방법 (사람이 직접)",
        "1. 각 샘플마다 영상을 보고 **진짜 단점 3개(정답)** 를 먼저 적는다.",
        "2. 각 조건의 결과를 보고 아래 요약표의 빈칸을 채운다.",
        "   - **정답 단점 일치**: AI 단점 중 정답과 같은 것의 수 (0~3)",
        "   - **광고 오판**: 걸러낸 문장 중 광고가 아니라 리뷰어 의견인데 잘못 거른 수",
        "\n## 🤖 자동 채점 (코드가 계산)",
        "   - **단점 수**: 단점을 몇 개 찾았나 (0이면 핵심 기능 실패)",
        "   - **근거 실재**: 장단점 근거 문장 중 원문에 실제로 있는 비율 (지어낸 근거 확인)",
        "   - **광고문장 실재**: 걸러낸 광고 문장 중 원문에 실제로 있는 비율 (지어낸 문장 확인)\n",
    ]
    rows = []
    for name, body, comments in samples:
        lines.append(f"\n---\n## 📄 {name}  (댓글 {len(comments)}개)\n")
        lines.append("**👤 사람이 정한 정답 단점:** 1. ______  2. ______  3. ______\n")
        for v, model in CONDITIONS:
            label = f"프롬프트 {v} × {model}"
            t0 = time.time()
            try:
                result, usage = analyze(body, comments, v, model, api_key=KEY)
            except Exception as e:  # 실험은 실패해도 계속
                lines.append(f"\n### {label}: ❌ 오류 {e}\n")
                continue
            sec = time.time() - t0
            ads = result.get("ad_sentences", [])
            lines.append(f"\n### {label}  (광고지수 {result.get('ad_score')}, {sec:.1f}초)\n")
            lines.append("**단점**")
            for c in result.get("cons", []):
                lines.append(f"- {c.get('point')}  ← 근거: {c.get('evidence')}")
            lines.append("\n**장점**")
            for p in result.get("pros", []):
                lines.append(f"- {p.get('point')}")
            lines.append(f"\n**걸러낸 광고 문장 ({len(ads)}개)**")
            for s in ads:
                lines.append(f"- {s}")
            ca = result.get("comment_analysis") or {}
            if ca:
                lines.append(
                    f"\n**댓글**: 광고 지적 {ca.get('ad_suspicion_count', 0)} / "
                    f"바이럴 {ca.get('viral_count', '-')} / 실사용자 {len(ca.get('real_user_opinions', []))}"
                )
            chk = auto_check(result, body, comments)
            lines.append(f"\n🤖 자동 채점: 단점 {chk['cons_n']}개 · 근거 실재 {chk['evidence']} · 광고문장 실재 {chk['ad_real']}")
            tokens = usage["input_tokens"] + usage["output_tokens"]
            rows.append((name, v, model, result.get("ad_score"), len(ads), tokens, round(sec, 1), chk))
            print(f"  {name} / {label} 완료")

    lines.append("\n---\n## 📊 요약표 (오른쪽 2칸은 사람이 채우기)\n")
    lines.append("| 샘플 | 프롬프트 | 모델 | 광고지수 | 걸러낸 문장 | 단점 수 | 근거 실재 | 광고문장 실재 | 토큰 | 시간(초) | 정답 단점 일치(/3) | 광고 오판 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        c = r[7]
        lines.append(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {c['cons_n']} | {c['evidence']} | {c['ad_real']} | {r[5]} | {r[6]} |  |  |")

    # 조건별 평균 (보고서용)
    lines.append("\n## 📈 조건별 합계 (보고서에 넣을 표)\n")
    lines.append("| 조건 | 단점 0개인 샘플 | 평균 단점 수 | 근거 실재율 | 광고문장 실재율 | 평균 시간(초) | 평균 토큰 |")
    lines.append("|---|---|---|---|---|---|---|")
    for v, model in CONDITIONS:
        rs = [r for r in rows if r[1] == v and r[2] == model]
        if not rs:
            continue
        n = len(rs)
        zero = sum(r[7]["cons_n"] == 0 for r in rs)
        cons_avg = sum(r[7]["cons_n"] for r in rs) / n
        ev_ok = sum(r[7]["ev_ok"] for r in rs); ev_n = sum(r[7]["ev_n"] for r in rs) or 1
        ad_ok = sum(r[7]["ad_ok"] for r in rs); ad_n = sum(r[7]["ad_n"] for r in rs) or 1
        t = sum(r[6] for r in rs) / n
        tok = sum(r[5] for r in rs) / n
        lines.append(f"| {v} × {model} | {zero}/{n} | {cons_avg:.1f} | {ev_ok/ev_n:.0%} | {ad_ok/ad_n:.0%} | {t:.1f} | {tok:.0f} |")
    lines.append("\n## 🏁 결론 (실험 후 작성)\n- 가장 좋았던 조건: \n- 이유: \n- 최종 선택: ")
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n✅ 저장 완료: {OUT}")


if __name__ == "__main__":
    main()
