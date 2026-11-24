"""분석 결과를 친구에게 보내기 좋은 이미지 카드(PNG)로 만든다."""
import io
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ASSETS = Path(__file__).parent / "assets"

# 컴퓨터마다 한글 글꼴 위치가 달라서 차례대로 찾아본다
FONT_CANDIDATES = {
    "bold": [
        ASSETS / "font-bold.ttf",
        Path("C:/Windows/Fonts/malgunbd.ttf"),
        Path("/System/Library/Fonts/AppleSDGothicNeo.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"),
        Path("/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf"),
    ],
    "regular": [
        ASSETS / "font.ttf",
        Path("C:/Windows/Fonts/malgun.ttf"),
        Path("/System/Library/Fonts/AppleSDGothicNeo.ttc"),
        Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
        Path("/usr/share/fonts/truetype/nanum/NanumGothic.ttf"),
    ],
}

W = 1080
PAD = 72
BG = (251, 250, 247)
INK = (31, 31, 28)
SUB = (107, 107, 102)
LINE = (231, 229, 223)
RED = (210, 69, 59)
GREEN = (46, 139, 87)
HL = (255, 240, 160)


def _font(kind: str, size: int):
    for path in FONT_CANDIDATES[kind]:
        if path.exists():
            try:
                return ImageFont.truetype(str(path), size)
            except OSError:
                continue
    return None


def fonts_available() -> bool:
    return _font("bold", 20) is not None and _font("regular", 20) is not None


def _wrap(text: str, font, width: int) -> list[str]:
    """글자 단위로 잘라 width 픽셀 안에 들어가게 줄을 나눈다."""
    lines, cur = [], ""
    for ch in str(text or "").replace("\n", " "):
        if font.getlength(cur + ch) <= width:
            cur += ch
        else:
            lines.append(cur.rstrip())
            cur = ch.lstrip()
    if cur.strip():
        lines.append(cur.rstrip())
    return lines or [""]


def make_card(title: str, source: str, trust: int | None, ad: int, cons: list[dict], pros: list[dict],
              checklist: list[str]) -> bytes | None:
    """결과 카드 PNG를 만든다. 한글 글꼴을 못 찾으면 None."""
    if not fonts_available():
        return None
    f_title, f_h, f_b, f_s, f_xs = (_font("bold", 46), _font("bold", 32), _font("bold", 30),
                                    _font("regular", 26), _font("regular", 22))
    inner = W - PAD * 2
    blocks = []  # (높이, 그리는 함수)

    def add_text(lines, font, color, gap, after=0):
        h = len(lines) * gap + after

        def draw(d, y):
            for i, line in enumerate(lines):
                d.text((PAD, y + i * gap), line, font=font, fill=color)
        blocks.append((h, draw))

    def add_space(h):
        blocks.append((h, lambda d, y: None))

    def add_rule():
        blocks.append((40, lambda d, y: d.line([(PAD, y + 20), (W - PAD, y + 20)], fill=LINE, width=2)))

    add_text(_wrap(title, f_title, inner)[:3], f_title, INK, 60, 6)
    add_text(_wrap(source, f_s, inner)[:1], f_s, SUB, 36, 18)

    badge = f"신뢰 점수 {trust}/100   ·   광고 의심 {ad}점" if trust is not None else f"광고 의심 {ad}점"

    def draw_badge(d, y):
        w = f_b.getlength(badge) + 48
        d.rounded_rectangle([PAD, y, PAD + w, y + 64], radius=16, fill=(255, 255, 255), outline=LINE, width=2)
        d.text((PAD + 24, y + 14), badge, font=f_b, fill=INK)
    blocks.append((88, draw_badge))

    def section(name, color, items, key):
        if not items:
            return
        add_rule()
        add_text([name], f_h, color, 46, 8)
        for i, it in enumerate(items[:3], 1):
            point = it.get(key) if isinstance(it, dict) else it
            lines = _wrap(f"{i}. {point}", f_b, inner)
            add_text(lines[:3], f_b, INK, 42, 4)
            ev = it.get("evidence") if isinstance(it, dict) else ""
            if ev:
                short = ev if len(ev) <= 70 else ev[:68] + "…"
                ev_lines = _wrap(short, f_s, inner - 40)[:2]
                ev_lines[0] = "“" + ev_lines[0]
                ev_lines[-1] = ev_lines[-1] + "”"
                add_text(ev_lines, f_s, SUB, 36, 14)
            else:
                add_space(10)

    section("치명적 단점", RED, cons, "point")
    section("핵심 장점", GREEN, pros, "point")
    if checklist:
        add_rule()
        add_text(["사기 전에 확인할 것"], f_h, INK, 46, 8)
        for c in checklist[:3]:
            add_text(_wrap(f"□ {c}", f_s, inner)[:2], f_s, INK, 38, 6)
    add_rule()
    add_text(["RealPick AI · AI 분석은 틀릴 수 있어요. 근거는 원문에서 확인하세요."], f_xs, SUB, 30)

    logo_h = 0
    logo = None
    if (ASSETS / "logo.png").exists():
        logo = Image.open(ASSETS / "logo.png").convert("RGBA")
        logo = logo.resize((300, int(logo.height * 300 / logo.width)))
        logo_h = logo.height + 36

    height = PAD + logo_h + sum(h for h, _ in blocks) + PAD
    img = Image.new("RGB", (W, height), BG)
    if logo:
        img.paste(logo, (PAD - 8, PAD), logo)
    d = ImageDraw.Draw(img)
    y = PAD + logo_h
    for h, draw in blocks:
        draw(d, y)
        y += h
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()
