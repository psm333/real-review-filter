"""유튜브 자막/댓글, 블로그 본문을 가져오는 함수들."""
import re

import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}


class FetchError(Exception):
    """가져오기 실패 시 사용자에게 보여줄 메시지를 담는 예외."""


# ---------------- 유튜브 ----------------
def extract_video_id(url: str) -> str:
    patterns = [
        r"[?&]v=([\w-]{11})",
        r"youtu\.be/([\w-]{11})",
        r"shorts/([\w-]{11})",
        r"embed/([\w-]{11})",
    ]
    for p in patterns:
        m = re.search(p, url)
        if m:
            return m.group(1)
    raise FetchError("유튜브 링크에서 영상 ID를 찾지 못했어요. 링크를 다시 확인해 주세요.")


def get_youtube_transcript(video_id: str) -> str:
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
    except ImportError as e:
        raise FetchError("youtube-transcript-api 가 설치되지 않았어요.") from e

    langs = ["ko", "en"]
    try:
        try:  # 새 버전(1.x)
            fetched = YouTubeTranscriptApi().fetch(video_id, languages=langs)
            return " ".join(s.text for s in fetched)
        except AttributeError:  # 옛 버전(0.x)
            data = YouTubeTranscriptApi.get_transcript(video_id, languages=langs)
            return " ".join(d["text"] for d in data)
    except Exception as e:
        raise FetchError(
            "이 영상은 자막을 가져올 수 없어요 (자막이 없거나 막혀 있음). "
            "'직접 붙여넣기'로 내용을 넣어 주세요."
        ) from e


def get_youtube_comments(video_id: str, api_key: str, max_count: int = 50) -> list[str]:
    """YouTube Data API v3로 인기순 댓글 + 답글을 가져온다. 작성자 이름은 가져오지 않는다."""
    if not api_key:
        return []
    url = "https://www.googleapis.com/youtube/v3/commentThreads"
    params = {
        "part": "snippet,replies",
        "videoId": video_id,
        "maxResults": min(max_count, 100),
        "order": "relevance",
        "textFormat": "plainText",
        "key": api_key,
    }
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
    except requests.RequestException as e:
        raise FetchError("댓글을 가져오지 못했어요 (댓글이 꺼져 있거나 API 키 문제).") from e
    comments = []
    for it in r.json().get("items", []):
        comments.append(it["snippet"]["topLevelComment"]["snippet"]["textOriginal"])
        # 답글(대댓글)도 포함 — API는 스레드당 최대 5개까지 함께 준다
        for rep in it.get("replies", {}).get("comments", []):
            comments.append("↳ (답글) " + rep["snippet"]["textOriginal"])
    return comments[:max_count]


# ---------------- 블로그 ----------------
def _naver_postview_url(url: str) -> str | None:
    m = re.search(r"blog\.naver\.com/([\w-]+)/(\d+)", url)
    if m:
        return f"https://blog.naver.com/PostView.naver?blogId={m.group(1)}&logNo={m.group(2)}"
    m = re.search(r"blogId=([\w-]+).*?logNo=(\d+)", url)
    if m:
        return f"https://blog.naver.com/PostView.naver?blogId={m.group(1)}&logNo={m.group(2)}"
    return None


def get_blog_text(url: str) -> str:
    target = _naver_postview_url(url) or url
    try:
        r = requests.get(target, headers=HEADERS, timeout=10)
        r.raise_for_status()
    except requests.RequestException as e:
        raise FetchError("블로그 글을 열지 못했어요. '직접 붙여넣기'를 써 주세요.") from e

    soup = BeautifulSoup(r.text, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    main = (
        soup.select_one("div.se-main-container")  # 네이버 스마트에디터
        or soup.select_one("#postViewArea")  # 네이버 옛 에디터
        or soup.find("article")  # 티스토리 등
        or soup.body
    )
    text = main.get_text("\n", strip=True) if main else ""
    if len(text) < 100:
        raise FetchError("블로그 본문이 거의 비어 있어요. '직접 붙여넣기'를 써 주세요.")
    return text
