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


def get_youtube_segments(video_id: str) -> list[dict]:
    """자막을 [{"start": 초, "text": 문장}, ...] 로 가져온다. 근거 문장 → 영상 장면 이동에 쓴다."""
    try:
        from youtube_transcript_api import YouTubeTranscriptApi
    except ImportError as e:
        raise FetchError("youtube-transcript-api 가 설치되지 않았어요.") from e

    langs = ["ko", "en"]
    try:
        try:  # 새 버전(1.x)
            fetched = YouTubeTranscriptApi().fetch(video_id, languages=langs)
            return [{"start": float(s.start), "text": s.text} for s in fetched]
        except AttributeError:  # 옛 버전(0.x)
            data = YouTubeTranscriptApi.get_transcript(video_id, languages=langs)
            return [{"start": float(d["start"]), "text": d["text"]} for d in data]
    except Exception as e:
        raise FetchError(
            "이 영상은 자막을 가져올 수 없어요 (자막이 없거나 막혀 있음). "
            "리뷰 내용을 복사해서 붙여넣어 주세요."
        ) from e


def get_youtube_transcript(video_id: str) -> str:
    return " ".join(seg["text"] for seg in get_youtube_segments(video_id))


def search_youtube_reviews(product: str, api_key: str, n: int = 5) -> list[str]:
    """제품 이름으로 유튜브 리뷰 영상을 찾아 링크 목록을 돌려준다 (YouTube Data API 필요)."""
    if not api_key:
        raise FetchError("제품 이름 검색은 YOUTUBE_API_KEY 가 있어야 해요. 리뷰 링크를 직접 넣어 주세요.")
    params = {
        "part": "snippet",
        "q": f"{product} 리뷰",
        "type": "video",
        "maxResults": min(max(n * 2, n), 15),  # 자막 없는 영상이 섞일 수 있어 넉넉히
        "relevanceLanguage": "ko",
        "regionCode": "KR",
        "videoCaption": "any",
        "key": api_key,
    }
    try:
        r = requests.get("https://www.googleapis.com/youtube/v3/search", params=params, timeout=10)
        r.raise_for_status()
    except requests.RequestException as e:
        raise FetchError("유튜브 검색에 실패했어요. 잠시 후 다시 시도하거나 링크를 직접 넣어 주세요.") from e
    ids = [it["id"]["videoId"] for it in r.json().get("items", []) if it.get("id", {}).get("videoId")]
    if not ids:
        raise FetchError(f"'{product}' 리뷰 영상을 찾지 못했어요. 제품 이름을 바꿔 보세요.")
    return [f"https://www.youtube.com/watch?v={v}" for v in ids]


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


def get_youtube_meta(video_id: str) -> dict:
    """영상 제목·채널·썸네일. 키가 필요 없는 oEmbed를 쓰고, 실패해도 기본값으로 진행한다."""
    url = f"https://www.youtube.com/watch?v={video_id}"
    meta = {
        "title": "유튜브 리뷰 영상",
        "channel": "YouTube",
        "thumb": f"https://i.ytimg.com/vi/{video_id}/mqdefault.jpg",
        "url": url,
    }
    try:
        r = requests.get(
            "https://www.youtube.com/oembed",
            params={"url": url, "format": "json"},
            headers=HEADERS,
            timeout=8,
        )
        if r.ok:
            j = r.json()
            meta["title"] = j.get("title") or meta["title"]
            meta["channel"] = j.get("author_name") or meta["channel"]
    except (requests.RequestException, ValueError):
        pass
    return meta


# ---------------- 블로그 ----------------
def _naver_postview_url(url: str) -> str | None:
    m = re.search(r"blog\.naver\.com/([\w-]+)/(\d+)", url)
    if m:
        return f"https://blog.naver.com/PostView.naver?blogId={m.group(1)}&logNo={m.group(2)}"
    m = re.search(r"blogId=([\w-]+).*?logNo=(\d+)", url)
    if m:
        return f"https://blog.naver.com/PostView.naver?blogId={m.group(1)}&logNo={m.group(2)}"
    return None


def get_blog_post(url: str) -> tuple[dict, str]:
    """블로그 글의 (정보, 본문)을 가져온다. 정보 = 제목, 블로그 이름, 대표 이미지, 주소."""
    target = _naver_postview_url(url) or url
    try:
        r = requests.get(target, headers=HEADERS, timeout=10)
        r.raise_for_status()
    except requests.RequestException as e:
        raise FetchError("블로그 글을 열지 못했어요. 글 내용을 복사해서 붙여넣어 주세요.") from e

    soup = BeautifulSoup(r.text, "html.parser")

    def og(prop: str) -> str:
        tag = soup.find("meta", attrs={"property": prop})
        return (tag.get("content") or "").strip() if tag else ""

    title_tag = soup.select_one(".se-title-text")
    meta = {
        "title": og("og:title") or (title_tag.get_text(strip=True) if title_tag else "")
        or (soup.title.get_text(strip=True) if soup.title else "블로그 리뷰"),
        "channel": og("og:site_name") or ("네이버 블로그" if "naver" in url else "블로그"),
        "thumb": og("og:image"),
        "url": url,
    }

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
        raise FetchError("블로그 본문이 거의 비어 있어요. 글 내용을 복사해서 붙여넣어 주세요.")
    return meta, text


def get_blog_text(url: str) -> str:
    return get_blog_post(url)[1]
