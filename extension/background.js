// RealPick 앱 주소. 인터넷에 올렸다면 그 주소로 바꾸세요 (예: https://realpick.streamlit.app)
const APP_URL = "http://localhost:8501";

// 툴바의 RealPick 버튼을 누르면, 지금 보고 있는 페이지 주소를 앱으로 넘겨서 새 탭에서 분석한다.
chrome.action.onClicked.addListener((tab) => {
  if (!tab.url || !/^https?:/.test(tab.url)) return;
  chrome.tabs.create({ url: `${APP_URL}/?url=${encodeURIComponent(tab.url)}` });
});
