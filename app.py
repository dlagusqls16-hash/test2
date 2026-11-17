import logging
import os
import time

import openai
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

LANGUAGES = {
    "영어": "English",
    "일본어": "Japanese",
    "베트남어": "Vietnamese",
}
FLAGS = {"영어": "🇺🇸", "일본어": "🇯🇵", "베트남어": "🇻🇳"}

# 테마 색을 하드코딩하지 않고 반투명/상속 값만 사용 → 라이트/다크 모두 대응
CUSTOM_CSS = """
<style>
.block-container { max-width: 760px; padding-top: 2.5rem; padding-bottom: 3rem; }
.app-subtitle { opacity: .7; margin: -0.5rem 0 1.5rem; font-size: 1.02rem; }
div[data-testid="stTextArea"] textarea {
    border-radius: 14px; padding: .9rem 1rem; font-size: 1rem; line-height: 1.55;
}
div[data-testid="stButton"] button {
    border-radius: 12px; padding: .65rem .5rem; font-weight: 600;
    transition: transform .08s ease, box-shadow .15s ease;
}
div[data-testid="stButton"] button:hover {
    transform: translateY(-1px); box-shadow: 0 4px 14px rgba(128,128,128,.25);
}
div[data-testid="stVerticalBlockBorderWrapper"] { border-radius: 16px; }
.result-label {
    display: inline-block; padding: .15rem .7rem; border-radius: 999px;
    background: rgba(128,128,128,.16); font-size: .85rem; font-weight: 600;
    margin-bottom: .4rem;
}
.char-count { text-align: right; opacity: .65; font-size: .85rem; margin-top: -0.6rem; }
@media (max-width: 640px) {
    .block-container { padding-left: 1rem; padding-right: 1rem; padding-top: 1.5rem; }
    h1 { font-size: 1.7rem !important; white-space: nowrap; }
    /* 좁은 화면에서도 버튼 3개를 한 줄에 유지 */
    div[data-testid="stHorizontalBlock"] { flex-wrap: nowrap; gap: .4rem; }
    div[data-testid="stHorizontalBlock"] > div { min-width: 0; }
    div[data-testid="stButton"] button { padding: .55rem .2rem; font-size: .9rem; }
}
</style>
"""
MAX_CHARS = 5000
DEFAULT_MODEL = "gpt-6.1-sol"
REQUEST_TIMEOUT_SEC = 30
COOLDOWN_SEC = 3  # 같은 세션에서 연속 호출 최소 간격
MAX_REQUESTS_PER_SESSION = 50  # 세션당 최대 호출 수 (비용 방어)

logger = logging.getLogger("translator")

SYSTEM_PROMPT = (
    "You are a professional translator. Translate the user's text into {lang}. "
    "Detect the source language automatically. Preserve meaning, tone, and line breaks. "
    "Output only the translation, with no explanations, quotes, or prefixes."
)


class ConfigError(Exception):
    pass


def get_setting(name: str, default: str | None = None) -> str | None:
    """환경변수(.env 포함) 우선, 없으면 st.secrets 조회."""
    value = os.getenv(name)
    if value:
        return value
    try:
        value = st.secrets.get(name)
    except Exception:
        value = None
    return value or default


def translate(text: str, target_language: str) -> str:
    api_key = get_setting("OPENAI_API_KEY")
    if not api_key:
        raise ConfigError("API 키가 설정되지 않았습니다. .env 파일(또는 Streamlit Secrets)에 OPENAI_API_KEY를 추가하세요.")

    client = openai.OpenAI(
        api_key=api_key,
        base_url=get_setting("OPENAI_BASE_URL"),
        timeout=REQUEST_TIMEOUT_SEC,
        max_retries=1,
    )
    # temperature는 지정하지 않는다: gpt-6.1-sol은 기본값(1) 외의 값을 거부함(400).
    response = client.chat.completions.create(
        model=get_setting("OPENAI_MODEL", DEFAULT_MODEL),
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT.format(lang=target_language)},
            {"role": "user", "content": text},
        ],
    )
    return (response.choices[0].message.content or "").strip()


def friendly_error(exc: Exception) -> str:
    if isinstance(exc, ConfigError):
        return str(exc)
    if isinstance(exc, openai.AuthenticationError):
        return "API 키가 올바르지 않습니다. 키를 확인해 주세요."
    if isinstance(exc, openai.RateLimitError):
        return "요청 한도를 초과했습니다. 잠시 후 다시 시도해 주세요."
    if isinstance(exc, openai.NotFoundError):
        return "모델을 찾을 수 없습니다. OPENAI_MODEL 설정을 확인해 주세요."
    if isinstance(exc, openai.PermissionDeniedError):
        return "이 API 키로는 해당 모델을 사용할 권한이 없습니다. 키와 모델 설정을 확인해 주세요."
    if isinstance(exc, openai.BadRequestError):
        return "요청이 거부되었습니다. 모델 설정(OPENAI_MODEL)이나 요청 값을 확인해 주세요."
    if isinstance(exc, openai.APITimeoutError):
        return f"응답이 {REQUEST_TIMEOUT_SEC}초 안에 오지 않았습니다. 잠시 후 다시 시도해 주세요."
    if isinstance(exc, openai.APIConnectionError):
        return "네트워크 연결에 문제가 있습니다. 잠시 후 다시 시도해 주세요."
    if isinstance(exc, openai.APIError):
        return "번역 서비스에서 오류가 발생했습니다. 잠시 후 다시 시도해 주세요."
    return "알 수 없는 오류가 발생했습니다. 잠시 후 다시 시도해 주세요."


def log_error(exc: Exception) -> None:
    """원인 파악용 서버 로그. 사용자 입력 원문은 남기지 않는다."""
    status = getattr(exc, "status_code", None)
    message = getattr(exc, "message", None) or str(exc)
    logger.error("translate failed: %s status=%s message=%s", type(exc).__name__, status, message[:300])


def check_rate_limit() -> str | None:
    """세션 단위 호출 제한. 제한에 걸리면 안내 메시지를 반환."""
    state = st.session_state
    count = state.get("request_count", 0)
    if count >= MAX_REQUESTS_PER_SESSION:
        return f"이 세션의 번역 횟수({MAX_REQUESTS_PER_SESSION}회)를 모두 사용했습니다. 새로고침 후 다시 이용해 주세요."
    wait = COOLDOWN_SEC - (time.monotonic() - state.get("last_request_at", float("-inf")))
    if wait > 0:
        return f"요청이 너무 빠릅니다. {wait:.0f}초 후 다시 눌러 주세요." if wait >= 1 else "요청이 너무 빠릅니다. 잠시 후 다시 눌러 주세요."
    return None


def handle_translate(text: str, label: str, target: str) -> None:
    st.session_state["result"] = None  # 새 요청이면 이전 결과는 지운다
    text = text.strip()
    if not text:
        st.warning("번역할 텍스트를 입력해 주세요.")
        return
    if len(text) > MAX_CHARS:
        st.error(f"입력은 최대 {MAX_CHARS:,}자까지 가능합니다. (현재 {len(text):,}자)")
        return
    limit_msg = check_rate_limit()
    if limit_msg:
        st.warning(limit_msg)
        return
    st.session_state["request_count"] = st.session_state.get("request_count", 0) + 1
    st.session_state["last_request_at"] = time.monotonic()
    try:
        with st.spinner("번역 중..."):
            result = translate(text, target)
    except Exception as exc:  # 앱이 멈추지 않도록 모든 오류를 안내 메시지로 변환
        log_error(exc)
        st.error(friendly_error(exc))
        return
    if not result:
        st.error("번역 결과가 비어 있습니다. 다시 시도해 주세요.")
        return
    st.session_state["result"] = (label, result)


def show_result() -> None:
    saved = st.session_state.get("result")
    if not saved:
        return
    label, result = saved
    st.markdown(
        f'<span class="result-label">{FLAGS[label]} {label}로 번역됨</span>',
        unsafe_allow_html=True,
    )
    st.code(result, language=None, wrap_lines=True)


def render_ui() -> None:
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    st.title("🌐 다국어 번역기")
    st.markdown(
        '<p class="app-subtitle">글을 입력하고 원하는 언어 버튼을 누르면 바로 번역해 드려요.</p>',
        unsafe_allow_html=True,
    )

    text = st.text_area(
        "번역할 텍스트",
        height=180,
        placeholder="여기에 번역할 내용을 입력하세요…",
        label_visibility="collapsed",
        key="input_text",
    )
    n_chars = len(text.strip())  # 검증(handle_translate)과 같은 기준으로 센다
    over_style = ' style="color:#e5484d;opacity:1"' if n_chars > MAX_CHARS else ""
    st.markdown(
        f'<div class="char-count"{over_style}>{n_chars:,} / {MAX_CHARS:,}자</div>',
        unsafe_allow_html=True,
    )

    cols = st.columns(len(LANGUAGES))
    clicked = None
    for col, (label, target) in zip(cols, LANGUAGES.items()):
        if col.button(f"{FLAGS[label]} {label}", use_container_width=True):
            clicked = (label, target)

    with st.container(border=True):
        st.markdown("**번역 결과**")
        if clicked:
            handle_translate(text, *clicked)
        show_result()
        if not clicked and not st.session_state.get("result"):
            st.caption("언어 버튼을 누르면 결과가 여기에 표시됩니다.")


def main() -> None:
    st.set_page_config(page_title="다국어 번역기", page_icon="🌐", layout="centered")
    render_ui()


if __name__ == "__main__":
    main()
