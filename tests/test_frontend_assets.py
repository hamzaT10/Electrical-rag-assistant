from pathlib import Path


def test_frontend_assets_exist() -> None:
    root = Path(__file__).resolve().parents[1]
    assert (root / "frontend" / "index.html").exists()
    assert (root / "frontend" / "styles.css").exists()
    assert (root / "frontend" / "app.js").exists()
    assert (root / "frontend" / "nginx.conf").exists()


def test_frontend_uses_api_proxy_prefix() -> None:
    root = Path(__file__).resolve().parents[1]
    app_js = (root / "frontend" / "app.js").read_text(encoding="utf-8")
    assert "/api/chat" in app_js
    assert "/api/health" in app_js
    assert "/api/documents/upload" in app_js
    assert "/api/documents" in app_js
    assert "/api/ingestion/jobs" in app_js
