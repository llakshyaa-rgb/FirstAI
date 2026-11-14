"""Optional browser-based inspection of Orbit's own loopback preview.

The inspector never submits forms or clicks controls. Network requests are
restricted to the exact loopback preview host and port.
"""
from __future__ import annotations

from urllib.parse import urlsplit


def inspect_preview(preview_url: str, timeout_ms: int = 12000) -> dict:
    parsed = urlsplit(str(preview_url or ""))
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost"} or not parsed.port:
        raise PermissionError("Preview inspection is limited to an explicit loopback HTTP preview URL.")
    try:
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise RuntimeError("Browser inspection is optional. Install requirements-browser.txt and the Playwright Chromium browser.") from exc

    errors = []
    failed_requests = []
    allowed_origin = (parsed.hostname, parsed.port)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=True, args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu"])
        try:
            page = browser.new_page(viewport={"width": 1280, "height": 800})
            timeout = max(1000, min(int(timeout_ms), 20000))
            page.set_default_timeout(timeout)
            page.on("pageerror", lambda error: errors.append(str(error)[:500]))
            page.on("console", lambda message: errors.append(message.text[:500]) if message.type == "error" else None)
            page.on("requestfailed", lambda request: failed_requests.append({
                "url": request.url[:500], "error": (request.failure or "request failed")[:300]
            }))
            def guard(route):
                request_url = urlsplit(route.request.url)
                if request_url.scheme in {"data", "blob"}:
                    route.continue_()
                elif (request_url.scheme == "http" and request_url.hostname == allowed_origin[0]
                        and request_url.port == allowed_origin[1]):
                    route.continue_()
                else:
                    route.abort("blockedbyclient")
            page.route("**/*", guard)
            response = page.goto(preview_url, wait_until="domcontentloaded", timeout=timeout)
            page.wait_for_timeout(500)
            details = page.evaluate("""() => ({
                title: document.title || "",
                headings: Array.from(document.querySelectorAll("h1,h2")).slice(0,30).map(x => (x.innerText || "").trim().slice(0,160)),
                buttons: Array.from(document.querySelectorAll("button,[role=button]")).slice(0,60).map(x => ({
                    text: (x.innerText || x.getAttribute("aria-label") || "").trim().slice(0,120),
                    disabled: !!x.disabled
                })),
                links: Array.from(document.querySelectorAll("a[href]")).slice(0,60).map(x => ({
                    text: (x.innerText || x.getAttribute("aria-label") || "").trim().slice(0,120),
                    href: x.getAttribute("href")
                })),
                forms: Array.from(document.forms).slice(0,20).map(x => ({
                    method: (x.method || "get").toUpperCase(),
                    fields: Array.from(x.querySelectorAll("input,select,textarea")).slice(0,30).map(y => ({
                        name: y.name || y.id || "",
                        type: (y.type || y.tagName).toLowerCase(),
                        label: (y.getAttribute("aria-label") || y.placeholder || "").slice(0,120)
                    }))
                })),
                images: Array.from(document.images).slice(0,30).map(x => ({
                    alt: x.alt || "", loaded: x.complete && x.naturalWidth > 0
                })),
                text_preview: (document.body?.innerText || "").slice(0,5000)
            })""")
            return {
                "url": page.url, "http_status": response.status if response else None,
                "title": details["title"], "headings": details["headings"],
                "buttons": details["buttons"], "links": details["links"],
                "forms": details["forms"], "images": details["images"],
                "text_preview": details["text_preview"],
                "console_errors": errors[:30], "failed_requests": failed_requests[:30],
                "safety": "Read-only inspection: no buttons clicked and no forms submitted.",
            }
        finally:
            browser.close()
