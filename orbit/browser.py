"""Optional, bounded browser automation for Orbit.

The browser is a single shared Playwright session. Navigation is restricted to
public HTTP(S) hosts; local/private IPs and non-web schemes are rejected.
State-changing clicks are exposed only through explicit owner commands.
"""
from __future__ import annotations
import ipaddress
import os
import socket
import threading
from urllib.parse import urlparse


def validate_public_url(url: str) -> str:
    if not isinstance(url, str) or len(url) > 2048:
        raise ValueError("Enter a valid public website URL.")
    parsed = urlparse(url.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Only public HTTP and HTTPS websites are supported.")
    host = parsed.hostname.rstrip(".").lower()
    if host in {"localhost", "localhost.localdomain"} or host.endswith(".localhost") or host.endswith(".local"):
        raise PermissionError("Orbit's browser cannot access local or private network hosts.")
    try:
        addresses = {ipaddress.ip_address(host)}
    except ValueError:
        try:
            addresses = {ipaddress.ip_address(item[4][0]) for item in socket.getaddrinfo(host, parsed.port or (443 if parsed.scheme == "https" else 80))}
        except OSError as exc:
            raise ValueError(f"Website hostname could not be resolved: {exc}") from exc
    if not addresses or any(not item.is_global for item in addresses):
        raise PermissionError("Orbit's browser only opens publicly routable websites.")
    return url.strip()


class BrowserTools:
    def __init__(self, timeout_ms=None):
        self.timeout_ms = int(timeout_ms or os.environ.get("ORBIT_BROWSER_TIMEOUT_MS", "20000"))
        self._lock = threading.RLock()
        self._playwright = None
        self._browser = None
        self._page = None
        self._last_url = ""

    def _ensure_page(self):
        if self._page is not None and not self._page.is_closed():
            return self._page
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            raise RuntimeError("Browser automation is optional. Install requirements-browser.txt and the Playwright Chromium browser.") from exc
        if self._playwright is None:
            self._playwright = sync_playwright().start()
            self._browser = self._playwright.chromium.launch(
                headless=True,
                args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-gpu"]
            )
        self._page = self._browser.new_page()
        self._page.route("**/*", self._guard_request)
        self._page.set_default_timeout(self.timeout_ms)
        self._page.set_viewport_size({"width": 1280, "height": 800})
        return self._page

    @staticmethod
    def _guard_request(route):
        try:
            validate_public_url(route.request.url)
            route.continue_()
        except Exception:
            route.abort("blockedbyclient")

    def open(self, url: str) -> dict:
        safe_url = validate_public_url(url)
        with self._lock:
            page = self._ensure_page()
            response = page.goto(safe_url, wait_until="domcontentloaded", timeout=self.timeout_ms)
            self._last_url = page.url
            return {
                "url": page.url,
                "title": page.title()[:300],
                "status": response.status if response else None,
                "text": page.locator("body").inner_text(timeout=self.timeout_ms)[:16000],
            }

    def read(self) -> dict:
        with self._lock:
            if self._page is None or self._page.is_closed():
                raise RuntimeError("No website is open. Ask Orbit to open a public URL first.")
            return {"url": self._page.url, "title": self._page.title()[:300],
                    "text": self._page.locator("body").inner_text(timeout=self.timeout_ms)[:20000]}

    def click(self, selector: str) -> dict:
        if not isinstance(selector, str) or not selector.strip() or len(selector) > 500:
            raise ValueError("A CSS selector or accessible selector is required.")
        with self._lock:
            if self._page is None or self._page.is_closed():
                raise RuntimeError("No website is open.")
            before = self._page.url
            target = self._page.locator(selector).first
            details = target.evaluate("""el => ({
                tag: (el.tagName || '').toLowerCase(),
                type: (el.getAttribute('type') || '').toLowerCase(),
                role: (el.getAttribute('role') || '').toLowerCase(),
                text: (el.innerText || el.getAttribute('aria-label') || '').trim().slice(0, 200)
            })""")
            risky_words = ("buy", "purchase", "pay", "checkout", "delete", "remove",
                           "submit", "send", "confirm", "place order", "transfer", "subscribe")
            if details["type"] == "submit" or details["role"] == "submit" or any(
                word in details["text"].casefold() for word in risky_words
            ):
                raise PermissionError("That control may submit a form or perform a consequential action. Orbit will not click it automatically.")
            self._page.locator(selector).first.click(timeout=self.timeout_ms)
            self._last_url = self._page.url
            return {"url": self._page.url, "previous_url": before,
                    "title": self._page.title()[:300],
                    "text": self._page.locator("body").inner_text(timeout=self.timeout_ms)[:12000]}

    def fill(self, selector: str, value: str) -> dict:
        if not isinstance(selector, str) or not selector.strip() or len(selector) > 500:
            raise ValueError("A CSS selector is required.")
        if not isinstance(value, str) or len(value) > 10000:
            raise ValueError("Text to enter must be at most 10,000 characters.")
        with self._lock:
            if self._page is None or self._page.is_closed():
                raise RuntimeError("No website is open.")
            self._page.locator(selector).first.fill(value, timeout=self.timeout_ms)
            return {"url": self._page.url, "selector": selector, "filled": True,
                    "note": "Text entered; nothing was submitted."}

    def screenshot(self) -> bytes:
        with self._lock:
            if self._page is None or self._page.is_closed():
                raise RuntimeError("No website is open.")
            return self._page.screenshot(full_page=False, type="png")

    def close(self):
        with self._lock:
            if self._browser is not None:
                self._browser.close()
            if self._playwright is not None:
                self._playwright.stop()
            self._browser = self._playwright = self._page = None
            self._last_url = ""
