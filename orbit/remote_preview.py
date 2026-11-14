"""Same-origin HTTP proxy for a user-configured public cloud preview tunnel.

The target is fixed by ORBIT_REMOTE_PREVIEW_URL; request paths cannot select hosts.
Only GET/POST are proxied. WebSockets/HMR are intentionally not proxied.
"""
from __future__ import annotations

import json
import os
import re
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlsplit, urlunsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

MAX_REQUEST_BYTES = 2 * 1024 * 1024
MAX_RESPONSE_BYTES = 8 * 1024 * 1024


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def configured_base():
    raw = os.environ.get("ORBIT_REMOTE_PREVIEW_URL", "").strip()
    if not raw:
        return None
    parsed = urlsplit(raw)
    if (parsed.scheme != "https" or not parsed.hostname or parsed.username or
            parsed.password or parsed.query or parsed.fragment):
        raise ValueError("ORBIT_REMOTE_PREVIEW_URL must be a public HTTPS origin or path without credentials, query or fragment.")
    return raw.rstrip("/") + "/"


def proxy(handler, method):
    base = configured_base()
    if not base:
        return handler._json(503, {"error": "Remote preview is not configured. Set ORBIT_REMOTE_PREVIEW_URL to your public HTTPS tunnel URL."})
    if method not in {"GET", "POST"}:
        return handler._json(405, {"error": "Remote preview supports GET and POST only."})
    prefix = "/remote-preview/"
    raw_path = handler.path[len(prefix):]
    path_only = raw_path.split("?", 1)[0]
    decoded_parts = unquote(path_only).split("/")
    if any(part in {".", ".."} or "\\" in part for part in decoded_parts):
        return handler._json(400, {"error": "Invalid remote preview path."})
    base_parts = urlsplit(base)
    target = urlsplit(base.rstrip("/") + "/" + raw_path)
    if target.scheme != "https" or target.netloc != base_parts.netloc or not target.path.startswith(base_parts.path):
        return handler._json(400, {"error": "Remote preview target escaped the configured origin."})
    body = None
    headers = {"Accept": handler.headers.get("Accept", "text/html,application/json,*/*"),
               "User-Agent": "Orbit-Remote-Preview/1.0"}
    if method == "POST":
        try:
            length = int(handler.headers.get("Content-Length", "0"))
        except ValueError:
            return handler._json(400, {"error": "Invalid request length."})
        if length < 0 or length > MAX_REQUEST_BYTES:
            return handler._json(413, {"error": "Remote preview POST body exceeds 2 MB."})
        body = handler.rfile.read(length) if length else b""
        content_type = handler.headers.get("Content-Type", "application/json")
        if content_type.split(";", 1)[0].strip().lower() not in {
            "application/json", "application/x-www-form-urlencoded", "text/plain"
        }:
            return handler._json(415, {"error": "Unsupported remote preview content type."})
        headers["Content-Type"] = content_type
    request = Request(urlunsplit(target), data=body, headers=headers, method=method)
    opener = build_opener(NoRedirect)
    status, response_headers, payload = 502, {}, b""
    try:
        with opener.open(request, timeout=20) as response:
            status = response.status
            response_headers = response.headers
            payload = response.read(MAX_RESPONSE_BYTES + 1)
    except HTTPError as exc:
        status = exc.code
        response_headers = exc.headers
        payload = exc.read(MAX_RESPONSE_BYTES + 1)
    except (URLError, TimeoutError, OSError) as exc:
        return handler._json(502, {"error": "Remote preview could not reach the configured tunnel: " + type(exc).__name__})
    if len(payload) > MAX_RESPONSE_BYTES:
        return handler._json(502, {"error": "Remote preview response exceeded the 8 MB limit."})
    content_type = response_headers.get("Content-Type", "application/octet-stream")
    if "text/html" in content_type.lower():
        page = payload.decode("utf-8", errors="replace")
        base_tag = '<base href="/remote-preview/">'
        if re.search(r"<head(?:\s[^>]*)?>", page, flags=re.I):
            page = re.sub(r"(<head(?:\s[^>]*)?>)", r"\1" + base_tag, page, count=1, flags=re.I)
        else:
            page = base_tag + page
        # Rewrite root-relative HTML resources so they remain inside this proxy path.
        page = re.sub(r'((?:src|href|action)=["\'])/(?!/)', r'\1/remote-preview/', page, flags=re.I)
        payload = page.encode("utf-8")
        content_type = "text/html; charset=utf-8"
    elif "text/css" in content_type.lower():
        css = payload.decode("utf-8", errors="replace")
        css = re.sub(r"url\(([\"']?)/(?!/)", r"url(\1/remote-preview/", css, flags=re.I)
        payload = css.encode("utf-8")
    handler.send_response(status)
    location = response_headers.get("Location")
    if location:
        location_parts = urlsplit(location)
        if location.startswith("/") and not location.startswith("//"):
            handler.send_header("Location", "/remote-preview/" + location.lstrip("/"))
        elif location_parts.netloc == base_parts.netloc and location_parts.scheme in {"", "https"}:
            handler.send_header("Location", "/remote-preview/" + location_parts.path.lstrip("/") +
                                (("?" + location_parts.query) if location_parts.query else ""))
    handler.send_header("Content-Type", content_type)
    handler.send_header("Content-Length", str(len(payload)))
    handler.send_header("Cache-Control", "no-store")
    handler.send_header("X-Content-Type-Options", "nosniff")
    handler.end_headers()
    handler.wfile.write(payload)
