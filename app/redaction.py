"""Privacy and Personally Identifiable Information (PII) redaction utilities."""

import re

# Compiled regex patterns for high-throughput client-side masking
RE_EMAIL = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
    re.IGNORECASE
)

RE_URL = re.compile(
    r"(?:https?://\S+|www\.\S+|\b[A-Za-z0-9.-]+\.(?:com|org|net|in|co|io|xyz|info|app|biz|online|store|site)\b(?:/\S*)?)",
    re.IGNORECASE
)

RE_PHONE = re.compile(
    r"(?:(?:\+?91[\s.-]?)?[6-9]\d{9}\b|(?:\+?\d{1,3}[\s.-]?)?\(?\d{3,5}\)?[\s.-]?\d{3,5}[\s.-]?\d{3,5}\b)"
)

# Popular Indian UPI handles and payment address patterns
RE_UPI = re.compile(
    r"\b[A-Za-z0-9.\-_]{2,256}@(okhdfcbank|okhdfc|okaxis|oksbi|okicici|paytm|upi|ybl|ibl|axl|apl|fbl|idfcbank|kotak)\b",
    re.IGNORECASE
)

# Common Indian Identifiers (Aadhaar 12-digit, PAN 10-char alphanumeric, OTP codes)
RE_AADHAAR = re.compile(r"\b[2-9]\d{3}[\s-]?\d{4}[\s-]?\d{4}\b")
RE_PAN = re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]{1}\b")
RE_OTP = re.compile(r"(?i)\b(?:otp|one[- ]time[- ]password|verification[- ]code)[:\s]*\d{4,6}\b")


def redact_text(text: str) -> str:
    """
    Redacts sensitive identifiers and replaces them with standardized tokens:
    - Emails -> [EMAIL]
    - Phone numbers -> [PHONE]
    - Web links/URLs -> [URL]
    - UPI IDs / payment references -> [UPI_ID]
    - Indian personal identifiers (Aadhaar, PAN, OTPs) -> [PERSONAL_IDENTIFIER]

    NOTE: Raw input text and redacted text must NEVER be logged by default.
    """
    if not isinstance(text, str):
        return ""

    # Redact UPI first to avoid handle getting split by email regex
    cleaned = RE_UPI.sub("[UPI_ID]", text)
    cleaned = RE_EMAIL.sub("[EMAIL]", cleaned)
    cleaned = RE_URL.sub("[URL]", cleaned)
    cleaned = RE_PHONE.sub("[PHONE]", cleaned)
    cleaned = RE_AADHAAR.sub("[PERSONAL_IDENTIFIER]", cleaned)
    cleaned = RE_PAN.sub("[PERSONAL_IDENTIFIER]", cleaned)
    cleaned = RE_OTP.sub("[PERSONAL_IDENTIFIER]", cleaned)

    return cleaned
