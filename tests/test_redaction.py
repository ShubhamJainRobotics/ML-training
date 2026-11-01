"""Unit tests for PII and sensitive data redaction."""

from app.redaction import redact_text


def test_redact_emails():
    text = "Send your resume to hr.recruiter@techjobs-corp.in or contact@quickhire.com immediately."
    redacted = redact_text(text)
    assert "[EMAIL]" in redacted
    assert "hr.recruiter@techjobs-corp.in" not in redacted
    assert "contact@quickhire.com" not in redacted


def test_redact_phone_numbers():
    text = "Call HR coordinator on +91 9876543210 or WhatsApp 9123456789 for interview slot."
    redacted = redact_text(text)
    assert "[PHONE]" in redacted
    assert "9876543210" not in redacted
    assert "9123456789" not in redacted


def test_redact_urls():
    text = "Visit our hiring portal at https://jobs-careers-verification.in/apply or http://tinyurl.com/hr-test for details."
    redacted = redact_text(text)
    assert "[URL]" in redacted
    assert "https://jobs-careers-verification.in/apply" not in redacted
    assert "http://tinyurl.com/hr-test" not in redacted


def test_redact_upi_ids():
    text = "Transfer registration fee of Rs 499 to payment.desk@okaxis or hrjobs@paytm to confirm."
    redacted = redact_text(text)
    assert "[UPI_ID]" in redacted
    assert "payment.desk@okaxis" not in redacted
    assert "hrjobs@paytm" not in redacted


def test_redact_aadhaar_and_pan():
    text = "Send copy of Aadhaar card 4589 1234 5678 and PAN number ABCDE1234F for salary KYC."
    redacted = redact_text(text)
    assert "[PERSONAL_IDENTIFIER]" in redacted
    assert "4589 1234 5678" not in redacted
    assert "ABCDE1234F" not in redacted


def test_redact_otp_references():
    text = "Please share the OTP 482910 sent to your phone for payroll verification."
    redacted = redact_text(text)
    assert "[PERSONAL_IDENTIFIER]" in redacted
    assert "482910" not in redacted


def test_redaction_preserves_surrounding_context():
    text = "Software Engineer position at Bengaluru. Required: Python, Django, SQL. Zero registration fee."
    redacted = redact_text(text)
    assert "Software Engineer" in redacted
    assert "Python, Django, SQL" in redacted
    assert "Zero registration fee" in redacted


def test_redaction_handles_empty_input():
    assert redact_text("") == ""
    assert redact_text(None) == ""

