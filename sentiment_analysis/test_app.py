from app import clean_text, health


def test_clean_text():
    text = "Amazing App! Visit https://example.com"
    result = clean_text(text)

    assert "amazing app" in result
    assert "https" not in result


def test_health():
    response = health()

    assert response["status"] == "healthy"
    
