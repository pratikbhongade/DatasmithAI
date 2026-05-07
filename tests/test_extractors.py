import pytest
from app.services.extractor import ExtractorRouter
from app.services.youtube_service import YouTubeService

# Simple mock test for Youtube Service regex
def test_youtube_url_extraction():
    url1 = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    assert YouTubeService.extract_video_id(url1) == "dQw4w9WgXcQ"
    
    url2 = "https://youtu.be/dQw4w9WgXcQ?t=1"
    assert YouTubeService.extract_video_id(url2) == "dQw4w9WgXcQ"
    
    url3 = "not a youtube link"
    assert YouTubeService.extract_video_id(url3) is None

# Test ExtractorRouter routing logic (mocking actual extraction to avoid needing files)
def test_extractor_router_text_only(monkeypatch):
    router = ExtractorRouter()
    
    # Mocking audio service init
    monkeypatch.setattr("app.services.audio_service.AudioService.__init__", lambda x: None)
    
    res = router.extract(None, None, None, "Hello world")
    assert res.source_type == "text"
    assert "Hello world" in res.content

def test_extractor_router_image(monkeypatch):
    router = ExtractorRouter()
    
    def mock_extract_text(img_bytes):
        return "Mocked OCR text"
    
    monkeypatch.setattr("app.services.ocr_service.OCRService.extract_text", mock_extract_text)
    
    res = router.extract(b"fakebytes", "image/png", "test.png", "Explain this")
    assert res.source_type == "image"
    assert "Mocked OCR text" in res.content
    assert "Explain this" in res.content
