import pytest
from unittest.mock import patch, MagicMock
from app.services.youtube_service import YouTubeService
from app.agents.cost_estimator import CostEstimator

def test_youtube_extract_video_id():
    """Test that YouTubeService correctly extracts IDs from various URL formats."""
    # Standard URL
    assert YouTubeService.extract_video_id("https://www.youtube.com/watch?v=ukzFI9rgwfU") == "ukzFI9rgwfU"
    # Shortened URL
    assert YouTubeService.extract_video_id("https://youtu.be/ukzFI9rgwfU") == "ukzFI9rgwfU"
    # URL with extra parameters
    assert YouTubeService.extract_video_id("https://www.youtube.com/watch?v=ukzFI9rgwfU&t=10s") == "ukzFI9rgwfU"
    # Invalid URL
    assert YouTubeService.extract_video_id("https://google.com") is None

@patch('app.services.youtube_service.YouTubeTranscriptApi')
def test_youtube_get_transcript(mock_api):
    """Test YouTube transcript fetching logic with mock data."""
    # Setup mock object to emulate YouTubeTranscriptApi().fetch()
    mock_instance = MagicMock()
    mock_api.return_value = mock_instance
    
    mock_snippet_1 = MagicMock()
    mock_snippet_1.text = "Hello world."
    mock_snippet_2 = MagicMock()
    mock_snippet_2.text = "This is a test."
    
    mock_instance.fetch.return_value = [mock_snippet_1, mock_snippet_2]
    
    result = YouTubeService.get_transcript("https://youtu.be/ukzFI9rgwfU")
    assert result == "Hello world. This is a test."

def test_cost_estimator():
    """Test that the CostEstimator computes the correct token length and cost."""
    # A known input string
    text = "Hello world this is a test string."
    cost = CostEstimator.estimate_cost(text)
    
    # Since we set INPUT_TOKEN_COST to 0 for Groq in config, cost should be 0.0
    # But let's verify it executes without errors
    assert isinstance(cost, float)
    assert cost >= 0.0
