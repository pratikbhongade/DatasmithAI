import re
from youtube_transcript_api import YouTubeTranscriptApi
from app.core.logger import get_logger

logger = get_logger(__name__)

class YouTubeService:
    @staticmethod
    def extract_video_id(url: str) -> str:
        pattern = r'(?:https?:\/\/)?(?:www\.)?(?:youtube\.com\/(?:[^\/\n\s]+\/\S+\/|(?:v|e(?:mbed)?)\/|\S*?[?&]v=)|youtu\.be\/)([a-zA-Z0-9_-]{11})'
        match = re.search(pattern, url)
        if match:
            return match.group(1)
        return None

    @staticmethod
    def get_transcript(url: str) -> str:
        video_id = YouTubeService.extract_video_id(url)
        if not video_id:
            logger.warning(f"Invalid YouTube URL provided: {url}")
            return None
        
        try:
            transcript = YouTubeTranscriptApi().fetch(video_id)
            # Combine transcript text
            full_text = " ".join([entry.text for entry in transcript])
            return full_text
        except Exception as e:
            logger.error(f"Failed to fetch YouTube transcript for {video_id}: {e}")
            return f"[Failed to fetch transcript. Error: {str(e)}]"
