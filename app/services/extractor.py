from app.models.schemas import UserInput, ExtractionResult
from app.services.ocr_service import OCRService
from app.services.pdf_service import PDFService
from app.services.audio_service import AudioService
from app.services.youtube_service import YouTubeService
from app.core.logger import get_logger
import re

logger = get_logger(__name__)

class ExtractorRouter:
    def __init__(self):
        # Initialize AudioService which loads the Whisper model
        self.audio_service = AudioService()

    def extract(self, file_bytes: bytes, file_type: str, file_name: str, text_prompt: str) -> ExtractionResult:
        content = ""
        source_type = "text"
        
        if text_prompt:
            yt_id = YouTubeService.extract_video_id(text_prompt)
            if yt_id:
                logger.info(f"Detected YouTube URL. Fetching transcript...")
                yt_content = YouTubeService.get_transcript(text_prompt)
                if yt_content:
                    content += yt_content + "\n"
                    source_type = "youtube"

        if file_bytes and file_type:
            logger.info(f"Processing file of type: {file_type}")
            if file_type.startswith("image/"):
                content += OCRService.extract_text(file_bytes)
                source_type = "image"
            elif file_type == "application/pdf":
                content += PDFService.extract_text(file_bytes)
                source_type = "pdf"
            elif file_type.startswith("audio/") or file_type.startswith("video/") or file_name.endswith(('.mp3', '.wav', '.m4a', '.mp4', '.mov', '.mkv')):
                suffix = "." + file_name.split('.')[-1] if '.' in file_name else ".mp3"
                content += self.audio_service.transcribe(file_bytes, suffix=suffix)
                source_type = "video" if file_type.startswith("video/") or file_name.endswith(('.mp4', '.mov', '.mkv')) else "audio"
            else:
                logger.warning(f"Unsupported file type: {file_type}")
                content += f"[Unsupported file type: {file_type}]"

        if text_prompt:
            content = f"User Prompt/Instruction:\n{text_prompt}\n\nExtracted Content:\n{content}"
            # If there was no file and no youtube video, it's purely text
            if not file_bytes and not YouTubeService.extract_video_id(text_prompt):
                source_type = "text"

        return ExtractionResult(
            content=content.strip(),
            source_type=source_type
        )
