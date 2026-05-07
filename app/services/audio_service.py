import whisper
import tempfile
import os
from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)

class AudioService:
    def __init__(self):
        # tiny/base is fast enough for demos, bump to 'small' if accuracy matters
        logger.info(f"Loading Whisper model ({settings.WHISPER_MODEL_NAME})...")
        self.model = whisper.load_model(settings.WHISPER_MODEL_NAME)
        
    def transcribe(self, audio_bytes: bytes, suffix: str) -> str:
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name
                
            result = self.model.transcribe(tmp_path)
            
            duration_str = ""
            segments = result.get("segments", [])
            if segments:
                final_end = segments[-1].get("end", 0)
                mins = int(final_end // 60)
                secs = int(final_end % 60)
                duration_str = f"\n\n[Audio Duration: {mins}m {secs}s]"
            
            return result["text"].strip() + duration_str
        except Exception as e:
            logger.error(f"Error during audio transcription: {e}")
            raise Exception("Failed to transcribe audio.")
        finally:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
