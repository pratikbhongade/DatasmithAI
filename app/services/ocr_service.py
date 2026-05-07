import pytesseract
from PIL import Image
import io
from app.core.logger import get_logger

logger = get_logger(__name__)

class OCRService:
    @staticmethod
    def extract_text(image_bytes: bytes) -> str:
        try:
            image = Image.open(io.BytesIO(image_bytes))
            # Use image_to_data to get confidences
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
            
            text_blocks = []
            confidences = []
            
            for i in range(len(data['text'])):
                word = data['text'][i].strip()
                try:
                    conf = int(data['conf'][i])
                except (ValueError, TypeError):
                    continue
                if word and conf > -1:
                    text_blocks.append(word)
                    confidences.append(conf)
            
            # print(f"confidences found: {len(confidences)}")
            text = " ".join(text_blocks)
            avg_conf = sum(confidences) / len(confidences) if confidences else 0
            
            result = text.strip()
            
            warning = ""
            if avg_conf < 80.0 and len(result) > 0:
                warning = "⚠️ WARNING: OCR confidence is low. Some extracted text may be inaccurate.\n\n"
                
            if result:
                result = warning + result + f"\n\n[OCR Average Confidence: {avg_conf:.2f}%]"
            
            return result
        except Exception as e:
            logger.error(f"Error during OCR extraction: {e}")
            raise Exception("Failed to extract text from image.")
