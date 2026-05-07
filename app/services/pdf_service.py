import fitz  # PyMuPDF
from app.services.ocr_service import OCRService
from app.core.config import settings
from app.core.logger import get_logger

logger = get_logger(__name__)

class PDFService:
    @staticmethod
    def extract_text(pdf_bytes: bytes) -> str:
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            full_text = ""
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text()
                
                # if barely any text came back, probably scanned - run OCR instead
                if len(text.strip()) < settings.PDF_OCR_THRESHOLD:
                    logger.info(f"Low text content on page {page_num}, falling back to OCR.")
                    pix = page.get_pixmap()
                    img_bytes = pix.tobytes("png")
                    text = OCRService.extract_text(img_bytes)
                
                full_text += text + "\n"
            return full_text.strip()
        except Exception as e:
            logger.error(f"Error during PDF extraction: {e}")
            raise Exception("Failed to extract text from PDF.")
