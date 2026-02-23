import easyocr
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import tempfile
import os

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass


class OCR:
    def __init__(self):
        print("Loading OCR model (English + German)...")
        self.reader = easyocr.Reader(["en", "de"], gpu=False)
        print("OCR model loaded.")

    def _convert_to_png(self, image_path: str) -> str:
        """Convert any image format to PNG using Pillow (works on all OS)."""
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp_path = tmp.name
        tmp.close()

        try:
            img = Image.open(image_path).convert("RGB")
            img.save(tmp_path)
            return tmp_path
        except Exception as e:
            print(f"[OCR] Conversion failed: {e}")
            return image_path

    def _preprocess_image(self, image_path: str) -> str:
        """Enhance image for better OCR accuracy."""
        img = Image.open(image_path).convert("L")
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(2.0)
        img = img.filter(ImageFilter.SHARPEN)

        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        img.save(tmp.name)
        tmp.close()
        return tmp.name

    def extract_text(self, image_path: str) -> str:
        if image_path is None:
            return ""

        converted_path = None
        preprocessed_path = None

        try:
            converted_path = self._convert_to_png(image_path)
            preprocessed_path = self._preprocess_image(converted_path)
            results = self.reader.readtext(preprocessed_path)

            if not results:
                return ""

            lines = [
                text for (_, text, confidence) in results
                if confidence > 0.2 and text.strip()
            ]
            return " ".join(lines).strip()

        except Exception as e:
            print(f"[OCR] Error extracting text: {e}")
            return ""

        finally:
            for path in [converted_path, preprocessed_path]:
                if path and path != image_path and os.path.exists(path):
                    try:
                        os.unlink(path)
                    except Exception:
                        pass