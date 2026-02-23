import easyocr
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import tempfile
import os
import subprocess


class OCR:
    def __init__(self):
        print("Loading OCR model (English + German)...")
        self.reader = easyocr.Reader(["en", "de"], gpu=False)
        print("OCR model loaded.")

    def _convert_to_png(self, image_path: str) -> str:
        """Convert any image format (including HEIC) to PNG using macOS sips."""
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp_path = tmp.name
        tmp.close()

        try:
            subprocess.run(
                ["sips", "-s", "format", "png", image_path, "--out", tmp_path],
                check=True,
                capture_output=True
            )
            return tmp_path
        except Exception as e:
            print(f"[OCR] sips conversion failed: {e}")
            return image_path

    def _preprocess_image(self, image_path: str) -> str:
        """Enhance image contrast and sharpness for better OCR accuracy."""
        img = Image.open(image_path).convert("L")  # grayscale

        # Increase contrast
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(2.0)

        # Sharpen
        img = img.filter(ImageFilter.SHARPEN)

        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        img.save(tmp.name)
        tmp.close()
        return tmp.name

    def extract_text(self, image_path: str) -> str:
        """
        Extract text from an image file.
        Handles HEIC, JPG, PNG and any format supported by sips.
        Applies preprocessing to improve accuracy on handwritten text.

        Args:
            image_path: Path to the image file

        Returns:
            Extracted text as a string, or empty string if nothing found.
        """
        if image_path is None:
            return ""

        converted_path = None
        preprocessed_path = None

        try:
            # Step 1: Convert to PNG (handles HEIC and other formats)
            converted_path = self._convert_to_png(image_path)

            # Step 2: Preprocess for better accuracy
            preprocessed_path = self._preprocess_image(converted_path)

            # Step 3: Run OCR
            results = self.reader.readtext(preprocessed_path)

            if not results:
                return ""

            # Lower threshold (0.2) for handwritten text
            lines = [
                text for (_, text, confidence) in results
                if confidence > 0.2 and text.strip()
            ]

            return " ".join(lines).strip()

        except Exception as e:
            print(f"[OCR] Error extracting text: {e}")
            return ""

        finally:
            # Always clean up temp files
            for path in [converted_path, preprocessed_path]:
                if path and path != image_path and os.path.exists(path):
                    try:
                        os.unlink(path)
                    except Exception:
                        pass