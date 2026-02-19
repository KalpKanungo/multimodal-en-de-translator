import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from backend.model import load_models


class Translator:
    def __init__(self):

        self.device = torch.device("cpu")
        self.translation_models = load_models()

        for direction in self.translation_models:
            tokenizer, model = self.translation_models[direction]
            model.to(self.device)
            model.eval()

        self.lang_tokenizer = AutoTokenizer.from_pretrained(
            "papluca/xlm-roberta-base-language-detection"
        )
        self.lang_model = AutoModelForSequenceClassification.from_pretrained(
            "papluca/xlm-roberta-base-language-detection"
        ).to(self.device)

        self.lang_model.eval()

    def detect_language(self, text: str):

        inputs = self.lang_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.lang_model(**inputs)

        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        confidence, predicted_class_id = torch.max(probs, dim=1)

        predicted_lang = self.lang_model.config.id2label[predicted_class_id.item()]
        confidence = confidence.item()

        if confidence < 0.50:
            return "unknown"

        return predicted_lang.lower()

    def translate(self, text):

        if not text.strip():
            return {"error": "Empty input"}

        detected_lang = self.detect_language(text)

        if detected_lang not in ["en", "de"]:
            detected_lang = "en"

        direction = "en-de" if detected_lang == "en" else "de-en"

        tokenizer, model = self.translation_models[direction]

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True
        )

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                        **inputs,
                        max_new_tokens=80,
                        num_beams=5,
                        length_penalty=0.8,
                        early_stopping=False
                    )

        translation = tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        token_length = len(text.split())

        return {
            "input": text,
            "detected_language": detected_lang,
            "direction": direction,
            "translation": translation,
            "token_length": token_length
        }

    def get_token_length(self, text: str, direction: str):
        tokenizer, _ = self.translation_models[direction]
        tokens = tokenizer(text, return_tensors="pt")
        return tokens["input_ids"].shape[1]
