from transformers import MarianMTModel, MarianTokenizer
import torch

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

EN_DE_MODEL_PATH = "./models/marian_finetuned"
DE_EN_MODEL_NAME = "Helsinki-NLP/opus-mt-de-en"

def load_models():

    en_de_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")
    en_de_model = MarianMTModel.from_pretrained(EN_DE_MODEL_PATH).to(DEVICE)
    en_de_model.eval()

    de_en_tokenizer = MarianTokenizer.from_pretrained(DE_EN_MODEL_NAME)
    de_en_model = MarianMTModel.from_pretrained(DE_EN_MODEL_NAME).to(DEVICE)
    de_en_model.eval()

    return {
        "en-de": (en_de_tokenizer, en_de_model),
        "de-en": (de_en_tokenizer, de_en_model),
    }
