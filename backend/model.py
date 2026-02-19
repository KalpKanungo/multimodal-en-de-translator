from transformers import MarianMTModel, MarianTokenizer


def load_models():

    models = {}


    en_de_path = "./models/marian_finetuned_20k_2ep"

    tokenizer_en_de = MarianTokenizer.from_pretrained(en_de_path)
    model_en_de = MarianMTModel.from_pretrained(en_de_path)

    models["en-de"] = (tokenizer_en_de, model_en_de)

    de_en_name = "Helsinki-NLP/opus-mt-de-en"

    tokenizer_de_en = MarianTokenizer.from_pretrained(de_en_name)
    model_de_en = MarianMTModel.from_pretrained(de_en_name)

    models["de-en"] = (tokenizer_de_en, model_de_en)

    return models
