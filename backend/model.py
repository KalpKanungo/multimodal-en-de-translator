from transformers import MarianMTModel, MarianTokenizer


from transformers import MarianMTModel, MarianTokenizer

def load_models():
    models = {}

    tokenizer_en_de = MarianTokenizer.from_pretrained("kalpkanungo/marian-en-de")
    model_en_de = MarianMTModel.from_pretrained("kalpkanungo/marian-en-de")
    models["en-de"] = (tokenizer_en_de, model_en_de)

    tokenizer_de_en = MarianTokenizer.from_pretrained("kalpkanungo/marian-de-en")
    model_de_en = MarianMTModel.from_pretrained("kalpkanungo/marian-de-en")
    models["de-en"] = (tokenizer_de_en, model_de_en)

    return models
