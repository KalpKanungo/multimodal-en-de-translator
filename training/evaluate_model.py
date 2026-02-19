import torch
import evaluate
from transformers import MarianMTModel, MarianTokenizer
from prepare_dataset import load_and_prepare_dataset

MODEL_NAME = "Helsinki-NLP/opus-mt-en-de"
FINE_TUNED_PATH = "./models/marian_finetuned"
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

bleu = evaluate.load("sacrebleu")


def generate_translations(model, tokenizer, sentences, batch_size=16):
    translations = []

    model.eval()
    model.to(DEVICE)

    for i in range(0, len(sentences), batch_size):
        batch_sentences = sentences[i:i+batch_size]

        inputs = tokenizer(
            batch_sentences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=64
        ).to(DEVICE)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=64)

        decoded = tokenizer.batch_decode(
            outputs,
            skip_special_tokens=True
        )

        translations.extend(decoded)

    return translations



def evaluate_model(model_path=None):

    print("Loading test dataset...")
    dataset = load_and_prepare_dataset(sample_size=20000)
    test_data = dataset["test"]

    references = [example["de"] for example in test_data]
    sources = [example["en"] for example in test_data]

    print("Loading tokenizer...")
    tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)

    if model_path:
        print("Loading fine-tuned model...")
        model = MarianMTModel.from_pretrained(model_path)
    else:
        print("Loading pretrained model...")
        model = MarianMTModel.from_pretrained(MODEL_NAME)

    print("Generating translations...")
    predictions = generate_translations(model, tokenizer, sources)

    print("Computing BLEU...")
    bleu_score = bleu.compute(
        predictions=predictions,
        references=[[ref] for ref in references]
    )

    return bleu_score["score"]


if __name__ == "__main__":

    print("Evaluating pretrained model...")
    pretrained_bleu = evaluate_model(model_path=None)

    print("\nEvaluating fine-tuned model...")
    finetuned_bleu = evaluate_model(model_path=FINE_TUNED_PATH)

    print("\nRESULTS:")
    print(f"Pretrained BLEU: {pretrained_bleu:.2f}")
    print(f"Fine-tuned BLEU: {finetuned_bleu:.2f}")
    print(f"Improvement: {finetuned_bleu - pretrained_bleu:.2f}")
