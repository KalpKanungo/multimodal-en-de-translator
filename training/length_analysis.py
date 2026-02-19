import torch
import evaluate
import matplotlib.pyplot as plt
from collections import defaultdict
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
        batch = sentences[i:i+batch_size]

        inputs = tokenizer(
            batch,
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


def sentence_bleu(pred, ref):
    result = bleu.compute(
        predictions=[pred],
        references=[[ref]]
    )
    return result["score"]


def bucket_length(length):
    if length <= 5:
        return "1-5"
    elif length <= 10:
        return "6-10"
    elif length <= 15:
        return "11-15"
    elif length <= 20:
        return "16-20"
    elif length <= 25:
        return "21-25"
    elif length <= 30:
        return "26-30"
    else:
        return "30+"


def evaluate_length_analysis():

    dataset = load_and_prepare_dataset(sample_size=20000)
    test_data = dataset["test"]

    sources = [ex["en"] for ex in test_data]
    references = [ex["de"] for ex in test_data]

    tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)

    pretrained = MarianMTModel.from_pretrained(MODEL_NAME)
    finetuned = MarianMTModel.from_pretrained(FINE_TUNED_PATH)

    print("Generating pretrained predictions...")
    preds_pre = generate_translations(pretrained, tokenizer, sources)

    print("Generating finetuned predictions...")
    preds_fine = generate_translations(finetuned, tokenizer, sources)

    bucket_scores_pre = defaultdict(list)
    bucket_scores_fine = defaultdict(list)

    for src, ref, p_pre, p_fine in zip(sources, references, preds_pre, preds_fine):

        length = len(src.split())
        bucket = bucket_length(length)

        bleu_pre = sentence_bleu(p_pre, ref)
        bleu_fine = sentence_bleu(p_fine, ref)

        bucket_scores_pre[bucket].append(bleu_pre)
        bucket_scores_fine[bucket].append(bleu_fine)

    bucket_order = ["1-5", "6-10", "11-15", "16-20", "21-25", "26-30", "30+"]
    buckets = [b for b in bucket_order if b in bucket_scores_pre]


    avg_pre = [sum(bucket_scores_pre[b])/len(bucket_scores_pre[b]) for b in buckets]
    avg_fine = [sum(bucket_scores_fine[b])/len(bucket_scores_fine[b]) for b in buckets]

    plt.figure(figsize=(10,6))
    plt.plot(buckets, avg_pre, marker='o', label="Pretrained")
    plt.plot(buckets, avg_fine, marker='o', label="Fine-tuned")

    plt.xlabel("Sentence Length (words)")
    plt.ylabel("Average Sentence BLEU")
    plt.title("Sentence Length vs Translation Quality")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    evaluate_length_analysis()
