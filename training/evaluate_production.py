import os
import json
import torch
import evaluate
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import MarianMTModel, MarianTokenizer


MODEL_PATH = "./models/marian_finetuned_20k_2ep"
BASE_MODEL = "Helsinki-NLP/opus-mt-en-de"

OUTPUT_DIR = "./evaluation_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

bleu = evaluate.load("sacrebleu")


def load_model():
    tokenizer = MarianTokenizer.from_pretrained(BASE_MODEL)
    model = MarianMTModel.from_pretrained(MODEL_PATH).to(device)
    model.eval()
    return tokenizer, model


def translate_batch(tokenizer, model, texts, batch_size=16):
    all_predictions = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]

        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=60,
                num_beams=2   # reduced from 4 to save memory
            )

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        all_predictions.extend(decoded)

        # Free memory manually (important on MPS)
        del inputs
        del outputs
        torch.mps.empty_cache()

    return all_predictions



def compute_bleu(preds, refs):
    return bleu.compute(
        predictions=preds,
        references=[[r] for r in refs]
    )["score"]


def bucket_analysis(sources, predictions, references):
    buckets = {
        "0-5": [],
        "6-10": [],
        "11-20": [],
        "21-40": [],
        "40+": []
    }

    for src, pred, ref in zip(sources, predictions, references):
        length = len(src.split())

        if length <= 5:
            buckets["0-5"].append((pred, ref))
        elif length <= 10:
            buckets["6-10"].append((pred, ref))
        elif length <= 20:
            buckets["11-20"].append((pred, ref))
        elif length <= 40:
            buckets["21-40"].append((pred, ref))
        else:
            buckets["40+"].append((pred, ref))

    bucket_scores = {}

    for key, pairs in buckets.items():
        if len(pairs) == 0:
            bucket_scores[key] = 0
        else:
            preds = [p[0] for p in pairs]
            refs = [p[1] for p in pairs]
            bucket_scores[key] = compute_bleu(preds, refs)

    return bucket_scores


def plot_bucket_scores(bucket_scores):
    keys = list(bucket_scores.keys())
    values = [bucket_scores[k] for k in keys]

    plt.figure()
    plt.plot(keys, values)
    plt.xlabel("Sentence Length Bucket")
    plt.ylabel("BLEU Score")
    plt.title("BLEU vs Input Length")
    plt.xticks(rotation=45)
    plt.tight_layout()

    plot_path = os.path.join(OUTPUT_DIR, "length_vs_bleu.png")
    plt.savefig(plot_path)
    plt.close()

    return plot_path


def main():
    print("Loading dataset...")
    dataset = load_dataset("opus_books", "de-en", split="train[:2000]")

    sources = [x["translation"]["en"] for x in dataset]
    references = [x["translation"]["de"] for x in dataset]

    tokenizer, model = load_model()

    print("Generating translations...")
    predictions = translate_batch(tokenizer, model, sources)

    print("Computing overall BLEU...")
    overall_bleu = compute_bleu(predictions, references)

    print("Computing bucket BLEU...")
    bucket_scores = bucket_analysis(sources, predictions, references)

    print("Saving metrics...")
    results = {
        "overall_bleu": overall_bleu,
        "length_bucket_bleu": bucket_scores
    }

    with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w") as f:
        json.dump(results, f, indent=4)

    print("Generating plot...")
    plot_path = plot_bucket_scores(bucket_scores)

    print("\nEvaluation Complete")
    print("Overall BLEU:", overall_bleu)
    print("Saved to:", OUTPUT_DIR)
    print("Plot file:", plot_path)


if __name__ == "__main__":
    main()
