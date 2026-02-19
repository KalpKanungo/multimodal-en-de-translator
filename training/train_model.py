import torch
from transformers import (
    MarianMTModel,
    MarianTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForSeq2Seq
)

from prepare_dataset import load_and_prepare_dataset
from tokenize_dataset import tokenize_dataset


MODEL_NAME = "Helsinki-NLP/opus-mt-en-de"
OUTPUT_DIR = "./models/marian_finetuned_20k_2ep"


def main():

    # ---------------------------------------------------
    # Device setup (MPS for Mac, else CPU)
    # ---------------------------------------------------
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon GPU)")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # ---------------------------------------------------
    # Load and prepare dataset (20k samples)
    # ---------------------------------------------------
    print("Preparing dataset...")
    dataset = load_and_prepare_dataset(sample_size=20000)

    print("Tokenizing dataset...")
    tokenized_datasets = tokenize_dataset(dataset)

    # ---------------------------------------------------
    # Load model & tokenizer
    # ---------------------------------------------------
    print("Loading model...")
    tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
    model = MarianMTModel.from_pretrained(MODEL_NAME)

    model.to(device)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model
    )

    # ---------------------------------------------------
    # Training configuration
    # ---------------------------------------------------
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=2,
        learning_rate=3e-5,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        logging_steps=100,
        report_to="none",
        fp16=False  # IMPORTANT for MPS
    )

    # ---------------------------------------------------
    # Trainer
    # ---------------------------------------------------
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
    )

    # ---------------------------------------------------
    # Training
    # ---------------------------------------------------
    print("Starting training...")
    trainer.train()

    # ---------------------------------------------------
    # Save final model
    # ---------------------------------------------------
    print("Saving model...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("Training complete.")
    print(f"Model saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
