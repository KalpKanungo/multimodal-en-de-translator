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
OUTPUT_DIR = "./models/marian_finetuned"


def main():

    print("Preparing dataset...")
    dataset = load_and_prepare_dataset(sample_size=50000)
    tokenized_datasets = tokenize_dataset(dataset)

    print("Loading model...")
    model = MarianMTModel.from_pretrained(MODEL_NAME)
    tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        learning_rate=3e-5,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        fp16=True,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator
    )

    print("Starting training...")
    trainer.train()

    print("Saving model...")
    trainer.save_model(OUTPUT_DIR)

    print("Training complete.")


if __name__ == "__main__":
    main()
