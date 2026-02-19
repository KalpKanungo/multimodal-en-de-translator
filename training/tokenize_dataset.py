from datasets import DatasetDict
from transformers import MarianTokenizer

MODEL_NAME = "Helsinki-NLP/opus-mt-en-de"
MAX_SOURCE_LENGTH = 64
MAX_TARGET_LENGTH = 64

def tokenize_dataset(dataset_dict: DatasetDict):
    tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)

    def preprocess_function(examples):
        inputs = examples["en"]
        targets = examples["de"]

        model_inputs = tokenizer(
            inputs,
            max_length=MAX_SOURCE_LENGTH,
            truncation=True
        )

        labels = tokenizer(
            text_target=targets,
            max_length=MAX_TARGET_LENGTH,
            truncation=True
        )

        model_inputs["labels"] = labels["input_ids"]

        return model_inputs


    tokenized_datasets = dataset_dict.map(
        preprocess_function,
        batched=True,
        remove_columns=["en", "de"]
    )

    return tokenized_datasets


if __name__ == "__main__":
    from prepare_dataset import load_and_prepare_dataset

    dataset = load_and_prepare_dataset(sample_size=20000)
    tokenized = tokenize_dataset(dataset)

    print(tokenized)
