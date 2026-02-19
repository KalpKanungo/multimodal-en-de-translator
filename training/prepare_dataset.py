import random
from datasets import load_dataset, DatasetDict
from sklearn.model_selection import train_test_split

SEED = 42
random.seed(SEED)

def load_and_prepare_dataset(sample_size=20000):
    print("Loading OPUS Books dataset (EN-DE)...")

    dataset = load_dataset("opus_books", "de-en")


    full_data = dataset["train"]

    print(f"Original dataset size: {len(full_data)}")

    
    full_data = full_data.shuffle(seed=SEED)

    
    if sample_size > len(full_data):
        raise ValueError("Requested sample size larger than dataset!")

    sampled_data = full_data.select(range(sample_size))

    print(f"Sampled dataset size: {len(sampled_data)}")

    
    en_sentences = []
    de_sentences = []

    for example in sampled_data:
        en_text = example["translation"]["en"].strip()
        de_text = example["translation"]["de"].strip()

        
        if len(en_text) > 0 and len(de_text) > 0:
            en_sentences.append(en_text)
            de_sentences.append(de_text)

    print(f"Cleaned sentence pairs: {len(en_sentences)}")

    data_pairs = [
        {"en": en, "de": de}
        for en, de in zip(en_sentences, de_sentences)
    ]

    
    train_data, temp_data = train_test_split(
        data_pairs,
        test_size=0.2,
        random_state=SEED
    )

    val_data, test_data = train_test_split(
        temp_data,
        test_size=0.5,
        random_state=SEED
    )

    dataset_dict = DatasetDict({
        "train": dataset["train"].from_list(train_data),
        "validation": dataset["train"].from_list(val_data),
        "test": dataset["train"].from_list(test_data)
    })

    print("Dataset split complete:")
    print(f"Train: {len(dataset_dict['train'])}")
    print(f"Validation: {len(dataset_dict['validation'])}")
    print(f"Test: {len(dataset_dict['test'])}")

    return dataset_dict


if __name__ == "__main__":
    dataset = load_and_prepare_dataset(sample_size=20000)
