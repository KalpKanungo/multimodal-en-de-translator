from transformers import MarianMTModel, MarianTokenizer

# Push your fine-tuned EN->DE model
model = MarianMTModel.from_pretrained("/Users/kalpkanungo/Desktop/English_German/models/marian_finetuned_20k_2ep")
tokenizer = MarianTokenizer.from_pretrained("/Users/kalpkanungo/Desktop/English_German/models/marian_finetuned_20k_2ep")
model.push_to_hub("kalpkanungo/marian-en-de")
tokenizer.push_to_hub("kalpkanungo/marian-en-de")
print("EN->DE pushed successfully")

# Push base Helsinki DE->EN model (since you haven't fine-tuned it yet)
model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-de-en")
tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-de-en")
model.push_to_hub("kalpkanungo/marian-de-en")
tokenizer.push_to_hub("kalpkanungo/marian-de-en")
print("DE->EN pushed successfully")