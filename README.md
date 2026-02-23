---
title: Multimodal English German Translator
emoji: 🌐
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 5.23.0
app_file: app.py
pinned: false
---

# 🌐 Multimodal English–German Translator

A bidirectional English↔German neural machine translation system with multimodal input support. Built as a full-stack NLP project combining fine-tuned transformer models, automatic speech recognition, OCR, text-to-speech, and attention visualization — all accessible through a clean Gradio web interface.

---

## Features

### Input Modalities
- **Text** — type directly in English or German
- **Speech** — record or upload audio, transcribed via OpenAI Whisper
- **Image** — upload a photo of printed text, a signboard, screenshot, or textbook page; text is extracted via EasyOCR

### Translation
- Automatic language detection — no need to specify the input language
- Bidirectional translation: English→German and German→English
- Sentence-level translation with hallucination prevention
- Confidence score displayed for every translation

### Output
- Translated text displayed instantly
- Text-to-speech audio output in the correct target language using Microsoft Edge TTS Neural voices (`en-US-JennyNeural` for English, `de-DE-KatjaNeural` for German)

### Attention Visualizer
- **Cross-attention heatmap** — see which input tokens the model attends to when generating each output token
- **Layer-by-layer attention grid** — visualize attention patterns across all decoder layers, showing how early layers attend broadly and later layers focus precisely

### Evaluation Dashboard
- Input your own reference translation and compare it against the model output
- **BLEU score** with quality label (Poor / Fair / Good / Excellent)
- **Language detection confidence score**

---

## Models

| Direction | Model | Type |
|-----------|-------|------|
| English → German | `kalpkanungo/marian-en-de` | Fine-tuned on OPUS Books |
| German → English | `kalpkanungo/marian-de-en` | Helsinki-NLP base model |

Both models are based on `Helsinki-NLP/opus-mt` (MarianMT architecture). The EN→DE model was fine-tuned on 20,000 sentence pairs from the OPUS Books dataset for 2 epochs using the HuggingFace Trainer API on Apple Silicon (MPS).

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Translation | MarianMT (Helsinki-NLP) |
| Language Detection | `papluca/xlm-roberta-base-language-detection` |
| Speech Recognition | OpenAI Whisper (`whisper-small`) |
| OCR | EasyOCR |
| Text-to-Speech | Microsoft Edge TTS |
| Attention Visualization | Matplotlib + Seaborn |
| Evaluation | NLTK BLEU Score |
| UI | Gradio |

---

## Project Structure

```
├── app.py                  # Main Gradio app
├── backend/
│   ├── model.py            # Model loading (HF Hub)
│   ├── translator.py       # Translation + attention extraction
│   ├── asr.py              # Whisper speech recognition
│   ├── tts.py              # Edge TTS speech synthesis
│   ├── ocr.py              # EasyOCR image text extraction
│   └── evaluator.py        # BLEU score evaluation
├── training/
│   ├── train_en_de.py      # Fine-tuning script EN→DE
│   ├── train_de_en.py      # Fine-tuning script DE→EN
│   ├── prepare_dataset.py  # Dataset loading and splitting
│   └── tokenize_dataset.py # Tokenization with direction support
└── requirements.txt
```

---

## Running Locally

```bash
# Clone the repo
git clone https://github.com/KalpKanungo/multimodal-en-de-translator.git
cd multimodal-en-de-translator

# Create and activate a conda environment
conda create -n translator python=3.10
conda activate translator

# Install dependencies
pip install -r requirements.txt

# Run the app
python app.py
```

---

## Example Translations

| Input | Output |
|-------|--------|
| Hello, my name is Kalp and I am twenty years old. | Hallo, mein Name ist Kalp und ich bin zwanzig Jahre alt. |
| Ich komme aus Indien und lebe in Mumbai. | I come from India and live in Mumbai. |

---

## Known Limitations

- Handwritten text OCR works partially — best results with clear, neat handwriting. Printed text works well.
- The attention visualizer uses greedy decoding (`num_beams=1`) which is required for attention extraction but may produce slightly lower quality translations than the main translator tab.
- TTS requires an internet connection as it uses Microsoft's Edge TTS API.

---

## Author

**Kalp Kanungo**  
[GitHub](https://github.com/KalpKanungo) · [Hugging Face](https://huggingface.co/kalpkanungo)