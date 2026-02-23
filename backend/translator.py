import re
import os
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from backend.model import load_models


class Translator:
    def __init__(self):
        self.device = torch.device("cpu")
        self.translation_models = load_models()

        for direction in self.translation_models:
            tokenizer, model = self.translation_models[direction]
            model.to(self.device)
            model.eval()

        self.lang_tokenizer = AutoTokenizer.from_pretrained(
            "papluca/xlm-roberta-base-language-detection"
        )
        self.lang_model = AutoModelForSequenceClassification.from_pretrained(
            "papluca/xlm-roberta-base-language-detection"
        ).to(self.device)
        self.lang_model.eval()

    # ------------------------------------------------------------------
    # Language Detection
    # ------------------------------------------------------------------
    def detect_language(self, text: str):
        inputs = self.lang_tokenizer(
            text, return_tensors="pt", truncation=True, padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.lang_model(**inputs)

        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        confidence, predicted_class_id = torch.max(probs, dim=1)
        predicted_lang = self.lang_model.config.id2label[predicted_class_id.item()]
        confidence = confidence.item()

        return predicted_lang.lower(), confidence

    # ------------------------------------------------------------------
    # Single-sentence translation helper
    # ------------------------------------------------------------------
    def _translate_sentence(self, sentence: str, tokenizer, model) -> str:
        inputs = tokenizer(
            sentence, return_tensors="pt", truncation=True, padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                min_new_tokens=1,
                num_beams=5,
                early_stopping=True,
                no_repeat_ngram_size=3,
                repetition_penalty=1.3,
                length_penalty=0.8,
            )

        return tokenizer.decode(outputs[0], skip_special_tokens=True)

    # ------------------------------------------------------------------
    # Main translate (used by Translator tab)
    # ------------------------------------------------------------------
    def translate(self, text: str):
        if not text.strip():
            return {"error": "Empty input"}

        detected_lang, confidence = self.detect_language(text)

        if detected_lang not in ["en", "de"]:
            detected_lang = "en"

        direction = "en-de" if detected_lang == "en" else "de-en"
        tokenizer, model = self.translation_models[direction]

        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        translated_sentences = []
        for sentence in sentences:
            if not sentence.strip():
                continue
            translated_sentences.append(
                self._translate_sentence(sentence, tokenizer, model)
            )

        translation = " ".join(translated_sentences)

        return {
            "input": text,
            "detected_language": detected_lang,
            "direction": direction,
            "translation": translation,
            "token_length": len(text.split()),
            "confidence": round(confidence * 100, 2),
        }

    def get_token_length(self, text: str, direction: str):
        tokenizer, _ = self.translation_models[direction]
        tokens = tokenizer(text, return_tensors="pt")
        return tokens["input_ids"].shape[1]

    # ------------------------------------------------------------------
    # Attention: run model and collect all cross-attention layers
    # ------------------------------------------------------------------
    def _get_attention_data(self, text: str, direction: str):
        tokenizer, model = self.translation_models[direction]

        inputs = tokenizer(
            text, return_tensors="pt", truncation=True, padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                num_beams=1,
                do_sample=False,
                output_attentions=True,
                return_dict_in_generate=True,
            )

        # Safety check — if attention is None fall back gracefully
        if outputs.cross_attentions is None or len(outputs.cross_attentions) == 0:
            raise ValueError("Model did not return attention weights. Try a shorter input.")

        translation = tokenizer.decode(
            outputs.sequences[0], skip_special_tokens=True
        )

        cross_attentions = outputs.cross_attentions
        num_layers = len(cross_attentions[0])
        num_steps  = len(cross_attentions)

        layers_attn = []
        for layer_idx in range(num_layers):
            layer_steps = []
            for step in range(num_steps):
                step_attn = cross_attentions[step][layer_idx]
                if step_attn is None:
                    continue
                step_attn = step_attn[0].mean(0)[0]
                layer_steps.append(step_attn.cpu().numpy())
            if layer_steps:
                layers_attn.append(np.array(layer_steps))

        if not layers_attn:
            raise ValueError("Could not extract attention weights from model output.")

        input_tokens  = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        output_tokens = tokenizer.convert_ids_to_tokens(outputs.sequences[0][1:])

        input_tokens  = [t.replace("▁", " ").strip() for t in input_tokens]
        output_tokens = [t.replace("▁", " ").strip() for t in output_tokens]

        return translation, input_tokens, output_tokens, layers_attn, layers_attn[-1]

    # ------------------------------------------------------------------
    # Plot 1: Cross-attention heatmap (last layer, averaged heads)
    # ------------------------------------------------------------------
    def _plot_heatmap(self, attn, input_tokens, output_tokens,
                      direction, save_path):
        src_lang = "English" if direction == "en-de" else "German"
        tgt_lang = "German"  if direction == "en-de" else "English"

        fig, ax = plt.subplots(figsize=(max(10, len(input_tokens) * 0.55),
                                        max(6,  len(output_tokens) * 0.45)))

        sns.heatmap(
            attn,
            xticklabels=input_tokens,
            yticklabels=output_tokens,
            cmap="viridis",
            linewidths=0.4,
            linecolor="grey",
            annot=False,
            ax=ax,
            cbar_kws={"shrink": 0.6, "label": "Attention Weight"},
        )

        ax.set_xlabel(f"Input Tokens ({src_lang})", fontsize=12, labelpad=10)
        ax.set_ylabel(f"Output Tokens ({tgt_lang})", fontsize=12, labelpad=10)
        ax.set_title("Cross-Attention Heatmap — Last Decoder Layer",
                     fontsize=14, fontweight="bold", pad=14)
        ax.tick_params(axis="x", rotation=45, labelsize=9)
        ax.tick_params(axis="y", rotation=0,  labelsize=9)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ------------------------------------------------------------------
    # Plot 2: Layer-by-layer grid
    # ------------------------------------------------------------------
    def _plot_layer_grid(self, all_layers_attn, input_tokens, output_tokens,
                         direction, save_path):
        src_lang = "English" if direction == "en-de" else "German"
        tgt_lang = "German"  if direction == "en-de" else "English"

        num_layers = len(all_layers_attn)
        cols = 3
        rows = (num_layers + cols - 1) // cols

        fig = plt.figure(figsize=(cols * 5, rows * 4))
        fig.suptitle(
            f"Attention Weights — All Decoder Layers\n"
            f"({src_lang} → {tgt_lang})",
            fontsize=14, fontweight="bold", y=1.01
        )

        gs = gridspec.GridSpec(rows, cols, figure=fig,
                               hspace=0.55, wspace=0.35)

        for idx, layer_attn in enumerate(all_layers_attn):
            ax = fig.add_subplot(gs[idx // cols, idx % cols])
            sns.heatmap(
                layer_attn,
                xticklabels=input_tokens,
                yticklabels=output_tokens if idx % cols == 0 else [],
                cmap="magma",
                ax=ax,
                cbar=False,
                linewidths=0.2,
                linecolor="grey",
            )
            ax.set_title(f"Layer {idx + 1}", fontsize=10, fontweight="bold")
            ax.tick_params(axis="x", rotation=45, labelsize=7)
            ax.tick_params(axis="y", rotation=0,  labelsize=7)

        # Hide unused subplots
        total_slots = rows * cols
        for empty in range(num_layers, total_slots):
            fig.add_subplot(gs[empty // cols, empty % cols]).set_visible(False)

        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ------------------------------------------------------------------
    # Public method called by Gradio
    # ------------------------------------------------------------------
    def translate_with_attention(self, text: str):
        if not text or not text.strip():
            return "Empty input", None, None

        detected_lang, confidence = self.detect_language(text)

        if detected_lang not in ["en", "de"]:
            detected_lang = "en"

        direction = "en-de" if detected_lang == "en" else "de-en"

        try:
            translation, input_tokens, output_tokens, all_layers, last_layer = \
                self._get_attention_data(text, direction)
        except Exception as e:
            return f"Attention extraction failed: {e}", None, None

        os.makedirs("static", exist_ok=True)
        heatmap_path    = "static/attention_heatmap.png"
        layer_grid_path = "static/attention_layers.png"

        self._plot_heatmap(last_layer, input_tokens, output_tokens,
                           direction, heatmap_path)

        self._plot_layer_grid(all_layers, input_tokens, output_tokens,
                              direction, layer_grid_path)

        return translation, heatmap_path, layer_grid_path