import gradio as gr
from backend.translator import Translator
from backend.asr import ASR
from backend.tts import TTS
from backend.evaluator import Evaluator

translator  = Translator()
asr         = ASR()
tts_engine  = TTS()
evaluator   = Evaluator()


# ------------------------------------------------------------------
# Handler: Translator tab
# ------------------------------------------------------------------
def process_input(text, audio):
    if text and text.strip():
        input_text = text
    elif audio is not None:
        input_text = asr.transcribe(audio)
    else:
        return "", None

    result = translator.translate(input_text)

    if "error" in result:
        return result["error"], None

    translated_text = result["translation"]
    target_lang     = "de" if result["direction"] == "en-de" else "en"
    audio_file      = tts_engine.synthesize(translated_text, target_lang)

    return translated_text, audio_file


# ------------------------------------------------------------------
# Handler: Attention Visualizer tab
# ------------------------------------------------------------------
def process_with_attention(text):
    if not text or not text.strip():
        return "Please enter some text.", None, None

    translation, heatmap_path, layer_grid_path = \
        translator.translate_with_attention(text)

    return translation, heatmap_path, layer_grid_path


# ------------------------------------------------------------------
# Handler: Evaluation Dashboard tab
# ------------------------------------------------------------------
def process_evaluation(input_text, reference_text):
    """
    Translates input_text, then evaluates the output against
    the user-provided reference translation using BLEU + confidence.
    Resets every time (no session history).
    """
    if not input_text or not input_text.strip():
        return "Please enter input text.", "", "", "", "", ""

    if not reference_text or not reference_text.strip():
        return "Please provide a reference translation to compute BLEU.", "", "", "", "", ""

    # --- Translate ---
    result = translator.translate(input_text)

    if "error" in result:
        return result["error"], "", "", "", "", ""

    translation = result["translation"]
    confidence  = result["confidence"]   # already 0–100
    direction   = result["direction"]

    # --- Evaluate ---
    metrics = evaluator.evaluate(
        reference=reference_text,
        hypothesis=translation,
        confidence=confidence
    )

    direction_label = "English → German" if direction == "en-de" else "German → English"

    bleu_display       = f"{metrics['bleu_percent']}%  ({metrics['bleu_label']})"
    confidence_display = f"{metrics['confidence']}%  ({metrics['confidence_label']})"

    return (
        translation,
        direction_label,
        bleu_display,
        confidence_display,
    )


# ------------------------------------------------------------------
# Gradio UI
# ------------------------------------------------------------------
with gr.Blocks(title="English–German Translator") as app:

    gr.Markdown("# 🌐 Multimodal English–German Translator")

    # ── Tab 1: Translator ─────────────────────────────────────────
    with gr.Tab("Translator"):
        gr.Markdown("### Translate text or speech between English and German")

        with gr.Row():
            with gr.Column():
                text_input    = gr.Textbox(label="Text Input", lines=4,
                                           placeholder="Type English or German here…")
                audio_input   = gr.Audio(label="Or record / upload audio",
                                         type="filepath")
                translate_btn = gr.Button("Translate", variant="primary")

            with gr.Column():
                output_text  = gr.Textbox(label="Translated Output", lines=4)
                output_audio = gr.Audio(label="Speech Output")

        translate_btn.click(
            process_input,
            inputs=[text_input, audio_input],
            outputs=[output_text, output_audio],
        )

    # ── Tab 2: Attention Visualizer ───────────────────────────────
    with gr.Tab("Attention Visualizer"):
        gr.Markdown(
            "### Cross-Attention Visualizer\n"
            "See how the model attends to each input token when generating "
            "each output token. Works for both English → German and German → English."
        )

        with gr.Row():
            with gr.Column(scale=1):
                attn_input = gr.Textbox(
                    label="Input Text", lines=4,
                    placeholder="Enter English or German text…"
                )
                attn_btn = gr.Button("Translate + Visualize Attention",
                                     variant="primary")
                attn_translation = gr.Textbox(label="Translation Output",
                                              lines=3, interactive=False)

            with gr.Column(scale=2):
                gr.Markdown("#### Last Layer — Cross-Attention Heatmap")
                gr.Markdown(
                    "Each row is an output (translated) token. "
                    "Each column is an input token. "
                    "Brighter = stronger attention."
                )
                heatmap_output = gr.Image(label="Cross-Attention Heatmap",
                                          type="filepath")

        gr.Markdown("#### All Decoder Layers — Attention Grid")
        gr.Markdown(
            "Each panel shows attention weights for one decoder layer. "
            "Early layers tend to attend broadly; later layers focus more precisely."
        )
        layer_grid_output = gr.Image(label="Layer-by-Layer Attention",
                                     type="filepath")

        attn_btn.click(
            process_with_attention,
            inputs=[attn_input],
            outputs=[attn_translation, heatmap_output, layer_grid_output],
        )

    # ── Tab 3: Evaluation Dashboard ───────────────────────────────
    with gr.Tab("Evaluation Dashboard"):

        gr.Markdown("### 📊 Translation Evaluation Dashboard")
        gr.Markdown(
            "Enter your text and a **reference translation** (what you expect the correct "
            "translation to be). The dashboard will show how well the model performed."
        )

        with gr.Row():
            with gr.Column():
                eval_input = gr.Textbox(
                    label="Input Text",
                    lines=4,
                    placeholder="Type English or German text to translate…"
                )
                eval_reference = gr.Textbox(
                    label="Reference Translation",
                    lines=4,
                    placeholder="Type the expected correct translation here…"
                )
                eval_btn = gr.Button("Evaluate", variant="primary")

            with gr.Column():
                eval_translation = gr.Textbox(
                    label="Model's Translation",
                    lines=4,
                    interactive=False
                )
                eval_direction = gr.Textbox(
                    label="Detected Direction",
                    interactive=False
                )

        gr.Markdown("#### Metrics")

        with gr.Row():
            eval_bleu = gr.Textbox(
                label="BLEU Score",
                interactive=False,
                info="Measures how closely the model's translation matches your reference. Higher is better."
            )
            eval_confidence = gr.Textbox(
                label="Language Detection Confidence",
                interactive=False,
                info="How confident the model was when identifying the input language."
            )

        gr.Markdown(
            "_**BLEU Score Guide:** Poor < 20% · Fair 20–40% · Good 40–60% · Excellent > 60%_"
        )

        eval_btn.click(
            process_evaluation,
            inputs=[eval_input, eval_reference],
            outputs=[
                eval_translation,
                eval_direction,
                eval_bleu,
                eval_confidence,
            ],
        )

app.launch()