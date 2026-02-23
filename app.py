import gradio as gr
from backend.translator import Translator
from backend.asr import ASR
from backend.tts import TTS
from backend.evaluator import Evaluator
from backend.ocr import OCR

translator  = Translator()
asr         = ASR()
tts_engine  = TTS()
evaluator   = Evaluator()
ocr_engine  = OCR()


# ------------------------------------------------------------------
# Handler: Translator tab
# ------------------------------------------------------------------
def process_input(text, audio, image):
    """
    Priority order: text > audio > image
    Whichever input is provided first gets used.
    """
    if text and text.strip():
        input_text = text
        source = "text"

    elif audio is not None:
        input_text = asr.transcribe(audio)
        source = "audio"
        if not input_text or not input_text.strip():
            return "Could not transcribe audio. Please try again.", None, ""

    elif image is not None:
        image_path = image.name  # gr.File returns an object with .name = filepath
        input_text = ocr_engine.extract_text(image_path)
        source = "image"
        if not input_text or not input_text.strip():
            return "No text detected in image. Please try a clearer image.", None, ""

    else:
        return "Please provide text, audio, or an image.", None, ""

    result = translator.translate(input_text)

    if "error" in result:
        return result["error"], None, ""

    translated_text = result["translation"]
    target_lang     = "de" if result["direction"] == "en-de" else "en"
    audio_file      = tts_engine.synthesize(translated_text, target_lang)

    detected        = result["detected_language"].upper()
    confidence      = result["confidence"]
    direction_arrow = "English → German" if result["direction"] == "en-de" else "German → English"
    status          = f"Detected: {detected} ({confidence}% confidence)  |  {direction_arrow}  |  Input: {source}"

    return translated_text, audio_file, status


# ------------------------------------------------------------------
# Handler: Attention Visualizer tab
# ------------------------------------------------------------------
def process_with_attention(text):
    if not text or not text.strip():
        return "Please enter some text.", None, None

    translation, heatmap_path, layer_grid_path = \
        translator.translate_with_attention(text)

    return translation, heatmap_path, layer_grid_path


#
def process_evaluation(input_text, reference_text):
    if not input_text or not input_text.strip():
        return "Please enter input text.", "", "", ""

    if not reference_text or not reference_text.strip():
        return "Please provide a reference translation to compute BLEU.", "", "", ""

    result = translator.translate(input_text)

    if "error" in result:
        return result["error"], "", "", ""

    translation = result["translation"]
    confidence  = result["confidence"]
    direction   = result["direction"]

    metrics = evaluator.evaluate(
        reference=reference_text,
        hypothesis=translation,
        confidence=confidence
    )

    direction_label    = "English → German" if direction == "en-de" else "German → English"
    bleu_display       = f"{metrics['bleu_percent']}%  ({metrics['bleu_label']})"
    confidence_display = f"{metrics['confidence']}%  ({metrics['confidence_label']})"

    return translation, direction_label, bleu_display, confidence_display


with gr.Blocks(title="English–German Translator") as app:

    gr.Markdown("# 🌐 Multimodal English–German Translator")

    
    with gr.Tab("Translator"):
        gr.Markdown(
            "### Translate between English and German\n"
            "Input can be **typed text**, **recorded/uploaded audio**, or an **image** "
            "containing text (signboards, textbooks, screenshots, handwriting)."
        )

        with gr.Row():
            with gr.Column():
                text_input  = gr.Textbox(
                    label="Text Input",
                    lines=4,
                    placeholder="Type English or German here…"
                )
                audio_input = gr.Audio(
                    label="Audio Input — record or upload",
                    type="filepath"
                )
                image_input = gr.File(
                    label="Image Input — signboard, textbook, screenshot, handwriting (JPG, PNG, HEIC)",
                    file_types=["image", ".heic", ".heif"]
                )
                translate_btn = gr.Button("Translate", variant="primary")

            with gr.Column():
                status_output = gr.Textbox(
                    label="Detection Info",
                    interactive=False,
                    lines=1
                )
                output_text  = gr.Textbox(label="Translated Output", lines=4)
                output_audio = gr.Audio(label="Speech Output")

        translate_btn.click(
            process_input,
            inputs=[text_input, audio_input, image_input],
            outputs=[output_text, output_audio, status_output],
        )

    
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
                attn_translation = gr.Textbox(
                    label="Translation Output", lines=3, interactive=False
                )

            with gr.Column(scale=2):
                gr.Markdown("#### Last Layer — Cross-Attention Heatmap")
                gr.Markdown(
                    "Each row is an output (translated) token. "
                    "Each column is an input token. "
                    "Brighter = stronger attention."
                )
                heatmap_output = gr.Image(
                    label="Cross-Attention Heatmap", type="filepath"
                )

        gr.Markdown("#### All Decoder Layers — Attention Grid")
        gr.Markdown(
            "Each panel shows attention weights for one decoder layer. "
            "Early layers tend to attend broadly; later layers focus more precisely."
        )
        layer_grid_output = gr.Image(
            label="Layer-by-Layer Attention", type="filepath"
        )

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
                    label="Input Text", lines=4,
                    placeholder="Type English or German text to translate…"
                )
                eval_reference = gr.Textbox(
                    label="Reference Translation", lines=4,
                    placeholder="Type the expected correct translation here…"
                )
                eval_btn = gr.Button("Evaluate", variant="primary")

            with gr.Column():
                eval_translation = gr.Textbox(
                    label="Model's Translation", lines=4, interactive=False
                )
                eval_direction = gr.Textbox(
                    label="Detected Direction", interactive=False
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