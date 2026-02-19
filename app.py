import gradio as gr
from backend.translator import Translator
from backend.asr import ASR

translator = Translator()
asr = ASR()

def process_input(text, audio):

    if text and text.strip():
        input_text = text
    elif audio is not None:
        input_text = asr.transcribe(audio)
        print("ASR OUTPUT:", input_text)
    else:
        return ""

    result = translator.translate(input_text)

    if "error" in result:
        return result["error"]

    return result["translation"]


with gr.Blocks() as app:

    gr.Markdown("# Multimodal English–German Translator")

    text_input = gr.Textbox(label="Text Input", lines=3)

    audio_input = gr.Audio(label="Record or Upload Audio", type="filepath")

    translate_button = gr.Button("Translate")

    output_text = gr.Textbox(label="Translated Output")

    translate_button.click(
        process_input,
        inputs=[text_input, audio_input],
        outputs=output_text
    )

app.launch()
