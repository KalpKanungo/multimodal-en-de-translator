import asyncio
import edge_tts
import os
import uuid

class TTS:
    def __init__(self):
        self.output_dir = "tts_outputs"
        os.makedirs(self.output_dir, exist_ok=True)

    async def _generate_audio(self, text: str, voice: str, output_path: str):
        communicate = edge_tts.Communicate(text=text, voice=voice)
        await communicate.save(output_path)

    def synthesize(self, text: str, lang: str):
        if not text.strip():
            return None

        # Choose voice based on language
        if lang == "de":
            voice = "de-DE-KatjaNeural"
        else:
            voice = "en-GB-RyanNeural"

        filename = f"{uuid.uuid4()}.mp3"
        output_path = os.path.join(self.output_dir, filename)

        asyncio.run(self._generate_audio(text, voice, output_path))

        return output_path
