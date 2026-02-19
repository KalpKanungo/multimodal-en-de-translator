import torch
import librosa
from transformers import WhisperProcessor, WhisperForConditionalGeneration


class ASR:
    def __init__(self):
        self.device = torch.device("cpu")
        self.processor = WhisperProcessor.from_pretrained("openai/whisper-small")
        self.model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small").to(self.device)
        self.model.eval()

    def transcribe(self, audio_path):
        audio, sr = librosa.load(audio_path, sr=16000)
        inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            predicted_ids = self.model.generate(**inputs)

        transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        return transcription
