# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md
import json
import whisperx
import torch
from cog import BasePredictor, Input, Path
import os
os.environ['HF_HOME'] = '/src/hf_models'
os.environ['TORCH_HOME'] = '/src/torch_models'


compute_type = "float16"


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.device = "cuda"
        self.model = whisperx.load_model(
            "large-v2", self.device, compute_type=compute_type)
        self.allign_model_en, self.metadata_en = whisperx.load_align_model(language_code='en', device=self.device)
        self.allign_model_ru, self.metadata_ru = whisperx.load_align_model(language_code='ru', device=self.device)

    def predict(
        self,
        audio: Path = Input(description="Audio file", default="https://pyannote-speaker-diarization.s3.eu-west-2.amazonaws.com/lex-levin-4min.mp3"),
        batch_size: int = Input(description="Parallelization of input audio transcription", default=32),
        hugging_face_token: str = Input(description="Your Hugging Face access token. If empty skip diarization.", default=None),
        debug: bool = Input(description="Print out memory usage information.", default=False)
    ) -> str:
        """Run a single prediction on the model"""
        with torch.inference_mode():
            # 1. Transcribe with original whisper (batched)
            audio = whisperx.load_audio(str(audio))
            result = self.model.transcribe(audio, batch_size=batch_size)

            # 2. Align whisper output
            lang = result["language"]
            if lang == 'en':
                result = whisperx.align(result['segments'], self.allign_model_en, self.metadata_en, audio, self.device, return_char_alignments=False)
            elif lang == 'ru':
                result = whisperx.align(result['segments'], self.allign_model_ru, self.metadata_ru, audio, self.device, return_char_alignments=False)
            else:
                model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=self.device)
                result = whisperx.align(result['segments'], model_a, metadata, audio, self.device, return_char_alignments=False)

            # 3. Assign speaker labels
            if hugging_face_token:
                diarize_model = whisperx.DiarizationPipeline(use_auth_token=hugging_face_token, device=self.device)
                diarize_segments = diarize_model(audio)
                result = whisperx.assign_word_speakers(diarize_segments, result)

            if debug:
                print(
                    f"max gpu memory allocated over runtime: {torch.cuda.max_memory_reserved() / (1024 ** 3):.2f} GB")
        return json.dumps(result['segments'])
