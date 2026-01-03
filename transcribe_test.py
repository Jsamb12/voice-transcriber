import whisper

AUDIO_FILE = "test.wav"
MODEL_SIZE = "base"

model = whisper.load_model(MODEL_SIZE, device="cpu")
result = model.transcribe(AUDIO_FILE, fp16=False)

print(result["text"].strip())