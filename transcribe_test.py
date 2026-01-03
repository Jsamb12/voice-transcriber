import whisper

AUDIO_FILE = "test.wav"
MODEL_SIZE = "base"

model = whisper.load_model(MODEL_SIZE)
result = model.transcribe(AUDIO_FILE)

print(result["text"].strip())