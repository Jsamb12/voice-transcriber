import sounddevice as sd
from scipy.io.wavfile import write 

SAMPLE_RATE = 16_000
SECONDS = 5

print("Recording... Speak now!")
audio = sd.rec(int(SECONDS * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype="int16")
sd.wait()
write("test.wav", SAMPLE_RATE, audio)

print("Saved test.wav")