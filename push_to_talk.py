import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write
import whisper 

SAMPLE_RATE = 16_000
CHANNELS = 1
AUDIO_FILE = "input.wav"
MODEL_SIZE = "base"

def record_until_enter(sample_rate: int, channels: int) -> np.ndarray: 
    """  
    Record from the default microphone until the user presses ENTER. 
    Returns a NumPy array of shape (n_samples, channels) with dtype int16. 
    """
    input("Press ENTER to start recording...")
    print("Recording... press ENTER again to stop.")

    chunks: list[np.ndarray] = []

    def callback(indata: np.ndarray, frames: int, time, status) -> None: 
        # This function is called repeatedly by sounddevice as audio arrives
        # Copy the data and store

        if status: # If no fatal warning, then we're good to go
            print(status, flush=True)
        chunks.append(indata.copy()) # Take a copy to freeze the data as it was for that moment

        # Start a live input stream; while it runs, the callback collects chunks.
        with sd.InputStream(
            samplerate=sample_rate, 
            channels=channels, 
            dtype="int16", 
            callback=callback
        ): 
            input() # wait for ENTER to stop
        
        print("Stopped recording.")

        if not chunks: 
            return np.empty((0, channels), dtype=np.int16)
    
        return np.concatenate(chunks, axis=0)

def main() -> None: 
    print("Push-to-talk demo (ENTER to start, ENTER to stop.")

    audio = record_until_enter(SAMPLE_RATE, CHANNELS)

    if audio.size == 0: 
        print("No audio captured.")
        return

    print(f"Captured audio: shape={audio.shape}, dtype={audio.dtype}")
    print("Done.")

if __name__ == "__main__":
    main()
