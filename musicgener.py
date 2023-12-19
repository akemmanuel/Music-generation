import torchaudio
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_write
import time
start_time = time.time()

model = MusicGen.get_pretrained('melody')
model.set_generation_params(duration=49)  # generate 8 seconds.

descriptions = ["classical guitar"]
wav = model.generate(descriptions)  # generates 3 samples.

#melody, sr = torchaudio.load('./assets/bach.mp3')
# generates using the melody from the given audio and the provided descriptions.
#wav = model.generate_with_chroma(descriptions, melody[None].expand(3, -1, -1), sr)

for idx, one_wav in enumerate(wav):
    # Will save under {idx}.wav, with loudness normalization at -14 db LUFS.
    audio_write(f'{descriptions[idx]}', one_wav.cpu(), model.sample_rate, strategy="loudness", loudness_compressor=True)

# Endzeitpunkt erfassen
end_time = time.time()

# Dauer berechnen
duration = end_time - start_time

print("Dauer:", round(duration) / 60, "Minuten")


