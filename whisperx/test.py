import time

import faster_whisper


filename = "my-audio.mp3"
initial_prompt = "My podcast recording"  # Or `None`
word_timestamps = False
vad_filter = True
temperature = 0.0
language = "pt"
model_size = "large-v3"
device, compute_type = "cuda", "float16"
# or: device, compute_type = "cpu", "float32"

model = faster_whisper.WhisperModel(model_size, device=device, compute_type=compute_type)

segments, transcription_info = model.transcribe(
    filename,
    word_timestamps=word_timestamps,
    vad_filter=vad_filter,
    temperature=temperature,
    language=language,
    initial_prompt=initial_prompt,
)
print(transcription_info)

start_time = time.time()
for segment in segments:
    row = {
        "start": segment.start,
        "end": segment.end,
        "text": segment.text,
    }
    if word_timestamps:
        row["words"] = [
            {"start": word.start, "end": word.end, "word": word.word}
            for word in segment.words
        ]
    print(row)
end_time = time.time()
print(f"Transcription finished in {end_time - start_time:.2f}s")

