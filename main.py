import whisper
import os
print("Running", flush=True)
print("Current working directory:", os.getcwd(), flush=True)

from pyannote.audio import Pipeline
from pyannote.core import Segment
from pydub import AudioSegment
import argparse
import torch
print(torch.cuda.is_available(), flush=True)  # Should return True if ROCm is working


def transcribe_and_diarize(audio_path, num_speakers=None, whisper_model_size="base", hf_token="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"):
    print(f"Transcribing and diarizing {audio_path} with {num_speakers} speakers...")
    print("CUDA available:", torch.cuda.is_available(), flush=True)
    has_cuda = torch.cuda.is_available()
    if not has_cuda:
        print("run with CPU, no CUDA detected:", flush=True)
    # Convert .m4a to .mp3 if needed
    if audio_path.endswith(".m4a"):
        print("Converting audio file to .wav format...")
        sound = AudioSegment.from_file(audio_path)
        audio_path = audio_path.replace(".m4a", ".wav").replace(".mp4", ".wav")
        sound.export(audio_path, format="wav")
        print(f"Converted audio saved to {audio_path}")

    print("audio path", audio_path, flush=True)
    # Load Whisper model
    print("loading whisper model", whisper_model_size, flush=True)

    whisper_model = whisper.load_model(whisper_model_size, device="cuda" if has_cuda else "cpu")

    # Transcribe audio
    print("Transcribing video of ", audio_path, flush=True)
    transcription = whisper_model.transcribe(audio_path)

    # Load Pyannote speaker diarization pipeline
    print("start speaker diarization", flush=True)
    diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", use_auth_token=hf_token)
    diarization_pipeline.to(torch.device("cuda" if has_cuda else "cpu"))
    # Apply diarization
    diarization = diarization_pipeline(audio_path, num_speakers=num_speakers)
    print("exporting result", flush=True)
    # Combine transcription with diarization
    result = []
    for segment in transcription["segments"]:
        start_time = segment["start"]
        end_time = segment["end"]
        text = segment["text"]

        # Find the most overlapping speaker
        speaker = "Unknown"
        max_overlap = 0.0

        for turn, _, speaker_label in diarization.itertracks(yield_label=True):
            overlap = turn & Segment(start_time, end_time)  # Get overlapping segment duration
            if overlap.duration > max_overlap:
                max_overlap = overlap.duration
                speaker = speaker_label

        result.append({"start": start_time, "end": end_time, "speaker": speaker, "text": text})

    # Output the combined result
    for entry in result:
        print(f"[{entry['start']:.2f}s - {entry['end']:.2f}s] Speaker {entry['speaker']}: {entry['text']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe and diarize an audio file.")
    parser.add_argument("audio_path", type=str, help="Path to the audio file.")
    parser.add_argument("--num_speakers", type=int, default=None, help="Number of speakers in the audio.")
    parser.add_argument("--whisper_model_size", type=str, default="base", help="Size of the Whisper model to use.")
    parser.add_argument("--hf_token", type=str, default="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", help="Hugging Face token for authentication.")

    args = parser.parse_args()

    transcribe_and_diarize(audio_path=args.audio_path, num_speakers=args.num_speakers, whisper_model_size=args.whisper_model_size, hf_token=args.hf_token)

# python main.py ./p12.m4a --num_speakers 2 --whisper_model_size large
