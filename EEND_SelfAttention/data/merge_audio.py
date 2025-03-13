from pydub import AudioSegment


def merge_audio(files, output_path, format="wav"):
    """
    Merges multiple audio files into one.
    
    Args:
        files (list of str): List of audio file paths to merge.
        output_path (str): Path for the output merged audio file.
        format (str): Output format (e.g., 'mp3', 'wav', 'ogg').
    """
    if not files:
        raise ValueError("No audio files provided for merging.")

    # Load the first file
    combined_audio = AudioSegment.from_file(files[0])

    # Append the rest of the files
    for file in files[1:]:
        next_audio = AudioSegment.from_file(file)
        combined_audio += next_audio  # Concatenation

    # Export merged audio
    combined_audio.export(output_path, format=format)
    print(f"Merged audio saved as {output_path}")


if __name__=='__main__':
    audio_files = ["audio1.wav", "audio2.wav"]
    merge_audio(audio_files, "merged_audio.wav")