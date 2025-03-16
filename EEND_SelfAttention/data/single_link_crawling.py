import argparse
from colorama import Fore, Style, init
from pytubefix import YouTube
from pydub import AudioSegment
import os
import re

# Initialize colorama for colored terminal output
init(autoreset=True)

def sanitize_url(url: str) -> str:
    """Removes any unwanted escape characters from the YouTube URL."""
    return re.sub(r'\\', '', url)

def download_and_convert_audio(link: str, output_path: str):
    """
    Downloads and converts a single YouTube video to WAV format and logs metadata.

    Args:
        link (str): The YouTube video URL.
        output_path (str): The directory where the audio file will be saved.
    """
    try:
        link = sanitize_url(link)  # Clean the URL
        yt = YouTube(link)
        
        # Extract audio stream
        video = yt.streams.filter(only_audio=True).first()
        if not video:
            print(Fore.RED + f"No audio stream found for '{yt.title}'." + Style.RESET_ALL)
            return

        # Ensure output directory exists
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # Define file paths
        out_file = os.path.join(output_path, "audio.mp4")
        wav_file = os.path.join(output_path, "audio.wav")

        # Download the audio file
        video.download(output_path=output_path, filename="audio.mp4")
        
        # Convert to WAV format
        audio = AudioSegment.from_file(out_file)
        audio.export(wav_file, format="wav")
        
        # Remove the original downloaded file
        os.remove(out_file)
        
        print(Fore.GREEN + f"'{yt.title}' has been successfully downloaded and converted to WAV." + Style.RESET_ALL)
    except Exception as e:
        print(Fore.RED + f"Failed to process '{link}': {str(e)}" + Style.RESET_ALL)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and convert a single YouTube video to WAV format.")
    parser.add_argument("link", type=str, help="YouTube video URL.")
    parser.add_argument("output_path", type=str, help="Directory to save the audio file.")
    
    args = parser.parse_args()
    
    download_and_convert_audio(args.link, args.output_path)
