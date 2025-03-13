import argparse
from colorama import Fore, Style, init
from pytubefix import YouTube
from pydub import AudioSegment
import os
import yt_dlp
import threading

# Initialize colorama for colored terminal output
init(autoreset=True)

def search_youtube_videos(query: str, max_results: int = 5, time_limit: int = 240):
    """
    Searches YouTube for videos based on a query and returns their links.

    Args:
        query (str): The search term to find YouTube videos.
        max_results (int, optional): The maximum number of results to return. Defaults to 5.
        time_limit (int, optional): The maximum duration (in seconds) for videos to be included. Defaults to 240 seconds (4 minutes).

    Returns:
        dict[str, list[str]]: A dictionary of titles and video links.
    """
    ydl_opts = {
        "quiet": True,
        "default_search": f"ytsearch{max_results * 2}",
        "skip_download": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(query, download=False)
        videos = info.get("entries", [])
    
    filtered_links: dict[str, list[str]] = {"title": [], "link": []}
    for video in videos:
        if not video:
            continue
        duration = video.get("duration", 0)
        if duration <= time_limit:
            filtered_links["title"].append(video["title"])
            filtered_links["link"].append(video["webpage_url"])
        if len(filtered_links["link"]) == max_results:
            break
    
    return filtered_links


def _download_and_convert_audio(link: str, output_path: str, index: int):
    """
    Downloads and converts a YouTube video to WAV format and logs metadata.

    Args:
        link (str): The YouTube video URL.
        output_path (str): The directory where the audio file will be saved.
        index (int): The index number to name the file uniquely.
    """
    try:
        yt = YouTube(link)
        
        # Extract audio stream
        video = yt.streams.filter(only_audio=True).first()
        if not video:
            print(Fore.RED + f"No audio stream found for '{yt.title}'." + Style.RESET_ALL)
            return

        # Download the audio file
        out_file = video.download(output_path=output_path, filename=f"{index}.mp4")
        
        # Convert to WAV format
        audio = AudioSegment.from_file(out_file)
        wav_file = os.path.join(output_path, f"{index}.wav")
        audio.export(wav_file, format="wav")
        
        # Remove the original downloaded file
        os.remove(out_file)
        
        print(Fore.GREEN + f"'{yt.title}' has been successfully downloaded and converted to WAV." + Style.RESET_ALL)
    except Exception as e:
        print(Fore.RED + f"Failed to process '{link}': {str(e)}" + Style.RESET_ALL)

def youtube_to_audio(video_info: dict[str, list[str]], query: str):
    """
    Downloads YouTube videos as audio files and converts them to WAV format using threads.
    Also creates an info.txt file storing metadata.

    Args:
        video_info (dict[str, list[str]]): A dictionary containing video titles and links.
        query (str): The search query, which is used to create a separate folder.
    """
    output_path = os.path.join(os.getcwd(), query)
    info_file = os.path.join(output_path, "info.txt")
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    threads = []
    links: list[str] = video_info["link"]
    for index, link in enumerate(links):
        # Edit info.txt
        with open(info_file, "a", encoding="utf-8") as f:
            print(video_info['title'][index])
            f.write(f"{index:03d} | {link} | {video_info['title'][index]}\n")

        thread = threading.Thread(target=_download_and_convert_audio, args=(link, output_path, index))
        threads.append(thread)
        thread.start()
    
    for thread in threads:
        thread.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and convert YouTube audio from search query.")
    parser.add_argument("query", type=str, help="Search query for YouTube videos.")
    parser.add_argument("max_results", type=int, help="Number of videos to fetch and process.")
    
    args = parser.parse_args()
    
    video_info = search_youtube_videos(args.query, args.max_results)
    youtube_to_audio(video_info, args.query)