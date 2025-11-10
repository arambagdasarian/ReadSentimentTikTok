#!/usr/bin/env python3
"""
TikTok Video and Audio Downloader
Downloads TikTok videos and extracts audio using yt-dlp
"""

import yt_dlp
import os
import sys
from pathlib import Path


def download_tiktok_video(url, output_dir='downloads'):
    """
    Download TikTok video with best quality
    
    Args:
        url (str): TikTok video URL
        output_dir (str): Directory to save the video
    
    Returns:
        str: Path to downloaded video file
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    output_template = os.path.join(output_dir, '%(title)s_%(id)s.%(ext)s')
    
    ydl_opts = {
        'outtmpl': output_template,
        'format': 'best',  # Download best quality
        'quiet': False,
        'no_warnings': False,
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info)
            print(f"✓ Video downloaded successfully: {filename}")
            return filename
    except Exception as e:
        print(f"✗ Error downloading video: {str(e)}")
        return None


def download_tiktok_audio(url, output_dir='downloads', audio_format='mp3'):
    """
    Download TikTok video and extract audio directly
    
    Args:
        url (str): TikTok video URL
        output_dir (str): Directory to save the audio
        audio_format (str): Audio format (mp3, m4a, wav, etc.)
    
    Returns:
        str: Path to extracted audio file
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    output_template = os.path.join(output_dir, '%(title)s_%(id)s.%(ext)s')
    
    ydl_opts = {
        'outtmpl': output_template,
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': audio_format,
            'preferredquality': '192',
        }],
        'quiet': False,
        'no_warnings': False,
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            # Prepare filename with audio extension
            base_filename = ydl.prepare_filename(info)
            audio_filename = os.path.splitext(base_filename)[0] + f'.{audio_format}'
            print(f"✓ Audio extracted successfully: {audio_filename}")
            return audio_filename
    except Exception as e:
        print(f"✗ Error extracting audio: {str(e)}")
        return None


def download_both(url, output_dir='downloads', audio_format='mp3'):
    """
    Download both video and audio
    
    Args:
        url (str): TikTok video URL
        output_dir (str): Directory to save files
        audio_format (str): Audio format for extraction
    """
    print("\n" + "="*60)
    print("TikTok Downloader")
    print("="*60)
    print(f"URL: {url}")
    print(f"Output directory: {output_dir}")
    print("="*60 + "\n")
    
    print("📥 Downloading video...")
    video_path = download_tiktok_video(url, output_dir)
    
    print("\n🎵 Extracting audio...")
    audio_path = download_tiktok_audio(url, output_dir, audio_format)
    
    print("\n" + "="*60)
    print("Download Complete!")
    print("="*60)
    if video_path:
        print(f"📹 Video: {video_path}")
    if audio_path:
        print(f"🎵 Audio: {audio_path}")
    print("="*60 + "\n")


def main():
    """Main function to handle command-line usage"""
    print("\n🎬 TikTok Video & Audio Downloader")
    print("=" * 60)
    
    # Check if URL is provided as command-line argument
    if len(sys.argv) > 1:
        url = sys.argv[1]
    else:
        url = input("\n📎 Enter TikTok video URL: ").strip()
    
    if not url:
        print("✗ No URL provided. Exiting.")
        sys.exit(1)
    
    # Ask user what they want to download
    print("\nWhat would you like to download?")
    print("1. Video only")
    print("2. Audio only")
    print("3. Both video and audio")
    
    choice = input("\nEnter your choice (1-3) [default: 3]: ").strip() or "3"
    
    output_dir = input("Enter output directory [default: downloads]: ").strip() or "downloads"
    
    try:
        if choice == "1":
            print("\n📥 Downloading video...")
            download_tiktok_video(url, output_dir)
        elif choice == "2":
            print("\n🎵 Extracting audio...")
            download_tiktok_audio(url, output_dir)
        else:
            download_both(url, output_dir)
    except KeyboardInterrupt:
        print("\n\n✗ Download cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ An error occurred: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()

