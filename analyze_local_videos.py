#!/usr/bin/env python3
"""
Analyze already-downloaded TikTok videos
No download needed - just transcription and analysis
"""

import os
import sys
import csv
from pathlib import Path
from datetime import datetime
import glob

try:
    import whisper
except ImportError:
    print("Error: whisper not installed. Run: uv pip install openai-whisper")
    sys.exit(1)

try:
    from moviepy.video.io.VideoFileClip import VideoFileClip
except ImportError:
    print("Error: moviepy not installed. Run: uv pip install moviepy")
    sys.exit(1)

try:
    from PIL import Image
except ImportError:
    print("Error: Pillow not installed. Run: uv pip install pillow")
    sys.exit(1)


def analyze_local_video(video_path, whisper_model='tiny'):
    """Analyze a local video file"""
    
    print(f"\n📹 Analyzing: {os.path.basename(video_path)}")
    
    # Extract audio and transcribe
    audio_path = str(Path(video_path).with_suffix('.wav'))
    
    try:
        print("  🎵 Extracting audio...")
        video = VideoFileClip(video_path)
        
        if video.audio is None:
            print("  ⚠ No audio track found")
            video.close()
            transcription = ""
            language = None
            duration = video.duration
        else:
            duration = video.duration
            video.audio.write_audiofile(audio_path, logger=None)
            video.close()
            
            # Transcribe
            print("  🎤 Transcribing...")
            model = whisper.load_model(whisper_model)
            result = model.transcribe(audio_path)
            transcription = result['text']
            language = result.get('language', 'unknown')
            
            # Clean up audio file
            os.remove(audio_path)
            
            print(f"  ✓ Transcription: {transcription[:100]}...")
        
        # Basic analysis
        text_lower = transcription.lower()
        
        # Sentiment
        positive_words = ['love', 'great', 'amazing', 'awesome', 'happy', 'best', 'wonderful', 'good', 'beautiful']
        negative_words = ['hate', 'bad', 'terrible', 'awful', 'sad', 'worst', 'horrible', 'wrong', 'ugly']
        
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        if pos_count > neg_count:
            sentiment = 'positive'
        elif neg_count > pos_count:
            sentiment = 'negative'
        else:
            sentiment = 'neutral'
        
        # Extract topics (simple word frequency)
        words = [w for w in text_lower.split() if len(w) > 4]
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        topics = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
        topics_list = [word for word, freq in topics]
        
        # Generate intelligent one-sentence summary
        if transcription:
            clean_text = transcription.strip()
            text_lower = clean_text.lower()
            
            # Extract key information intelligently
            # Look for main actors and actions
            summary = ""
            
            # Pattern matching for common structures
            if 'asked' in text_lower and 'gpt' in text_lower:
                if 'roast' in text_lower:
                    # Extract who's being roasted
                    if 'roast me' in text_lower:
                        summary = "Creator asks ChatGPT to roast them"
                    else:
                        # Try to find who's being roasted
                        words = clean_text.split()
                        for i, word in enumerate(words):
                            if word.lower() == 'roast' and i+1 < len(words):
                                target = words[i+1].strip('.,')
                                summary = f"ChatGPT roast of {target}"
                                break
                        if not summary:
                            summary = "ChatGPT roast request"
            
            elif 'tutorial' in text_lower or 'how to' in text_lower:
                # Extract what's being taught
                if 'how to' in text_lower:
                    idx = text_lower.find('how to')
                    rest = clean_text[idx+6:idx+50].strip()
                    summary = f"Tutorial on how to {rest.split('.')[0]}"
                else:
                    summary = "Educational tutorial"
            
            elif 'review' in text_lower:
                # Extract what's being reviewed
                summary = "Product/content review"
                
            elif 'story' in text_lower or 'time when' in text_lower:
                summary = "Personal story"
                
            elif 'dance' in text_lower or 'dancing' in text_lower:
                summary = "Dance performance"
                
            elif 'music' in text_lower or 'song' in text_lower:
                summary = "Music content"
            
            # Fallback: extract main idea from first sentence
            if not summary:
                sentences = [s.strip() for s in clean_text.split('.') if s.strip()]
                if sentences:
                    first = sentences[0]
                    # Simplify and extract core meaning
                    if len(first) > 100:
                        first = first[:97] + '...'
                    summary = first
                else:
                    summary = clean_text[:100]
            
            context_summary = summary if summary.endswith('.') else summary + '.'
        else:
            context_summary = "Silent video with no spoken content."
        
        # Extract video title and ID from filename if available
        filename = os.path.basename(video_path)
        # Format: "title"_videoid.mp4 or videoid.mp4
        if '_' in filename:
            parts = filename.rsplit('_', 1)
            title = parts[0].strip('"')
            video_id = parts[1].replace('.mp4', '').replace('.mov', '')
        else:
            title = filename.replace('.mp4', '').replace('.mov', '')
            video_id = ''
        
        result = {
            'timestamp': datetime.now().isoformat(),
            'filename': filename,
            'title': title,
            'video_id': video_id,
            'creator': '',  # Can be filled manually or from TikTok metadata
            'duration_seconds': round(duration, 2),
            'language': language if language else 'none',
            'transcription': transcription,
            'context': context_summary,
            'topics': ', '.join(topics_list),
            'sentiment': sentiment,
            'word_count': len(transcription.split()),
        }
        
        print(f"  ✓ Sentiment: {sentiment}")
        print(f"  ✓ Topics: {', '.join(topics_list[:3])}")
        print(f"  ✓ Context: {context_summary[:80]}...")
        
        return result
        
    except Exception as e:
        print(f"  ✗ Error: {str(e)}")
        return None


def main():
    print("\n🎬 Local TikTok Video Analyzer")
    print("="*70)
    
    # Get video files
    video_dir = input("Enter directory with videos [default: current directory]: ").strip() or "."
    
    # Find all video files
    video_extensions = ['*.mp4', '*.mov', '*.avi', '*.mkv']
    video_files = []
    seen_video_ids = set()  # Track video IDs to avoid duplicates
    
    for ext in video_extensions:
        found_files = glob.glob(os.path.join(video_dir, '**', ext), recursive=True)
        for filepath in found_files:
            # Skip files in analysis_output, batch_analysis_output, etc.
            if 'analysis_output' in filepath or 'batch_analysis' in filepath:
                continue
            
            # Extract video ID from filename to detect duplicates
            basename = os.path.basename(filepath)
            # Look for pattern like _7481747639336176939.mp4
            video_id = None
            if '_' in basename:
                parts = basename.rsplit('_', 1)
                potential_id = parts[1].replace('.mp4', '').replace('.mov', '')
                if potential_id.isdigit():
                    video_id = potential_id
            
            # Skip if we've seen this video ID before
            if video_id and video_id in seen_video_ids:
                continue
            
            video_files.append(filepath)
            if video_id:
                seen_video_ids.add(video_id)
    
    if not video_files:
        print(f"\n⚠ No video files found in '{video_dir}'")
        
        # Manual entry
        manual = input("\nEnter video file path manually? (y/n): ").strip().lower()
        if manual == 'y':
            while True:
                path = input("Video path (or 'quit'): ").strip()
                if path.lower() == 'quit':
                    break
                if os.path.exists(path):
                    video_files.append(path)
                else:
                    print(f"  ✗ File not found: {path}")
    
    if not video_files:
        print("\n⚠ No videos to analyze. Exiting.")
        sys.exit(0)
    
    print(f"\n📋 Found {len(video_files)} video(s)")
    for i, vf in enumerate(video_files, 1):
        print(f"  {i}. {os.path.basename(vf)}")
    
    # Choose model
    print("\nWhisper model:")
    print("  tiny   = fastest (less accurate)")
    print("  base   = balanced (recommended)")
    print("  small  = slower (more accurate)")
    model = input("Choose [default: tiny]: ").strip() or "tiny"
    
    # Load model once
    print(f"\n🔄 Loading Whisper model ({model})...")
    
    results = []
    
    # Process each video
    for i, video_path in enumerate(video_files, 1):
        print(f"\n[{i}/{len(video_files)}] " + "="*60)
        result = analyze_local_video(video_path, model)
        if result:
            results.append(result)
    
    # Save to CSV
    if results:
        csv_path = 'local_video_analysis.csv'
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            fieldnames = list(results[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        
        print("\n" + "="*70)
        print("✅ Analysis Complete!")
        print("="*70)
        print(f"📊 Results saved to: {csv_path}")
        print(f"📹 Videos analyzed: {len(results)}")
        print("="*70 + "\n")
    else:
        print("\n⚠ No videos were successfully analyzed.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
        sys.exit(0)

