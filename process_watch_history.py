#!/usr/bin/env python3

import os
import sys
import csv
import base64
import json
import time
from pathlib import Path
from datetime import datetime
import yt_dlp
import whisper
from moviepy.video.io.VideoFileClip import VideoFileClip
from PIL import Image
import io

try:
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from dotenv import load_dotenv
    # Load .env file from the script's directory
    script_dir = Path(__file__).parent.absolute()
    load_dotenv(dotenv_path=script_dir / '.env')
except ImportError:
    pass  # python-dotenv not installed, skip .env loading


def parse_watch_history(file_path):
    entries = []
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    current_date = None
    for line in lines:
        line = line.strip()
        if line.startswith('Date:'):
            date_str = line.replace('Date:', '').strip()
            try:
                current_date = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S UTC')
            except:
                current_date = None
        elif line.startswith('Link:'):
            url = line.replace('Link:', '').strip()
            if url and current_date:
                url = url.replace('tiktokv.com', 'tiktok.com')
                entries.append({'date': current_date, 'url': url})
    
    return entries


def download_video(url, output_dir='downloads', skip_if_exists=True, max_retries=3, use_cookies=False):
    """
    Download video with retry logic and optional cookie support.
    
    Args:
        url: Video URL to download
        output_dir: Directory to save video
        skip_if_exists: Skip if video already exists
        max_retries: Maximum number of retry attempts
        use_cookies: Whether to use browser cookies for authentication
    
    Returns:
        Tuple of (video_path, info_dict) or (None, None) if failed
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_template = os.path.join(output_dir, '%(id)s.%(ext)s')
    
    # Build yt-dlp options
    ydl_opts = {
        'outtmpl': output_template,
        'format': 'best',
        'quiet': True,
        'no_warnings': True,
        'ignoreerrors': False,
    }
    
    # Add cookie support if requested
    if use_cookies:
        # Try to get cookies from browser automatically
        ydl_opts['cookiesfrombrowser'] = ('chrome',)  # Can also use 'firefox', 'edge', etc.
    
    for attempt in range(max_retries):
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # First, try to get video info
                info = ydl.extract_info(url, download=False)
                video_id = info.get('id', '')
                video_path = ydl.prepare_filename(info)
                
                if skip_if_exists and os.path.exists(video_path):
                    print(f"  Video already downloaded: {os.path.basename(video_path)}")
                    return video_path, info
                
                # Download the video
                ydl.download([url])
                
                # Verify file exists
                if os.path.exists(video_path):
                    return video_path, info
                else:
                    raise Exception("Download completed but file not found")
                    
        except yt_dlp.utils.DownloadError as e:
            error_str = str(e)
            
            # Check for specific error types
            if 'IP address is blocked' in error_str:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 5  # 5, 10, 15 seconds
                    print(f"  IP blocked, retrying in {wait_time}s (attempt {attempt + 2}/{max_retries})...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"  Error: IP address blocked. Consider using cookies or VPN.")
                    return None, None
                    
            elif 'Log in for access' in error_str or 'cookies' in error_str.lower():
                if not use_cookies and attempt == 0:
                    print(f"  Warning: Video requires authentication. Consider using cookies.")
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 3
                    print(f"  Authentication required, retrying in {wait_time}s (attempt {attempt + 2}/{max_retries})...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"  Error: Video requires login. Use --cookies-from-browser or set use_cookies=True")
                    return None, None
                    
            else:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2
                    print(f"  Download failed, retrying in {wait_time}s (attempt {attempt + 2}/{max_retries})...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"  Error downloading: {error_str[:100]}")
                    return None, None
                    
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2
                print(f"  Unexpected error, retrying in {wait_time}s (attempt {attempt + 2}/{max_retries})...")
                time.sleep(wait_time)
                continue
            else:
                print(f"  Error downloading {url}: {str(e)[:100]}")
                return None, None
    
    return None, None


def analyze_with_llm(transcription, ocr_text, screenshots):
    """
    Analyze video content using OpenAI's GPT-4 Vision model.
    Returns structured analysis including description, sentiment, and image description.
    """
    if not OPENAI_AVAILABLE:
        print("  Warning: OpenAI not available, using fallback analysis")
        return None
    
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("  Warning: OPENAI_API_KEY not set, using fallback analysis")
        return None
    
    # Debug: Verify API key is loaded
    if len(api_key) < 20:
        print(f"  Warning: API key seems invalid (too short: {len(api_key)} chars), using fallback")
        return None
    
    try:
        client = OpenAI(api_key=api_key)
        
        # Prepare content for API
        content_parts = []
        
        # Add text content
        text_content = ""
        if transcription and transcription.strip():
            text_content += f"Audio Transcription: {transcription[:2000]}\n\n"
        if ocr_text and ocr_text.strip():
            text_content += f"Text extracted from video frames: {ocr_text[:1000]}\n\n"
        
        if text_content:
            content_parts.append({
                "type": "text",
                "text": text_content
            })
        
        # Add screenshots (up to 5)
        screenshot_count = 0
        for i, screenshot in enumerate(screenshots[:5]):
            try:
                base64_image = image_to_base64(screenshot)
                content_parts.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                })
                screenshot_count += 1
            except Exception as e:
                print(f"  Warning: Could not encode screenshot {i}: {e}")
        
        # If we have no screenshots but have transcription, still proceed with text-only analysis
        if screenshot_count == 0 and not text_content:
            print("  Warning: No screenshots and no transcription text available for LLM analysis")
            return None
        
        if not content_parts:
            print("  Warning: No content to analyze (no transcription, OCR text, or screenshots)")
            return None
        
        # Create the prompt
        prompt = """Analyze this TikTok video content and provide a comprehensive analysis in JSON format.

Consider:
1. The audio transcription (what was said)
2. Any text visible in the video frames
3. The visual content in the screenshots

Provide your analysis as a JSON object with the following structure:
{
  "description": "A one-sentence description of what the video is about",
  "sentiment_analysis": {
    "sentiment_label": "positive, negative, or neutral",
    "sentiment_score": A number between -1.0 (very negative) and 1.0 (very positive),
    "confidence": A number between 0.0 and 1.0 indicating confidence in the sentiment analysis,
    "reasoning": "Brief explanation of why this sentiment was assigned"
  },
  "image_description": "A brief description of the visual content shown in the video screenshots, including any notable elements, people, objects, scenes, or activities visible"
}

Be thorough and accurate in your analysis."""
        
        content_parts.insert(0, {"type": "text", "text": prompt})
        
        # Call OpenAI API
        response = client.chat.completions.create(
            model="gpt-4o",  # Using GPT-4o for vision capabilities
            messages=[
                {
                    "role": "user",
                    "content": content_parts
                }
            ],
            max_tokens=1000,
            temperature=0.3,
            response_format={"type": "json_object"}
        )
        
        result_text = response.choices[0].message.content
        result = json.loads(result_text)
        
        return result
        
    except Exception as e:
        error_msg = str(e)
        print(f"  Warning: LLM analysis failed: {error_msg}")
        # Print more details for debugging
        if "401" in error_msg or "authentication" in error_msg.lower():
            print("  Error: Invalid API key. Please check your OPENAI_API_KEY.")
        elif "429" in error_msg or "rate limit" in error_msg.lower():
            print("  Error: Rate limit exceeded. Please wait and try again.")
        elif "insufficient_quota" in error_msg.lower():
            print("  Error: Insufficient quota. Please check your OpenAI account balance.")
        return None


def analyze_sentiment_from_context(description, transcription):
    """
    Context-aware sentiment analysis based on generated description and transcription context.
    Uses semantic understanding of the content rather than word-by-word matching.
    Returns sentiment_label, sentiment_score, confidence, reasoning.
    """
    if not description and not transcription:
        return 'neutral', 0.0, 0.3, "No content available for sentiment analysis."
    
    # Combine description and transcription for context
    context = (description or "") + " " + (transcription or "")
    context_lower = context.lower()
    
    # Context-based sentiment patterns (semantic, not word-by-word)
    # Positive contexts
    positive_patterns = [
        'happy', 'joy', 'excited', 'love', 'amazing', 'wonderful', 'great', 'best',
        'success', 'win', 'achievement', 'grateful', 'thankful', 'appreciate',
        'beautiful', 'perfect', 'fantastic', 'brilliant', 'incredible',
        'funny', 'hilarious', 'entertaining', 'enjoy', 'pleased', 'satisfied'
    ]
    
    # Negative contexts
    negative_patterns = [
        'sad', 'angry', 'frustrated', 'disappointed', 'hate', 'terrible', 'awful',
        'worst', 'bad', 'horrible', 'disgusting', 'problem', 'issue', 'fail',
        'hurt', 'pain', 'scared', 'afraid', 'worried', 'fear', 'struggle',
        'difficult', 'hard', 'challenge', 'conflict', 'argument', 'fight'
    ]
    
    # Neutral/educational contexts
    neutral_patterns = [
        'tutorial', 'how to', 'explain', 'educational', 'informative', 'review',
        'discuss', 'share', 'story', 'remember', 'happened', 'music', 'dance'
    ]
    
    # Analyze context semantically
    pos_context_score = sum(1 for pattern in positive_patterns if pattern in context_lower)
    neg_context_score = sum(1 for pattern in negative_patterns if pattern in context_lower)
    neutral_context_score = sum(1 for pattern in neutral_patterns if pattern in context_lower)
    
    # Determine sentiment based on context dominance
    total_context_indicators = pos_context_score + neg_context_score + neutral_context_score
    
    if total_context_indicators == 0:
        # No clear context indicators - default to neutral
        return 'neutral', 0.0, 0.3, "Content context does not clearly indicate sentiment."
    
    # Calculate sentiment based on context ratios
    if pos_context_score > neg_context_score and pos_context_score > neutral_context_score:
        sentiment_label = 'positive'
        # Score based on how dominant positive context is
        dominance = pos_context_score / max(total_context_indicators, 1)
        sentiment_score = round(0.3 + (dominance * 0.5), 2)  # Range: 0.3 to 0.8
        confidence = round(min(0.5 + (pos_context_score / 10), 0.8), 2)
        reasoning = f"Context analysis indicates positive sentiment based on content themes."
    elif neg_context_score > pos_context_score and neg_context_score > neutral_context_score:
        sentiment_label = 'negative'
        dominance = neg_context_score / max(total_context_indicators, 1)
        sentiment_score = round(-0.3 - (dominance * 0.5), 2)  # Range: -0.3 to -0.8
        confidence = round(min(0.5 + (neg_context_score / 10), 0.8), 2)
        reasoning = f"Context analysis indicates negative sentiment based on content themes."
    else:
        # Neutral or mixed
        sentiment_label = 'neutral'
        # Slight bias based on pos/neg difference
        diff = pos_context_score - neg_context_score
        sentiment_score = round(diff * 0.1, 2)  # Small range: -0.3 to 0.3
        confidence = round(min(0.4 + (neutral_context_score / 10), 0.7), 2)
        reasoning = f"Context analysis indicates neutral sentiment or mixed content."
    
    return sentiment_label, sentiment_score, confidence, reasoning


def generate_context_with_rules(clean_text):
    """
    Improved rule-based context generation system as fallback.
    """
    if not clean_text or not clean_text.strip():
        return "No transcript or text content available."
    
    text_lower = clean_text.lower()
    sentences = [s.strip() for s in clean_text.split('.') if s.strip() and len(s.strip()) > 10]
    
    if not sentences:
        # If no sentences, try to create a summary from the text
        if len(clean_text) > 150:
            return clean_text[:147] + '...'
        return clean_text.strip() + '.' if clean_text.strip() else "No transcript available."
    
    first_sentence = sentences[0]
    
    if len(sentences) == 1:
        if len(first_sentence) > 200:
            return first_sentence[:197] + '...'
        else:
            return first_sentence + ('.' if not first_sentence.endswith('.') else '')
    
    # Pattern matching for different content types
    if 'dating' in text_lower or 'relationship' in text_lower:
        if 'standards' in text_lower or 'picky' in text_lower:
            return "Creator discusses their dating experiences and how their personal growth has raised their standards, making it harder to find compatible partners."
        else:
            return "Creator shares thoughts and experiences about dating and relationships."
    elif 'coffee' in text_lower and 'friend' in text_lower:
        return "Creator recounts a conversation with a friend, sharing personal reflections and insights."
    elif 'tutorial' in text_lower or 'how to' in text_lower or 'teach' in text_lower:
        if 'how to' in text_lower:
            idx = text_lower.find('how to')
            topic = clean_text[idx:idx+80].split('.')[0].strip()
            return f"Educational content explaining {topic}."
        else:
            return "Tutorial or educational content providing instructions or guidance."
    elif 'roast' in text_lower and ('gpt' in text_lower or 'chat' in text_lower):
        return "Creator uses AI to generate a humorous roast about themselves or someone else."
    elif 'son' in text_lower and 'girlfriend' in text_lower:
        return "Parent observes and discusses their son's potential romantic relationship."
    elif 'communication' in text_lower or 'communicate' in text_lower:
        if 'pareja' in text_lower or 'relationship' in text_lower:
            return "Advice about improving communication in romantic relationships."
        else:
            return "Content focused on communication skills and interpersonal relationships."
    elif 'story' in text_lower or 'happened' in text_lower or 'remember' in text_lower:
        return "Personal storytelling sharing an experience or memory."
    elif 'dance' in text_lower or 'dancing' in text_lower:
        return "Dance performance or dance-related content."
    elif 'music' in text_lower or 'song' in text_lower:
        return "Music-related content or performance."
    elif 'review' in text_lower or 'rating' in text_lower:
        return "Review or opinion about a product, service, or content."
    else:
        # Use first sentence or first 150 chars
        if len(first_sentence) > 150:
            words = first_sentence.split()
            summary_words = []
            char_count = 0
            for word in words:
                if char_count + len(word) + 1 > 150:
                    break
                summary_words.append(word)
                char_count += len(word) + 1
            return ' '.join(summary_words) + '...'
        else:
            return first_sentence + ('.' if not first_sentence.endswith('.') else '')


def extract_screenshots(video_path, num_frames=5):
    """
    Extract screenshots from video at evenly spaced intervals.
    Returns list of PIL Image objects.
    """
    screenshots = []
    try:
        video = VideoFileClip(video_path)
        duration = video.duration
        
        timestamps = []
        if num_frames == 1:
            timestamps = [duration / 2]
        else:
            for i in range(num_frames):
                if i == 0:
                    timestamp = duration * 0.1  # 10% in (avoid very start)
                elif i == num_frames - 1:
                    timestamp = duration * 0.9  # 90% in (avoid very end)
                else:
                    timestamp = (i / (num_frames - 1)) * duration * 0.8 + duration * 0.1
                timestamps.append(timestamp)
        
        for timestamp in timestamps:
            try:
                frame = video.get_frame(timestamp)
                img = Image.fromarray(frame)
                screenshots.append(img)
            except Exception as e:
                print(f"  Warning: Could not extract frame at {timestamp:.2f}s: {e}")
        
        video.close()
    except Exception as e:
        print(f"  Warning: Could not extract screenshots: {e}")
    
    return screenshots


def image_to_base64(image):
    """
    Convert PIL Image to base64 string for API transmission.
    """
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG", quality=85)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str


def extract_text_from_frames(video_path, num_frames=5):
    if not OCR_AVAILABLE:
        return ""
    
    try:
        video = VideoFileClip(video_path)
        duration = video.duration
        extracted_text = []
        
        for i in range(num_frames):
            timestamp = (i + 0.5) * duration / num_frames
            frame = video.get_frame(timestamp)
            img = Image.fromarray(frame)
            
            try:
                text = pytesseract.image_to_string(img, lang='eng')
                if text.strip():
                    extracted_text.append(text.strip())
            except:
                pass
        
        video.close()
        return ' '.join(extracted_text)
    except Exception as e:
        return ""


def analyze_video(video_path, whisper_model_instance):
    audio_path = str(Path(video_path).with_suffix('.wav'))
    
    try:
        video = VideoFileClip(video_path)
        duration = video.duration if hasattr(video, 'duration') else 0
        
        # Extract transcription
        transcription = ""
        language = 'unknown'
        
        if video.audio is not None:
            video.audio.write_audiofile(audio_path, logger=None)
            result = whisper_model_instance.transcribe(audio_path)
            transcription = result['text']
            language = result.get('language', 'unknown')
            os.remove(audio_path)
        
        # Extract OCR text from frames
        ocr_text = extract_text_from_frames(video_path)
        
        # If no transcription, use OCR text
        if not transcription.strip() and ocr_text:
            transcription = ocr_text
            language = 'unknown'
        
        # Extract screenshots for LLM analysis
        screenshots = extract_screenshots(video_path, num_frames=5)
        
        video.close()
        
        # Get modal words (top 5 most frequent words)
        text_lower = transcription.lower() if transcription else ""
        words = [w for w in text_lower.split() if len(w) > 4]
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        topics = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
        topics_list = [word for word, freq in topics]
        
        # Pad to 5 words with empty strings
        while len(topics_list) < 5:
            topics_list.append('')
        
        # Always try LLM analysis first (context-aware)
        llm_analysis = analyze_with_llm(transcription, ocr_text, screenshots)
        
        # Prepare result with LLM data or context-aware fallbacks
        if llm_analysis:
            # Use LLM context matching (preferred method)
            description = llm_analysis.get('description', '')
            sentiment_data = llm_analysis.get('sentiment_analysis', {})
            image_description = llm_analysis.get('image_description', '')
            
            sentiment_label = sentiment_data.get('sentiment_label', 'neutral')
            sentiment_score = sentiment_data.get('sentiment_score', 0.0)
            sentiment_confidence = sentiment_data.get('confidence', 0.0)
            sentiment_reasoning = sentiment_data.get('reasoning', '')
        else:
            # Fallback: Generate context-aware description first
            if transcription and transcription.strip():
                description = generate_context_with_rules(transcription)
                if len(description) > 200:
                    description = description[:197] + '...'
            else:
                description = 'Silent video with no spoken content or visible text.'
            
            # Then analyze sentiment from the context description (not word-by-word)
            sentiment_label, sentiment_score, sentiment_confidence, sentiment_reasoning = analyze_sentiment_from_context(description, transcription)
            
            image_description = 'Visual content analysis not available (LLM analysis failed or API key not set).'
        
        return {
            'transcription': transcription,
            'language': language,
            'duration': duration,
            'description': description,
            'sentiment_label': sentiment_label,
            'sentiment_score': sentiment_score,
            'sentiment_confidence': sentiment_confidence,
            'sentiment_reasoning': sentiment_reasoning,
            'image_description': image_description,
            'modal_word_1': topics_list[0] if len(topics_list) > 0 else '',
            'modal_word_2': topics_list[1] if len(topics_list) > 1 else '',
            'modal_word_3': topics_list[2] if len(topics_list) > 2 else '',
            'modal_word_4': topics_list[3] if len(topics_list) > 3 else '',
            'modal_word_5': topics_list[4] if len(topics_list) > 4 else '',
            'word_count': len(transcription.split()) if transcription else 0
        }
        
    except Exception as e:
        print(f"Error analyzing video: {str(e)}")
        return None


def get_processed_urls(csv_path):
    processed_urls = set()
    if os.path.exists(csv_path):
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get('url'):
                        processed_urls.add(row['url'])
        except:
            pass
    return processed_urls


def main():
    history_file = 'data/Watch History.txt'
    
    if not os.path.exists(history_file):
        print(f"Error: {history_file} not found")
        sys.exit(1)
    
    # Check for OpenAI API key and warn if missing
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print("\n" + "="*70)
        print("WARNING: OPENAI_API_KEY not set!")
        print("="*70)
        print("The script will use fallback sentiment analysis (less accurate).")
        print("For better results, set your API key:")
        print("  export OPENAI_API_KEY='your-api-key-here'")
        print("  Or create a .env file with: OPENAI_API_KEY=your-key-here")
        print("="*70 + "\n")
    else:
        # Verify the key looks valid
        if len(api_key) < 20:
            print("\n" + "="*70)
            print("WARNING: OPENAI_API_KEY appears invalid (too short)")
            print("="*70 + "\n")
        else:
            print(f"✓ OpenAI API key found ({len(api_key)} chars) - will use LLM analysis for better results\n")
    
    print("Parsing watch history...")
    entries = parse_watch_history(history_file)
    print(f"Found {len(entries)} TikTok videos in watch history")
    
    csv_path = 'watch_history_analysis.csv'
    processed_urls = get_processed_urls(csv_path)
    
    if processed_urls:
        print(f"\nFound existing CSV with {len(processed_urls)} processed videos")
        resume = input("Resume from where you left off? (y/n) [default: y]: ").strip().lower() or "y"
        if resume == 'y':
            entries = [e for e in entries if e['url'] not in processed_urls]
            print(f"Resuming: {len(entries)} videos remaining to process")
    else:
        resume = "n"
    
    if not entries:
        print("\nAll videos already processed!")
        sys.exit(0)
    
    limit_input = input(f"\nProcess only first N videos? (Enter number or 'all' for all {len(entries)}) [default: all]: ").strip()
    if limit_input.lower() == 'all' or limit_input == '':
        limit = len(entries)
    else:
        try:
            limit = int(limit_input)
            if limit > len(entries):
                limit = len(entries)
        except:
            limit = len(entries)
    
    entries = entries[:limit]
    print(f"Processing {len(entries)} videos")
    
    skip_downloaded = input("\nSkip already-downloaded videos? (y/n) [default: y]: ").strip().lower() or "y"
    skip_downloaded = skip_downloaded == 'y'
    
    use_cookies = input("\nUse browser cookies for authentication? (helps with blocked/private videos) (y/n) [default: n]: ").strip().lower() or "n"
    use_cookies = use_cookies == 'y'
    if use_cookies:
        print("  Note: Will attempt to use cookies from Chrome browser")
    
    print("\nWhisper model:")
    print("  tiny   = fastest (less accurate)")
    print("  base   = balanced (recommended)")
    print("  small  = slower (more accurate)")
    model = input("Choose [default: tiny]: ").strip() or "tiny"
    
    print(f"\nLoading Whisper model ({model})...")
    whisper_model = whisper.load_model(model)
    
    download_dir = 'downloads'
    
    print(f"\nProcessing {len(entries)} videos...")
    print("="*70)
    
    file_exists = os.path.exists(csv_path)
    
    for i, entry in enumerate(entries, 1):
        date = entry['date']
        url = entry['url']
        
        print(f"\n[{i}/{len(entries)}] Processing video watched on {date.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"URL: {url}")
        
        video_path, info = download_video(url, download_dir, skip_if_exists=skip_downloaded, use_cookies=use_cookies)
        
        if not video_path:
            print("  Skipping - download failed after retries")
            # Still write a record with minimal info
            result = {
                'watch_date': date.isoformat(),
                'url': url,
                'video_id': '',
                'video_description': '',
                'creator': '',
                'duration_seconds': 0,
                'language': 'unknown',
                'transcription': '',
                'description': 'Download failed - video unavailable or requires authentication',
                'image_description': '',
                'modal_word_1': '',
                'modal_word_2': '',
                'modal_word_3': '',
                'modal_word_4': '',
                'modal_word_5': '',
                'sentiment_label': 'unknown',
                'sentiment_score': 0.0,
                'sentiment_confidence': 0.0,
                'sentiment_reasoning': 'Download failed - unable to analyze',
            }
            with open(csv_path, 'a', newline='', encoding='utf-8') as f:
                fieldnames = list(result.keys())
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
                    file_exists = True
                writer.writerow(result)
            continue
        
        print("  Analyzing...")
        analysis = analyze_video(video_path, whisper_model)
        
        if not analysis:
            print("  Skipping - analysis failed")
            continue
        
        video_id = info.get('id', '') if info else Path(video_path).stem
        video_description = info.get('description', '') if info else ''
        uploader = info.get('uploader', '') if info else ''
        
        result = {
            'watch_date': date.isoformat(),
            'url': url,
            'video_id': video_id,
            'video_description': video_description,
            'creator': uploader,
            'duration_seconds': round(analysis['duration'], 2),
            'language': analysis['language'],
            'transcription': analysis['transcription'],
            'description': analysis['description'],
            'image_description': analysis['image_description'],
            'modal_word_1': analysis['modal_word_1'],
            'modal_word_2': analysis['modal_word_2'],
            'modal_word_3': analysis['modal_word_3'],
            'modal_word_4': analysis['modal_word_4'],
            'modal_word_5': analysis['modal_word_5'],
            'sentiment_label': analysis['sentiment_label'],
            'sentiment_score': analysis['sentiment_score'],
            'sentiment_confidence': analysis['sentiment_confidence'],
            'sentiment_reasoning': analysis['sentiment_reasoning'],
        }
        
        print(f"  Sentiment: {analysis['sentiment_label']} (score: {analysis['sentiment_score']:.2f})")
        print(f"  Description: {analysis['description'][:60]}...")
        
        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            fieldnames = list(result.keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
                file_exists = True
            writer.writerow(result)
        
        if i % 10 == 0:
            print(f"\nProgress: {i}/{len(entries)} videos processed (saved to {csv_path})")
    
    print("\n" + "="*70)
    print("Analysis Complete!")
    print("="*70)
    print(f"Results saved to: {csv_path}")
    print(f"Videos processed in this session: {len(entries)}")
    print("="*70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        csv_path = 'watch_history_analysis.csv'
        print(f"\n\nProcessing cancelled. Progress saved to {csv_path}")
        sys.exit(0)

