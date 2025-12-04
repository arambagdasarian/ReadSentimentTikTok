# TikTok Watch History Sentiment Analysis Pipeline
## Technical Documentation

### Executive Summary

This document describes a comprehensive data processing pipeline that analyzes TikTok watch history to extract sentiment, topics, and contextual insights from video content. The system combines video downloading, speech-to-text transcription, optical character recognition (OCR), and natural language processing to provide detailed analytics on user viewing patterns and content preferences.

---

## System Architecture

### Overview
The pipeline consists of five main components working in sequence:

1. **Data Ingestion**: Parse TikTok watch history export files
2. **Content Acquisition**: Download videos using yt-dlp
3. **Content Analysis**: Extract audio/visual information using Whisper and OCR
4. **Sentiment & Topic Analysis**: Process transcriptions for insights
5. **Data Export**: Save structured results to CSV format

### Technology Stack
- **Python 3.x**: Core programming language
- **yt-dlp**: Video downloading and metadata extraction
- **OpenAI Whisper**: Speech-to-text transcription
- **Tesseract OCR**: Text extraction from video frames
- **MoviePy**: Video processing and audio extraction
- **PIL (Pillow)**: Image processing for OCR

---

## Detailed Component Analysis

### 1. Data Ingestion Module (`parse_watch_history`)

**Purpose**: Converts TikTok's proprietary watch history format into structured data.

**Input Format**: Text file with alternating date and URL entries:
```
Date: 2024-01-15 14:30:22 UTC
Link: https://www.tiktok.com/@user/video/1234567890
```

**Processing Logic**:
- Reads file line by line using UTF-8 encoding
- Maintains state tracking for current date context
- Parses timestamps using `datetime.strptime()` with UTC format
- Normalizes URLs by replacing `tiktokv.com` with `tiktok.com`
- Creates structured entries with date-URL pairs

**Output**: List of dictionaries containing `{'date': datetime_object, 'url': string}`

**Error Handling**: Gracefully handles malformed dates by setting `current_date = None`

### 2. Content Acquisition Module (`download_video`)

**Purpose**: Downloads TikTok videos and extracts metadata using yt-dlp.

**Configuration**:
- Output template: `%(id)s.%(ext)s` (uses TikTok video ID as filename)
- Format selection: `'best'` (highest quality available)
- Quiet mode enabled to reduce console output
- Directory creation with `parents=True, exist_ok=True`

**Optimization Features**:
- **Skip existing files**: Checks file existence before downloading
- **Metadata extraction**: Retrieves video info without downloading first
- **Error resilience**: Returns `None, None` on failure, allowing pipeline continuation

**Return Values**: Tuple of `(video_path, metadata_dict)` or `(None, None)` on failure

### 3. Content Analysis Module

#### 3.1 Optical Character Recognition (`extract_text_from_frames`)

**Purpose**: Extracts text overlays from video frames for content without audio.

**Methodology**:
- Samples 5 frames evenly distributed across video duration
- Converts video frames to PIL Image objects
- Applies Tesseract OCR with English language model
- Aggregates text from all frames

**Frame Sampling Algorithm**:
```python
timestamp = (i + 0.5) * duration / num_frames
```
This ensures frames are sampled from the middle of each time segment, avoiding transition artifacts.

**Fallback Behavior**: Returns empty string if OCR unavailable or processing fails

#### 3.2 Video Analysis Engine (`analyze_video`)

**Purpose**: Main analysis orchestrator combining audio transcription, OCR, sentiment analysis, and topic extraction.

**Processing Flow**:

1. **Audio Detection & Processing**:
   - Checks for `video.audio` attribute
   - Extracts audio to temporary WAV file
   - Processes with Whisper model
   - Cleans up temporary files

2. **Silent Video Handling**:
   - Falls back to OCR text extraction
   - Generates word frequency analysis
   - Creates appropriate context summaries

3. **Transcription Fallback**:
   - If Whisper returns empty transcription, attempts OCR
   - Combines multiple content sources for comprehensive analysis

### 4. Natural Language Processing Module

#### 4.1 Sentiment Analysis

**Methodology**: Rule-based sentiment classification using predefined word lists.

**Word Lists**:
- **Positive**: ['love', 'great', 'amazing', 'awesome', 'happy', 'best', 'wonderful', 'good', 'beautiful']
- **Negative**: ['hate', 'bad', 'terrible', 'awful', 'sad', 'worst', 'horrible', 'wrong', 'ugly']

**Classification Logic**:
- Count occurrences of positive/negative words in transcription
- Assign sentiment based on majority: positive > negative → 'positive'
- Default to 'neutral' for ties or no matches

**Limitations**: Simple keyword matching; doesn't account for context, sarcasm, or negation

#### 4.2 Topic Extraction

**Algorithm**: Frequency-based topic identification

**Process**:
1. Filter words longer than 4 characters (removes common stop words)
2. Convert to lowercase for case-insensitive matching
3. Count word frequencies using dictionary
4. Sort by frequency and select top 5 words
5. Return as comma-separated string

**Rationale**: Longer words typically carry more semantic meaning than short function words

#### 4.3 Context Generation

**Purpose**: Generate human-readable summaries of video content using pattern matching.

**Pattern Recognition System**:
The system uses keyword-based pattern matching to identify common TikTok content categories:

- **Dating/Relationships**: Keywords 'dating', 'relationship', 'standards', 'picky'
- **Educational Content**: Keywords 'tutorial', 'how to', 'teach'
- **AI-Generated Content**: Keywords 'roast', 'gpt', 'chat'
- **Personal Stories**: Keywords 'story', 'happened', 'remember'
- **Performance Content**: Keywords 'dance', 'music', 'song'
- **Reviews**: Keywords 'review', 'rating'

**Fallback Strategy**:
1. Use first sentence if single sentence content
2. Truncate to 150 characters with ellipsis for long content
3. Ensure proper punctuation in summaries

### 5. Data Persistence & Resume Functionality

#### 5.1 Progress Tracking (`get_processed_urls`)

**Purpose**: Enable resume functionality for interrupted processing sessions.

**Implementation**:
- Reads existing CSV file to extract processed URLs
- Uses Python `set` for O(1) lookup performance
- Gracefully handles missing or corrupted CSV files

#### 5.2 Incremental Processing

**Strategy**: Filter out already-processed entries before main processing loop

**Benefits**:
- Saves processing time on large datasets
- Prevents duplicate entries in output
- Allows for iterative analysis refinement

---

## Pipeline Execution Flow

### 1. Initialization Phase
- Validate input file existence
- Load existing progress from CSV
- Prompt user for processing parameters:
  - Number of videos to process
  - Skip downloaded videos option
  - Whisper model selection (tiny/base/small)

### 2. Model Loading
- Initialize Whisper model based on user selection
- Model trade-offs:
  - **Tiny**: Fastest processing, lower accuracy
  - **Base**: Balanced performance (recommended)
  - **Small**: Slower processing, higher accuracy

### 3. Main Processing Loop
For each video entry:
1. Display progress information
2. Download video and extract metadata
3. Analyze content (transcription + OCR)
4. Perform sentiment and topic analysis
5. Generate contextual summary
6. Save results to CSV immediately
7. Display progress every 10 videos

### 4. Error Handling & Recovery
- Individual video failures don't stop pipeline
- Progress saved incrementally to prevent data loss
- Keyboard interrupt (Ctrl+C) handled gracefully
- Detailed error messages for debugging

---

## Output Data Schema

The pipeline generates a CSV file with the following columns:

| Column | Type | Description |
|--------|------|-------------|
| `watch_date` | ISO DateTime | When user watched the video |
| `url` | String | Original TikTok URL |
| `video_id` | String | TikTok video identifier |
| `description` | String | Video description from metadata |
| `creator` | String | Content creator username |
| `duration_seconds` | Float | Video length in seconds |
| `language` | String | Detected language from Whisper |
| `transcription` | String | Full text transcription |
| `context` | String | Generated content summary |
| `topics` | String | Comma-separated topic keywords |
| `sentiment` | String | positive/negative/neutral |

---

## Performance Considerations

### Scalability
- **Memory Usage**: Videos processed individually, not batch-loaded
- **Storage**: Temporary audio files cleaned up immediately
- **Processing Time**: Approximately 30-60 seconds per video depending on:
  - Video length
  - Whisper model size
  - OCR complexity
  - Network download speed

### Optimization Strategies
1. **Skip existing downloads**: Reduces redundant network operations
2. **Incremental CSV writing**: Prevents data loss on interruption
3. **Model caching**: Whisper model loaded once per session
4. **Temporary file cleanup**: Prevents disk space accumulation

---

## Limitations & Future Improvements

### Current Limitations
1. **Sentiment Analysis**: Simple keyword-based approach lacks contextual understanding
2. **Topic Extraction**: Frequency-based method may miss semantic relationships
3. **Language Support**: OCR configured for English only
4. **Content Types**: Pattern matching limited to predefined categories

### Potential Enhancements
1. **Advanced NLP**: Integration with transformer-based models (BERT, RoBERTa)
2. **Multi-language Support**: Expand OCR and analysis to multiple languages
3. **Visual Analysis**: Computer vision for scene/object recognition
4. **Temporal Analysis**: Track sentiment/topic trends over time
5. **User Behavior Modeling**: Analyze viewing patterns and preferences

---

## Technical Requirements

### System Dependencies
- Python 3.7+
- FFmpeg (for MoviePy video processing)
- Tesseract OCR engine
- Sufficient disk space for video downloads
- Internet connection for video downloading

### Python Package Dependencies
```
yt-dlp>=2023.1.6
openai-whisper>=20230314
moviepy>=1.0.3
Pillow>=9.0.0
pytesseract>=0.3.10
```

---

## Conclusion

This pipeline provides a comprehensive solution for analyzing TikTok watch history, combining multiple AI/ML technologies to extract meaningful insights from video content. The modular design allows for easy maintenance and future enhancements, while the robust error handling ensures reliable operation on large datasets.

The system successfully bridges the gap between raw video content and structured analytical data, enabling deeper understanding of user viewing patterns and content preferences in the TikTok ecosystem.

