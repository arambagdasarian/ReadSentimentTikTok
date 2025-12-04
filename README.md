# TikTok Sentiment Analysis

Analyze TikTok watch history and perform sentiment analysis on video content using AI.

## Quick Start

1. Initilize the virtual environment and install the dependencies:

   ```bas
   uv venv
   ```
   
   ```bas
   uv pip install 
   ```

   ```bas
   uv pip install pip
   ```

   ```bas
   uv pip install -r requirements.txt
   ```

2. **Set OpenAI API key (optional but recommended):**
   
   Create a `.env` file in the project root:
   ```bash
   echo 'OPENAI_API_KEY=sk-your-key-here' > .env
   ```
   
   Or set it temporarily:
   ```bash
   export OPENAI_API_KEY="sk-your-key-here"
   ```
   
   **Get a free API key:** Sign up at [platform.openai.com](https://platform.openai.com) - new accounts get $5 free credits.

3. **Run the analysis:**
   ```bash
   uv run process_watch_history.py
   ```

## Requirements

- Python 3.8+
- FFmpeg
- OpenAI API key (optional - script works without it using fallback analysis)
- TikTok watch history export in `data/Watch History.txt`

## Output

Results are saved to `watch_history_analysis.csv`. Videos are downloaded to `downloads/` (skips already downloaded files).

## Note

The script works **without an API key** using context-aware fallback analysis, but LLM analysis provides more accurate sentiment detection and better descriptions.
