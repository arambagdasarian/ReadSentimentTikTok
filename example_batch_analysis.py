#!/usr/bin/env python3
"""
Example: Batch TikTok Video Analysis

This script demonstrates how to analyze multiple TikTok videos
from a list of URLs and save results to CSV.
"""

from tiktok_analyzer import TikTokAnalyzer
import sys

def main():
    # Sample TikTok URLs (replace with your own)
    urls = [
        "https://www.tiktok.com/@ddlovato/video/7481747639336176939?lang=en",
        # "https://www.tiktok.com/@username/video/1234567890",
        # "https://www.tiktok.com/@username/video/0987654321",
    ]
    
    # Or read from a file
    import_from_file = input("Import URLs from file? (y/n) [default: n]: ").strip().lower() == 'y'
    
    if import_from_file:
        filename = input("Enter filename (one URL per line): ").strip()
        try:
            with open(filename, 'r') as f:
                urls = [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            print(f"Error: File '{filename}' not found")
            sys.exit(1)
    else:
        # Manually enter URLs
        print("\nEnter TikTok URLs (one per line, empty line to finish):")
        urls = []
        while True:
            url = input().strip()
            if not url:
                break
            urls.append(url)
    
    if not urls:
        print("No URLs provided. Exiting.")
        sys.exit(0)
    
    print(f"\n📋 Found {len(urls)} video(s) to analyze\n")
    
    # Configuration
    use_llm = input("Use LLM for advanced analysis? (y/n) [default: n]: ").strip().lower() == 'y'
    api_key = None
    provider = 'openai'
    
    if use_llm:
        provider = input("LLM provider (openai/anthropic) [default: openai]: ").strip() or "openai"
        api_key = input(f"Enter {provider.upper()} API key: ").strip()
        if not api_key:
            print("⚠ No API key provided, falling back to basic analysis")
            use_llm = False
    
    # Initialize analyzer
    print("\n🚀 Initializing analyzer...")
    analyzer = TikTokAnalyzer(
        output_dir='batch_analysis_output',
        whisper_model='tiny'  # Fast model for batch processing
    )
    
    # Process each video
    successful = 0
    failed = 0
    
    for i, url in enumerate(urls, 1):
        print(f"\n{'='*70}")
        print(f"Processing video {i}/{len(urls)}")
        print(f"{'='*70}")
        
        try:
            result = analyzer.analyze_video(
                url=url,
                use_llm=use_llm,
                api_key=api_key,
                provider=provider
            )
            
            if result:
                successful += 1
            else:
                failed += 1
                
        except KeyboardInterrupt:
            print("\n\n⚠ Batch processing interrupted by user")
            break
        except Exception as e:
            print(f"\n✗ Error processing {url}: {str(e)}")
            failed += 1
            continue
    
    # Save results
    if analyzer.results:
        csv_filename = 'batch_tiktok_analysis.csv'
        analyzer.save_to_csv(csv_filename)
        
        print("\n" + "="*70)
        print("🎉 Batch Analysis Complete!")
        print("="*70)
        print(f"✓ Successful: {successful}")
        print(f"✗ Failed: {failed}")
        print(f"📊 Results saved to: {csv_filename}")
        print("="*70 + "\n")
    else:
        print("\n⚠ No videos were successfully analyzed.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
        sys.exit(0)

