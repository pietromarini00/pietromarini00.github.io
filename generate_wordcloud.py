#!/usr/bin/env python3
"""
Main entry point for word cloud generation.
Run this script to generate a word cloud from your documents.

Usage:
    python generate_wordcloud.py [--mode semantic|tfidf] [--alpha 0.5] [--sample 0.5]
"""

import sys
import argparse
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from wordcloud_generator.semantic_scorer import main as semantic_scorer_main
from wordcloud_generator.visualizer_semantic import main as visualizer_main
from wordcloud_generator import config


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate word cloud from documents with hybrid TF-IDF + semantic scoring'
    )
    
    parser.add_argument(
        '--mode',
        choices=['semantic', 'tfidf'],
        default='semantic',
        help='Scoring mode (default: semantic)'
    )
    
    parser.add_argument(
        '--alpha',
        type=float,
        default=config.ALPHA,
        help=f'TF-IDF weight: 0=only semantic, 1=only TF-IDF (default: {config.ALPHA})'
    )
    
    parser.add_argument(
        '--sample',
        type=float,
        default=config.SAMPLE_RATIO,
        help=f'Fraction of terms to display (default: {config.SAMPLE_RATIO})'
    )
    
    parser.add_argument(
        '--deduplicate',
        action='store_true',
        default=config.DEDUPLICATE_SYNONYMS,
        help='Remove similar terms (default: enabled)'
    )
    
    parser.add_argument(
        '--no-deduplicate',
        dest='deduplicate',
        action='store_false',
        help='Keep all terms including similar ones'
    )
    
    parser.add_argument(
        '--threshold',
        type=float,
        default=config.SYNONYM_THRESHOLD,
        help=f'Similarity threshold for deduplication (default: {config.SYNONYM_THRESHOLD})'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Update config with command-line arguments
    config.ALPHA = args.alpha
    config.SAMPLE_RATIO = args.sample
    config.DEDUPLICATE_SYNONYMS = args.deduplicate
    config.SYNONYM_THRESHOLD = args.threshold
    
    print("="*70)
    print("WORD CLOUD GENERATOR")
    print("="*70)
    print(f"\nMode: {args.mode}")
    print(f"Alpha (TF-IDF weight): {config.ALPHA}")
    print(f"Sample ratio: {config.SAMPLE_RATIO}")
    print(f"Deduplicate synonyms: {config.DEDUPLICATE_SYNONYMS}")
    if config.DEDUPLICATE_SYNONYMS:
        print(f"Synonym threshold: {config.SYNONYM_THRESHOLD}")
    print()
    
    # Step 1: Score terms
    print("Step 1: Scoring terms...")
    if args.mode == 'semantic':
        semantic_scorer_main()
    else:
        print("TF-IDF mode not yet implemented in modular version")
        sys.exit(1)
    
    # Step 2: Generate visualization
    print("\nStep 2: Generating visualization...")
    visualizer_main()
    
    print("\n" + "="*70)
    print("DONE!")
    print("="*70)
    print(f"\nGenerated files:")
    print(f"  - {config.OUTPUT_IMAGE_PNG}")
    print(f"  - {config.OUTPUT_IMAGE_JPG}")
    print(f"  - {config.OUTPUT_KEYWORDS_FILE}")
    print(f"  - {config.OUTPUT_BREAKDOWN_FILE}")
    print()


if __name__ == "__main__":
    main()
