# Word Cloud Generator

Generate semantic word clouds from your research documents using hybrid TF-IDF + semantic similarity scoring.

## Quick Start

```bash
python generate_wordcloud.py
```

## Configuration

Edit `wordcloud_generator/config.py` to customize:

### Scoring Parameters
- `ALPHA`: Balance between TF-IDF (uniqueness) and semantic similarity (relevance)
  - `0.0` = Pure semantic similarity
  - `0.5` = Balanced (default)
  - `1.0` = Pure TF-IDF
- `DEDUPLICATE_SYNONYMS`: Remove similar terms (default: True)
- `SYNONYM_THRESHOLD`: Similarity threshold for deduplication (default: 0.85)

### Visualization
- `SAMPLE_RATIO`: Fraction of terms to display (default: 0.5 = 50%)
- `MIN_SIZE` / `MAX_SIZE`: Size range for rescaling (default: 0.2-1.0)
- `IMAGE_WIDTH` / `IMAGE_HEIGHT`: Output dimensions (default: 1400×600)

### Content Filtering
- `TARGET_CONCEPTS`: List of domain-specific concepts for semantic scoring
- `allowed_multigrams.txt`: Whitelist of multi-word phrases to include

## Command-Line Options

```bash
# Basic usage with defaults
python generate_wordcloud.py

# Adjust TF-IDF weight (60% TF-IDF, 40% semantic)
python generate_wordcloud.py --alpha 0.6

# Show more terms (70% of all terms)
python generate_wordcloud.py --sample 0.7

# Disable synonym deduplication
python generate_wordcloud.py --no-deduplicate

# Change deduplication threshold
python generate_wordcloud.py --threshold 0.90

# Combine options
python generate_wordcloud.py --alpha 0.6 --sample 0.7 --threshold 0.90
```

## Input Files

Place your documents in `wordcloud_input/`:
- `real_input.txt` - Your main content (CV, research statement, etc.)
- `allowed_multigrams.txt` - Whitelist of multi-word phrases

## Output Files

- `wordcloud_semantic.png` - Main word cloud image
- `wordcloud_semantic.jpg` - JPG version
- `wordcloud_keywords_semantic.txt` - Term scores summary
- `wordcloud_detailed_breakdown.txt` - Complete scoring breakdown

## How It Works

### 1. TF-IDF Scoring
Measures how unique each term is to your documents compared to general English text.

### 2. Semantic Similarity
Uses sentence-transformers to compute how semantically relevant each term is to:
- Your full document content (document similarity mode), or
- Predefined target concepts (target concepts mode)

### 3. Hybrid Combination
```
final_score = α × percentile(TF-IDF) + (1-α) × percentile(semantic)
```

Percentile normalization ensures both metrics contribute fairly regardless of their different scales.

### 4. Deduplication
Removes terms that are too similar (e.g., "data" vs "dataset") to reduce redundancy.

### 5. Category Balancing
Rescales monograms, bigrams, and trigrams independently to ensure balanced representation.

### 6. Sampling
Randomly samples terms to avoid clutter while maintaining diversity.

## Module Structure

```
wordcloud_generator/
├── __init__.py           # Package initialization
├── config.py             # Configuration settings
├── semantic_scorer.py    # TF-IDF + semantic scoring
└── visualizer_semantic.py # Word cloud visualization

generate_wordcloud.py     # Main CLI entry point
wordcloud_input/          # Input documents directory
```

## Dependencies

- sentence-transformers
- scikit-learn
- matplotlib
- wordcloud
- numpy

Install with:
```bash
pip install sentence-transformers scikit-learn matplotlib wordcloud numpy
```

## Customization Tips

### For More Technical Terms
- Increase `ALPHA` (e.g., 0.7) to emphasize TF-IDF uniqueness

### For More Readable Clouds
- Increase `SAMPLE_RATIO` (e.g., 0.3) to show fewer terms
- Decrease `SYNONYM_THRESHOLD` (e.g., 0.75) to remove more similar terms

### For Domain-Specific Focus
- Edit `TARGET_CONCEPTS` in `config.py` with your research areas
- Set `USE_DOCUMENT_SIMILARITY = False` to use target concepts mode
