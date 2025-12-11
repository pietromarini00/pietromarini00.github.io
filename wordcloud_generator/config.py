"""
Configuration for word cloud generation
"""

# Scoring weights
ALPHA = 0.5  # TF-IDF weight (0.5 = 50% TF-IDF, 50% semantic)

# Filtering options
USE_ALLOWED_FILTER = False  # If True, only score phrases in allowed_multigrams.txt
USE_DOCUMENT_SIMILARITY = True  # If True, compute similarity against full document

# Deduplication
DEDUPLICATE_SYNONYMS = True  # If True, remove terms too similar to higher-scored terms
SYNONYM_THRESHOLD = 0.85  # Cosine similarity threshold for considering terms as synonyms

# Frequency thresholds
MIN_FREQ = 2  # Minimum frequency for n-grams

# Output sizes
TOP_WORDS = 40
TOP_BIGRAMS = 15
TOP_TRIGRAMS = 15

# Visualization
SAMPLE_RATIO = 0.5  # Fraction of terms to display (0.5 = 50%)
MIN_SIZE = 0.2  # Minimum size in rescaled range
MAX_SIZE = 1.0  # Maximum size in rescaled range

# Image settings
IMAGE_WIDTH = 1400
IMAGE_HEIGHT = 600
IMAGE_DPI = 150
BACKGROUND_COLOR = 'white'
TEXT_COLOR = '#333333'
MIN_FONT_SIZE = 10
MAX_FONT_SIZE = 80

# Target concepts for semantic similarity (customize for your domain!)
TARGET_CONCEPTS = [
    "scientific data management",
    "machine learning research",
    "data infrastructure systems",
    "fair data principles",
    "semantic data integration",
    "reproducible computational science",
    "metadata extraction and provenance",
    "research engineering",
]

# Input/output paths
INPUT_DIR = "wordcloud_input"
OUTPUT_DIR = "wordcloud_output"
ALLOWED_MULTIGRAMS_FILE = "wordcloud_input/allowed_multigrams.txt"
OUTPUT_KEYWORDS_FILE = "wordcloud_output/wordcloud_keywords_semantic.txt"
OUTPUT_BREAKDOWN_FILE = "wordcloud_output/wordcloud_detailed_breakdown.txt"
OUTPUT_IMAGE_PNG = "wordcloud_output/wordcloud_semantic.png"
OUTPUT_IMAGE_JPG = "wordcloud_output/wordcloud_semantic.jpg"
