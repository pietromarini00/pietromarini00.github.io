"""
Generate a visual word cloud from hybrid TF-IDF + Semantic scores.
Creates a PNG/JPG image to replace the About section.
"""

import matplotlib.pyplot as plt
from wordcloud import WordCloud
from pathlib import Path
import random
from . import config

def read_allowed_multigrams():
    """Read allowed multi-word phrases from single file."""
    try:
        with open(config.ALLOWED_MULTIGRAMS_FILE, 'r') as f:
            return set(line.strip().lower() for line in f if line.strip())
    except FileNotFoundError:
        return set()

def rescale_to_range(scores_dict, min_val=None, max_val=None):
    """
    Rescale scores to a target range while preserving relative proportions.
    """
    if min_val is None:
        min_val = config.MIN_SIZE
    if max_val is None:
        max_val = config.MAX_SIZE
    if not scores_dict:
        return {}
    
    values = list(scores_dict.values())
    current_min = min(values)
    current_max = max(values)
    
    if current_max == current_min:
        # All values are the same, return middle of range
        return {k: (min_val + max_val) / 2 for k in scores_dict}
    
    # Linear rescaling: new_val = min_val + (val - current_min) / (current_max - current_min) * (max_val - min_val)
    rescaled = {}
    for term, score in scores_dict.items():
        normalized = (score - current_min) / (current_max - current_min)
        rescaled[term] = min_val + normalized * (max_val - min_val)
    
    return rescaled

def sample_balanced(scores_dict, sample_ratio=None):
    """
    Sample a percentage of terms while maintaining diversity.
    """
    if sample_ratio is None:
        sample_ratio = config.SAMPLE_RATIO
        
    if not scores_dict or sample_ratio >= 1.0:
        return scores_dict
    
    items = list(scores_dict.items())
    n_sample = max(1, int(len(items) * sample_ratio))
    
    # Random sample
    random.seed(42)  # For reproducibility
    sampled = random.sample(items, n_sample)
    
    return dict(sampled)

def remove_substring_overlaps(monogram_scores, bigram_scores, trigram_scores):
    """
    Remove monograms that appear as substrings in bigrams/trigrams.
    Remove bigrams that appear as substrings in trigrams.
    
    Args:
        monogram_scores: dict of single word -> score
        bigram_scores: dict of bigram -> score
        trigram_scores: dict of trigram -> score
    
    Returns:
        Filtered monogram and bigram dicts
    """
    # Collect all multi-word phrases
    all_bigrams = set(bg.lower() for bg in bigram_scores.keys())
    all_trigrams = set(tg.lower() for tg in trigram_scores.keys())
    all_multigrams = all_bigrams | all_trigrams
    
    # Filter monograms: remove if they appear in any multigram
    filtered_monograms = {}
    for word, score in monogram_scores.items():
        word_lower = word.lower()
        # Check if word appears as a whole word in any multigram
        is_substring = any(f' {word_lower} ' in f' {mg} ' or 
                          f' {mg} '.startswith(f'{word_lower} ') or 
                          f' {mg} '.endswith(f' {word_lower}')
                          for mg in all_multigrams)
        if not is_substring:
            filtered_monograms[word] = score
    
    # Filter bigrams: remove if they appear in any trigram
    filtered_bigrams = {}
    for bigram, score in bigram_scores.items():
        bigram_lower = bigram.lower()
        is_substring = any(bigram_lower in tg.lower() for tg in all_trigrams)
        if not is_substring:
            filtered_bigrams[bigram] = score
    
    removed_mono = len(monogram_scores) - len(filtered_monograms)
    removed_bi = len(bigram_scores) - len(filtered_bigrams)
    
    if removed_mono > 0 or removed_bi > 0:
        print(f"  Removed {removed_mono} monograms and {removed_bi} bigrams (substrings of multigrams)")
    
    return filtered_monograms, filtered_bigrams

def load_semantic_data():
    """Load hybrid scoring data from the keywords file and rescale by category."""
    monogram_scores = {}
    bigram_scores = {}
    trigram_scores = {}
    
    try:
        with open(config.OUTPUT_KEYWORDS_FILE, 'r') as f:
            content = f.read()
            
        # Extract words section
        if 'Top Words (Combined Score):' in content:
            words_section = content.split('Top Words (Combined Score):')[1].split('\n\nBigrams')[0]
            for line in words_section.strip().split('\n'):
                if ':' in line and '(' in line:
                    parts = line.split(':', 1)
                    word = parts[0].strip()
                    score_str = parts[1].split('(')[0].strip()
                    try:
                        score = float(score_str)
                        monogram_scores[word] = score
                    except ValueError:
                        continue
        
        # Load allowed multigrams
        allowed_multigrams = read_allowed_multigrams()
        
        # Extract bigrams
        if 'Bigrams (Combined Score):' in content:
            bigrams_section = content.split('Bigrams (Combined Score):')[1].split('\n\nTrigrams')[0]
            
            for line in bigrams_section.strip().split('\n'):
                if ':' in line and '(' in line:
                    parts = line.split(':', 1)
                    phrase = parts[0].strip()
                    if phrase.lower() in allowed_multigrams:
                        score_str = parts[1].split('(')[0].strip()
                        try:
                            score = float(score_str)
                            bigram_scores[phrase] = score
                        except ValueError:
                            continue
        
        # Extract trigrams
        if 'Trigrams (Combined Score):' in content:
            trigrams_section = content.split('Trigrams (Combined Score):')[1]
            
            for line in trigrams_section.strip().split('\n'):
                if ':' in line and '(' in line:
                    parts = line.split(':', 1)
                    phrase = parts[0].strip()
                    if phrase.lower() in allowed_multigrams:
                        score_str = parts[1].split('(')[0].strip()
                        try:
                            score = float(score_str)
                            trigram_scores[phrase] = score
                        except ValueError:
                            continue
        
        # Rescale each category independently to 0.2-1.0 range
        print(f"Rescaling scores by category:")
        print(f"  Monograms: {len(monogram_scores)} terms")
        print(f"  Bigrams: {len(bigram_scores)} terms")
        print(f"  Trigrams: {len(trigram_scores)} terms")
        
        # Remove substring overlaps
        print(f"\nRemoving substring overlaps...")
        monogram_scores, bigram_scores = remove_substring_overlaps(
            monogram_scores, bigram_scores, trigram_scores
        )
        
        print(f"After filtering:")
        print(f"  Monograms: {len(monogram_scores)} terms")
        print(f"  Bigrams: {len(bigram_scores)} terms")
        print(f"  Trigrams: {len(trigram_scores)} terms")
        
        # Sample from each category
        monogram_sampled = sample_balanced(monogram_scores, sample_ratio=config.SAMPLE_RATIO)
        bigram_sampled = sample_balanced(bigram_scores, sample_ratio=config.SAMPLE_RATIO)
        trigram_sampled = sample_balanced(trigram_scores, sample_ratio=config.SAMPLE_RATIO)
        
        print(f"\nSampled {int(config.SAMPLE_RATIO*100)}% randomly from each category:")
        print(f"  Monograms: {len(monogram_sampled)} terms")
        print(f"  Bigrams: {len(bigram_sampled)} terms")
        print(f"  Trigrams: {len(trigram_sampled)} terms")
        
        monogram_rescaled = rescale_to_range(monogram_sampled, min_val=config.MIN_SIZE, max_val=config.MAX_SIZE)
        bigram_rescaled = rescale_to_range(bigram_sampled, min_val=config.MIN_SIZE, max_val=config.MAX_SIZE)
        trigram_rescaled = rescale_to_range(trigram_sampled, min_val=config.MIN_SIZE, max_val=config.MAX_SIZE)
        
        # Combine all rescaled scores
        word_scores = {}
        word_scores.update(monogram_rescaled)
        word_scores.update(bigram_rescaled)
        word_scores.update(trigram_rescaled)
        
        return word_scores
        
    except FileNotFoundError:
        print(f"Error: {config.OUTPUT_KEYWORDS_FILE} not found!")
        print("Run generate_wordcloud.py first.")
        return {}
    except Exception as e:
        print(f"Error loading data: {e}")
        return {}
        return {}
    except Exception as e:
        print(f"Error loading data: {e}")
        return {}
    
    return word_scores

def generate_wordcloud_image(output_file=None):
    """Generate and save word cloud visualization."""
    
    if output_file is None:
        output_file = config.OUTPUT_IMAGE_PNG
    
    # Load data
    word_scores = load_semantic_data()
    
    if not word_scores:
        print("No word data found!")
        return
    
    print(f"Generating word cloud with {len(word_scores)} terms...")
    
    # Create word cloud with minimalist style
    wordcloud = WordCloud(
        width=config.IMAGE_WIDTH,
        height=config.IMAGE_HEIGHT,
        background_color=config.BACKGROUND_COLOR,
        color_func=lambda *args, **kwargs: config.TEXT_COLOR,
        font_path=None,  # Use default system font
        relative_scaling=0.5,
        min_font_size=config.MIN_FONT_SIZE,
        max_font_size=config.MAX_FONT_SIZE,
        prefer_horizontal=0.8,
        margin=20,
        colormap='Greys',
        mode='RGBA'
    ).generate_from_frequencies(word_scores)
    
    # Create figure
    plt.figure(figsize=(14, 6), facecolor='white')
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout(pad=0)
    
    # Save as PNG (high quality)
    plt.savefig(output_file, dpi=config.IMAGE_DPI, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print(f"\nWord cloud saved to: {output_file}")
    
    # Also save as JPG
    jpg_file = str(Path(output_file).with_suffix('.jpg'))
    plt.savefig(jpg_file, dpi=config.IMAGE_DPI, bbox_inches='tight', 
                facecolor='white', edgecolor='none', format='jpg')
    print(f"Word cloud also saved to: {jpg_file}")
    
    plt.close()

def main():
    print("="*70)
    print("SEMANTIC WORD CLOUD VISUALIZATION GENERATOR")
    print("="*70 + "\n")
    
    # Generate word cloud image
    generate_wordcloud_image(config.OUTPUT_IMAGE_PNG)
    
    print(f"\n{'='*70}")
    print(f"Done! Use {config.OUTPUT_IMAGE_PNG} in your website.")
    print("Adjust alpha parameter in config.py to tune")
    print("the balance between TF-IDF uniqueness and semantic relevance.")

if __name__ == "__main__":
    main()
