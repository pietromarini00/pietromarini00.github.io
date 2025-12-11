"""
Generate a word cloud from text files in the wordcloud_input directory.
Uses TF-IDF scoring against general English corpus to find unique terms.
Extracts meaningful n-grams (bigrams/trigrams) for entities.
"""

import os
from collections import Counter, defaultdict
import re
from pathlib import Path

# Common stop words to exclude
STOP_WORDS = {
    'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i', 'it', 'for', 
    'not', 'on', 'with', 'he', 'as', 'you', 'do', 'at', 'this', 'but', 'his', 'by',
    'from', 'they', 'we', 'say', 'her', 'she', 'or', 'an', 'will', 'my', 'one', 
    'all', 'would', 'there', 'their', 'what', 'so', 'up', 'out', 'if', 'about', 
    'who', 'get', 'which', 'go', 'me', 'when', 'make', 'can', 'like', 'time', 
    'no', 'just', 'him', 'know', 'take', 'people', 'into', 'year', 'your', 'good',
    'some', 'could', 'them', 'see', 'other', 'than', 'then', 'now', 'look', 'only',
    'come', 'its', 'over', 'think', 'also', 'back', 'after', 'use', 'two', 'how',
    'our', 'work', 'first', 'well', 'way', 'even', 'new', 'want', 'because', 'any',
    'these', 'give', 'day', 'most', 'us', 'is', 'was', 'are', 'been', 'has', 'had',
    'were', 'said', 'did', 'having', 'may', 'should', 'am', 'being', 'such', 'where',
    'through', 'here', 'more', 'very', 'each', 'those', 'while', 'both', 'between',
    'during', 'before', 'under', 'around', 'within', 'without', 'toward', 'across',
    'yet', 'still', 'never', 'ever', 'often', 'always', 'sometimes', 'usually',
}

def read_text_files(directory):
    """Read all text files from the specified directory."""
    texts = []
    input_dir = Path(directory)
    
    if not input_dir.exists():
        print(f"Directory {directory} does not exist!")
        return ""
    
    for file_path in input_dir.glob('*'):
        if file_path.suffix.lower() in ['.txt', '.md']:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    texts.append(f.read())
                print(f"Read: {file_path.name}")
            except Exception as e:
                print(f"Error reading {file_path.name}: {e}")
    
    return ' '.join(texts)

def extract_words(text):
    """Extract words, removing punctuation and numbers."""
    words = re.findall(r'\b[a-z]+(?:-[a-z]+)?\b', text.lower())
    return words

def extract_ngrams(text, n=2):
    """Extract n-grams from text."""
    text_lower = text.lower()
    text_clean = re.sub(r'[^\w\s-]', ' ', text_lower)
    words = text_clean.split()
    
    filtered_words = [w for w in words if w not in STOP_WORDS and len(w) > 2]
    
    ngrams = []
    for i in range(len(filtered_words) - n + 1):
        ngram = ' '.join(filtered_words[i:i+n])
        ngrams.append(ngram)
    
    return ngrams

def get_general_corpus_idf():
    """Return approximate IDF values for common English words."""
    common_words = {
        'said', 'get', 'make', 'know', 'go', 'see', 'come', 'think', 'take',
        'want', 'use', 'find', 'give', 'tell', 'ask', 'seem', 'feel', 'try', 'leave',
        'call', 'need', 'become', 'show', 'put', 'mean', 'keep', 'let', 'begin',
        'help', 'talk', 'turn', 'start', 'run', 'move', 'live', 'believe', 'bring',
        'happen', 'write', 'provide', 'sit', 'stand', 'lose', 'pay', 'meet', 'include',
        'continue', 'set', 'learn', 'change', 'lead', 'understand', 'watch', 'follow',
        'stop', 'create', 'speak', 'read', 'allow', 'add', 'spend', 'grow', 'open',
        'walk', 'win', 'offer', 'remember', 'love', 'consider', 'appear', 'buy', 'wait',
        'serve', 'die', 'send', 'expect', 'build', 'stay', 'fall', 'cut', 'reach',
        'kill', 'remain', 'suggest', 'raise', 'pass', 'sell', 'require', 'report',
        'decide', 'pull', 'thing', 'person', 'year', 'way', 'day', 'man', 'world',
        'life', 'hand', 'part', 'child', 'eye', 'woman', 'place', 'week',
        'case', 'point', 'government', 'company', 'number', 'group', 'problem', 'fact',
        'able', 'bad', 'best', 'better', 'big', 'black', 'certain', 'clear', 'different',
        'early', 'easy', 'economic', 'federal', 'free', 'full', 'great', 'hard',
        'high', 'human', 'important', 'international', 'large', 'late', 'little', 'local',
        'long', 'low', 'major', 'national', 'old', 'only', 'own',
        'political', 'possible', 'public', 'real', 'recent', 'right', 'small', 'social',
        'special', 'strong', 'sure', 'true', 'white', 'whole', 'young',
        'program', 'question', 'information', 'process', 'development', 'name', 'area',
        'level', 'order', 'result', 'type', 'example', 'service', 'member',
    }
    
    return defaultdict(lambda: 5.0, {word: 1.0 for word in common_words})

def calculate_tfidf_scores(word_counts, total_words):
    """Calculate TF-IDF scores for words."""
    general_idf = get_general_corpus_idf()
    
    tfidf_scores = {}
    for word, count in word_counts.items():
        tf = count / total_words
        idf = general_idf[word]
        tfidf_scores[word] = tf * idf
    
    return tfidf_scores

def generate_wordcloud_data(input_dir='wordcloud_input', top_n=40):
    """Generate word frequency data with TF-IDF and n-grams."""
    text = read_text_files(input_dir)
    
    if not text:
        print("No text found to process!")
        return {}, {}, {}
    
    print(f"Text length: {len(text)} characters\n")
    
    words = extract_words(text)
    print(f"Total words extracted: {len(words)}")
    
    filtered_words = [w for w in words if w not in STOP_WORDS and len(w) > 2]
    print(f"After removing stop words: {len(filtered_words)}")
    
    word_counts = Counter(filtered_words)
    print(f"Unique words: {len(word_counts)}")
    
    tfidf_scores = calculate_tfidf_scores(word_counts, len(filtered_words))
    
    meaningful_tfidf = {word: score for word, score in tfidf_scores.items() 
                        if word_counts[word] >= 2}
    
    top_tfidf_words = dict(sorted(meaningful_tfidf.items(), 
                                   key=lambda x: x[1], 
                                   reverse=True)[:top_n])
    
    bigrams = extract_ngrams(text, n=2)
    trigrams = extract_ngrams(text, n=3)
    
    bigram_counts = Counter(bigrams)
    trigram_counts = Counter(trigrams)
    
    meaningful_bigrams = {ng: count for ng, count in bigram_counts.items() if count >= 2}
    meaningful_trigrams = {ng: count for ng, count in trigram_counts.items() if count >= 2}
    
    print(f"\nTF-IDF scored words: {len(meaningful_tfidf)}")
    print(f"Meaningful bigrams: {len(meaningful_bigrams)}")
    print(f"Meaningful trigrams: {len(meaningful_trigrams)}\n")
    
    top_bigrams = dict(Counter(meaningful_bigrams).most_common(15))
    top_trigrams = dict(Counter(meaningful_trigrams).most_common(15))
    
    return top_tfidf_words, top_bigrams, top_trigrams

def main():
    print("Generating word cloud data with TF-IDF and n-grams...\n")
    
    tfidf_words, bigrams, trigrams = generate_wordcloud_data(input_dir='wordcloud_input', top_n=40)
    
    if not tfidf_words and not bigrams and not trigrams:
        print("\nNo terms found. Make sure to add text files to the 'wordcloud_input' directory.")
        return
    
    print("\n" + "="*70)
    print("TOP UNIQUE WORDS (TF-IDF Scored)")
    print("="*70 + "\n")
    
    sorted_tfidf = sorted(tfidf_words.items(), key=lambda x: x[1], reverse=True)
    
    for i, (word, score) in enumerate(sorted_tfidf, 1):
        print(f"{i:2d}. {word:25s} (TF-IDF: {score:.4f})")
    
    print("\n" + "="*70)
    print("TOP BIGRAMS (2-word phrases)")
    print("="*70 + "\n")
    
    sorted_bigrams = sorted(bigrams.items(), key=lambda x: x[1], reverse=True)
    for i, (phrase, count) in enumerate(sorted_bigrams, 1):
        print(f"{i:2d}. {phrase:35s} ({count:2d} occurrences)")
    
    print("\n" + "="*70)
    print("TOP TRIGRAMS (3-word phrases)")
    print("="*70 + "\n")
    
    sorted_trigrams = sorted(trigrams.items(), key=lambda x: x[1], reverse=True)
    for i, (phrase, count) in enumerate(sorted_trigrams, 1):
        print(f"{i:2d}. {phrase:40s} ({count:2d} occurrences)")
    
    output_file = 'wordcloud_keywords.txt'
    with open(output_file, 'w') as f:
        f.write("="*70 + "\n")
        f.write("UNIQUE TERMS (TF-IDF Scored)\n")
        f.write("="*70 + "\n\n")
        f.write("Single words:\n")
        f.write(", ".join([word for word, _ in sorted_tfidf[:20]]))
        f.write("\n\nBigrams:\n")
        f.write(", ".join([phrase for phrase, _ in sorted_bigrams[:10]]))
        f.write("\n\nTrigrams:\n")
        f.write(", ".join([phrase for phrase, _ in sorted_trigrams[:10]]))
        f.write("\n\n" + "="*70 + "\n")
        f.write("DETAILED SCORES\n")
        f.write("="*70 + "\n\n")
        f.write("TF-IDF Words:\n")
        for word, score in sorted_tfidf:
            f.write(f"  {word}: {score:.4f}\n")
        f.write("\nBigrams:\n")
        for phrase, count in sorted_bigrams:
            f.write(f"  {phrase}: {count}\n")
        f.write("\nTrigrams:\n")
        for phrase, count in sorted_trigrams:
            f.write(f"  {phrase}: {count}\n")
    
    print(f"\n{'='*70}")
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()
