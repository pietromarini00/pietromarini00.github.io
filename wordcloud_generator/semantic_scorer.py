"""
Generate a word cloud using hybrid TF-IDF + Semantic Similarity scoring.
Combines TF-IDF uniqueness with semantic relevance to target concepts.
Uses percentile-based normalization for fair weighting.
"""

import os
from collections import Counter, defaultdict
import re
from pathlib import Path
import numpy as np
from . import config

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

def get_word_embeddings():
    """
    Load or compute word embeddings for semantic similarity.
    Using sentence-transformers for semantic embeddings.
    """
    try:
        from sentence_transformers import SentenceTransformer
        print("Loading semantic model (sentence-transformers)...")
        model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight, fast model
        return model
    except ImportError:
        print("\nWARNING: sentence-transformers not installed!")
        print("Install with: pip install sentence-transformers")
        print("Falling back to TF-IDF only mode.\n")
        return None

def compute_semantic_similarity(model, terms, target_concepts):
    """
    Compute semantic similarity between terms and target concepts.
    Returns max cosine similarity to any target concept.
    """
    if model is None:
        return {term: 0.0 for term in terms}
    
    print(f"Computing semantic embeddings for {len(terms)} terms...")
    
    # Encode all terms and target concepts
    term_embeddings = model.encode(terms, show_progress_bar=False)
    target_embeddings = model.encode(target_concepts, show_progress_bar=False)
    
    # Compute cosine similarity (already normalized by sentence-transformers)
    from sklearn.metrics.pairwise import cosine_similarity
    
    similarities = {}
    for i, term in enumerate(terms):
        # Get max similarity to any target concept
        sims = cosine_similarity([term_embeddings[i]], target_embeddings)[0]
        similarities[term] = float(np.max(sims))
    
    return similarities

def percentile_normalize(scores_dict):
    """
    Normalize scores to 0-1 range using percentile ranks.
    This puts different metrics on the same scale.
    """
    if not scores_dict:
        return {}
    
    values = np.array(list(scores_dict.values()))
    normalized = {}
    
    for term, score in scores_dict.items():
        # Compute percentile rank (0-1)
        percentile = np.sum(values <= score) / len(values)
        normalized[term] = percentile
    
    return normalized

def combine_scores(tfidf_scores, semantic_scores, alpha=0.5):
    """
    Combine TF-IDF and semantic scores with weighting parameter alpha.
    
    final_score = alpha * tfidf_percentile + (1-alpha) * semantic_percentile
    
    Args:
        tfidf_scores: dict of term -> TF-IDF score
        semantic_scores: dict of term -> semantic similarity score
        alpha: weight for TF-IDF (0=only semantic, 1=only TF-IDF, 0.5=balanced)
    
    Returns:
        dict of term -> combined score
    """
    # Normalize both to 0-1 using percentiles
    tfidf_norm = percentile_normalize(tfidf_scores)
    semantic_norm = percentile_normalize(semantic_scores)
    
    # Combine with weighting
    combined = {}
    all_terms = set(tfidf_scores.keys()) | set(semantic_scores.keys())
    
    for term in all_terms:
        tfidf_val = tfidf_norm.get(term, 0.0)
        semantic_val = semantic_norm.get(term, 0.0)
        combined[term] = alpha * tfidf_val + (1 - alpha) * semantic_val
    
    return combined

def deduplicate_similar_terms(model, scored_terms, similarity_threshold=0.85):
    """
    Remove terms that are too similar to higher-scored terms (synonyms/duplicates).
    
    Args:
        model: SentenceTransformer model
        scored_terms: list of (term, score) tuples, sorted by score descending
        similarity_threshold: cosine similarity above which terms are considered duplicates
    
    Returns:
        list of (term, score) tuples with similar terms removed
    """
    if model is None or len(scored_terms) <= 1:
        return scored_terms
    
    print(f"Deduplicating terms (similarity threshold: {similarity_threshold})...")
    
    # Extract terms and compute embeddings
    terms = [t[0] for t in scored_terms]
    embeddings = model.encode(terms, show_progress_bar=False)
    
    from sklearn.metrics.pairwise import cosine_similarity
    
    # Keep track of selected terms
    selected = []
    selected_embeddings = []
    
    for i, (term, score) in enumerate(scored_terms):
        # Check similarity to all already-selected terms
        if selected_embeddings:
            sims = cosine_similarity([embeddings[i]], selected_embeddings)[0]
            max_sim = np.max(sims)
            
            if max_sim >= similarity_threshold:
                # Too similar to an existing term, skip it
                continue
        
        # Add to selected
        selected.append((term, score))
        selected_embeddings.append(embeddings[i])
    
    removed_count = len(scored_terms) - len(selected)
    if removed_count > 0:
        print(f"  Removed {removed_count} similar terms")
    
    return selected

def read_allowed_multigrams():
    """Read allowed multi-word phrases from single file."""
    try:
        with open(config.ALLOWED_MULTIGRAMS_FILE, 'r') as f:
            return [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        return []

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
    """Extract individual words from text."""
    words = re.findall(r'\b[a-z]{3,}\b', text.lower())
    return [w for w in words if w not in STOP_WORDS]

def extract_ngrams(text, n=2):
    """Extract n-grams from text."""
    words = re.findall(r'\b[a-z]+\b', text.lower())
    # Filter stop words
    words = [w for w in words if w not in STOP_WORDS and len(w) >= 3]
    
    ngrams = []
    for i in range(len(words) - n + 1):
        ngram = ' '.join(words[i:i+n])
        ngrams.append(ngram)
    
    return ngrams

# General English IDF values (estimated from large corpus)
GENERAL_IDF = {
    'data': 3.5, 'system': 4.0, 'use': 3.8, 'information': 4.2, 'based': 3.9,
    'using': 3.7, 'research': 4.5, 'science': 5.0, 'analysis': 4.8, 'model': 4.3,
    'method': 4.6, 'learning': 5.2, 'process': 4.1, 'development': 4.4, 'management': 4.7,
    'technology': 4.5, 'network': 4.8, 'design': 4.3, 'application': 4.2, 'computer': 5.1,
    'software': 5.0, 'engineering': 5.3, 'algorithm': 5.8, 'database': 5.5, 'knowledge': 4.9,
    'semantic': 6.2, 'infrastructure': 5.7, 'metadata': 6.5, 'ontology': 6.8, 'provenance': 7.0,
    'dataset': 5.9, 'scientific': 5.4, 'computational': 5.8, 'workflow': 5.6, 'integration': 5.2,
    'fair': 4.0, 'reproducible': 6.3, 'bayesian': 6.7, 'regression': 5.5, 'parsing': 5.9,
}

def calculate_tfidf(term_freq, total_docs=1):
    """Calculate TF-IDF scores."""
    tfidf_scores = {}
    
    for term, freq in term_freq.items():
        tf = freq
        idf = GENERAL_IDF.get(term, 8.0)  # High IDF for unknown terms (likely unique)
        tfidf_scores[term] = tf * (idf / 10.0)  # Normalize IDF to reasonable range
    
    return tfidf_scores

def compute_document_similarity(model, phrases, document_text):
    """
    Compute semantic similarity between phrases and the full document.
    This captures how relevant each phrase is to the overall document context.
    """
    if model is None:
        return {phrase: 0.0 for phrase in phrases}
    
    print(f"Computing document similarity for {len(phrases)} phrases...")
    
    # Encode the full document
    doc_embedding = model.encode([document_text], show_progress_bar=False)[0]
    
    # Encode all phrases
    phrase_embeddings = model.encode(phrases, show_progress_bar=False)
    
    # Compute cosine similarity
    from sklearn.metrics.pairwise import cosine_similarity
    
    similarities = {}
    for i, phrase in enumerate(phrases):
        sim = cosine_similarity([phrase_embeddings[i]], [doc_embedding])[0][0]
        similarities[phrase] = float(sim)
    
    return similarities

def main():
    print("="*70)
    print("HYBRID TF-IDF + SEMANTIC SIMILARITY WORD CLOUD GENERATOR")
    print("="*70)
    
    # Use config values
    ALPHA = config.ALPHA
    USE_ALLOWED_FILTER = config.USE_ALLOWED_FILTER
    USE_DOCUMENT_SIMILARITY = config.USE_DOCUMENT_SIMILARITY
    DEDUPLICATE_SYNONYMS = config.DEDUPLICATE_SYNONYMS
    SYNONYM_THRESHOLD = config.SYNONYM_THRESHOLD
    MIN_FREQ = config.MIN_FREQ
    TOP_WORDS = config.TOP_WORDS
    TOP_BIGRAMS = config.TOP_BIGRAMS
    TOP_TRIGRAMS = config.TOP_TRIGRAMS
    
    print(f"\nConfiguration:")
    print(f"  Alpha (TF-IDF weight): {ALPHA}")
    print(f"  1-Alpha (Semantic weight): {1-ALPHA}")
    print(f"  Use allowed_multigrams filter: {USE_ALLOWED_FILTER}")
    print(f"  Use document similarity: {USE_DOCUMENT_SIMILARITY}")
    print(f"  Deduplicate synonyms: {DEDUPLICATE_SYNONYMS} (threshold: {SYNONYM_THRESHOLD})")
    if not USE_DOCUMENT_SIMILARITY:
        print(f"  Target concepts: {len(config.TARGET_CONCEPTS)}")
    print()
    
    # Read input text
    text = read_text_files(config.INPUT_DIR)
    
    if not text:
        print("No text found!")
        return
    
    # Extract words and compute TF-IDF
    print("\n" + "="*70)
    print("COMPUTING TF-IDF SCORES")
    print("="*70)
    
    words = extract_words(text)
    word_freq = Counter(words)
    word_freq = {w: c for w, c in word_freq.items() if c >= MIN_FREQ}
    
    tfidf_scores = calculate_tfidf(word_freq)
    
    # Load semantic model and compute similarities
    print("\n" + "="*70)
    print("COMPUTING SEMANTIC SIMILARITIES")
    print("="*70)
    
    model = get_word_embeddings()
    
    if USE_DOCUMENT_SIMILARITY:
        print("Using document similarity mode (comparing to full text)")
        semantic_scores = compute_document_similarity(model, list(word_freq.keys()), text)
    else:
        print("Using target concepts mode")
        semantic_scores = compute_semantic_similarity(model, list(word_freq.keys()), config.TARGET_CONCEPTS)
    
    # Combine scores
    print("\n" + "="*70)
    print("COMBINING SCORES")
    print("="*70)
    
    combined_scores = combine_scores(tfidf_scores, semantic_scores, alpha=ALPHA)
    
    # Sort by combined score
    top_words_raw = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Deduplicate synonyms if enabled
    if DEDUPLICATE_SYNONYMS and model:
        top_words_raw = deduplicate_similar_terms(model, top_words_raw, SYNONYM_THRESHOLD)
    
    top_words = top_words_raw[:TOP_WORDS]
    
    # Extract n-grams or load from allowed list
    print("\n" + "="*70)
    print("LOADING N-GRAMS")
    print("="*70)
    
    # Load allowed multigrams if filtering is enabled
    if USE_ALLOWED_FILTER:
        allowed_phrases = read_allowed_multigrams()
        print(f"Loaded {len(allowed_phrases)} allowed multigrams")
        print("Using ONLY phrases from allowed_multigrams.txt (no extraction)")
        
        # Separate by n-gram size
        bigram_list = [p for p in allowed_phrases if len(p.split()) == 2]
        trigram_list = [p for p in allowed_phrases if len(p.split()) == 3]
        
        print(f"  - {len(bigram_list)} bigrams")
        print(f"  - {len(trigram_list)} trigrams")
        
        # Assign frequency 1 to all (or could look up in text if needed)
        bigram_freq = {bg: 1 for bg in bigram_list}
        trigram_freq = {tg: 1 for tg in trigram_list}
    else:
        # Extract all n-grams from text
        print("Extracting n-grams from text...")
        bigrams = extract_ngrams(text, n=2)
        bigram_freq = Counter(bigrams)
        bigram_freq = {bg: c for bg, c in bigram_freq.items() if c >= MIN_FREQ}
        
        trigrams = extract_ngrams(text, n=3)
        trigram_freq = Counter(trigrams)
        trigram_freq = {tg: c for tg, c in trigram_freq.items() if c >= MIN_FREQ}
        
        print(f"  - {len(bigram_freq)} bigrams (freq >= {MIN_FREQ})")
        print(f"  - {len(trigram_freq)} trigrams (freq >= {MIN_FREQ})")
    
    # Compute semantic scores for n-grams
    if model:
        if USE_DOCUMENT_SIMILARITY:
            bigram_semantic = compute_document_similarity(model, list(bigram_freq.keys()), text)
            trigram_semantic = compute_document_similarity(model, list(trigram_freq.keys()), text)
        else:
            bigram_semantic = compute_semantic_similarity(model, list(bigram_freq.keys()), config.TARGET_CONCEPTS)
            trigram_semantic = compute_semantic_similarity(model, list(trigram_freq.keys()), config.TARGET_CONCEPTS)
    else:
        bigram_semantic = {bg: 0.0 for bg in bigram_freq}
        trigram_semantic = {tg: 0.0 for tg in trigram_freq}
    
    # Normalize frequencies by percentile before weighting
    bigram_freq_norm = percentile_normalize(bigram_freq)
    trigram_freq_norm = percentile_normalize(trigram_freq)
    
    # Use frequency as TF-IDF proxy for n-grams (now percentile-normalized)
    bigram_combined = combine_scores(
        {bg: bigram_freq_norm[bg] for bg in bigram_freq},
        bigram_semantic,
        alpha=ALPHA
    )
    
    trigram_combined = combine_scores(
        {tg: trigram_freq_norm[tg] for tg in trigram_freq},
        trigram_semantic,
        alpha=ALPHA
    )
    
    # Sort and deduplicate
    top_bigrams_raw = sorted(bigram_combined.items(), key=lambda x: x[1], reverse=True)
    top_trigrams_raw = sorted(trigram_combined.items(), key=lambda x: x[1], reverse=True)
    
    if DEDUPLICATE_SYNONYMS and model:
        top_bigrams_raw = deduplicate_similar_terms(model, top_bigrams_raw, SYNONYM_THRESHOLD)
        top_trigrams_raw = deduplicate_similar_terms(model, top_trigrams_raw, SYNONYM_THRESHOLD)
    
    top_bigrams = top_bigrams_raw[:TOP_BIGRAMS]
    top_trigrams = top_trigrams_raw[:TOP_TRIGRAMS]
    
    # Write results
    with open(config.OUTPUT_KEYWORDS_FILE, 'w') as f:
        f.write(f"Hybrid TF-IDF + Semantic Similarity Keywords\n")
        f.write(f"Alpha (TF-IDF weight): {ALPHA}\n")
        f.write(f"{'='*70}\n\n")
        
        f.write("Top Words (Combined Score):\n\n")
        for word, score in top_words:
            tfidf = tfidf_scores.get(word, 0)
            semantic = semantic_scores.get(word, 0)
            f.write(f"{word}: {score:.4f} (TF-IDF: {tfidf:.4f}, Semantic: {semantic:.4f})\n")
        
        f.write("\n\nBigrams (Combined Score):\n\n")
        for bg, score in top_bigrams:
            freq = bigram_freq.get(bg, 0)
            freq_norm = bigram_freq_norm.get(bg, 0)
            semantic = bigram_semantic.get(bg, 0)
            f.write(f"{bg}: {score:.4f} (Freq: {freq}, FreqPercentile: {freq_norm:.4f}, Semantic: {semantic:.4f})\n")
        
        f.write("\n\nTrigrams (Combined Score):\n\n")
        for tg, score in top_trigrams:
            freq = trigram_freq.get(tg, 0)
            freq_norm = trigram_freq_norm.get(tg, 0)
            semantic = trigram_semantic.get(tg, 0)
            f.write(f"{tg}: {score:.4f} (Freq: {freq}, FreqPercentile: {freq_norm:.4f}, Semantic: {semantic:.4f})\n")
    
    # Also write detailed breakdown
    with open(config.OUTPUT_BREAKDOWN_FILE, 'w') as f:
        f.write(f"DETAILED SCORING BREAKDOWN\n")
        f.write(f"Alpha: {ALPHA} (TF-IDF weight), {1-ALPHA} (Semantic weight)\n")
        f.write(f"{'='*80}\n\n")
        
        f.write("SINGLE WORDS:\n")
        f.write(f"{'Term':<25} {'Combined':<10} {'TF-IDF':<12} {'TF-IDF%':<12} {'Semantic':<12} {'Sem%':<10}\n")
        f.write(f"{'-'*80}\n")
        
        # Compute percentiles for display
        tfidf_norm = percentile_normalize(tfidf_scores)
        semantic_norm = percentile_normalize(semantic_scores)
        
        for word, score in top_words:
            tfidf_raw = tfidf_scores.get(word, 0)
            tfidf_pct = tfidf_norm.get(word, 0)
            sem_raw = semantic_scores.get(word, 0)
            sem_pct = semantic_norm.get(word, 0)
            f.write(f"{word:<25} {score:<10.4f} {tfidf_raw:<12.4f} {tfidf_pct:<12.4f} {sem_raw:<12.4f} {sem_pct:<10.4f}\n")
        
        f.write(f"\n\nBIGRAMS:\n")
        f.write(f"{'Term':<30} {'Combined':<10} {'Freq':<8} {'Freq%':<10} {'Semantic':<12} {'Sem%':<10}\n")
        f.write(f"{'-'*80}\n")
        
        bigram_sem_norm = percentile_normalize(bigram_semantic)
        
        for bg, score in top_bigrams:
            freq = bigram_freq.get(bg, 0)
            freq_pct = bigram_freq_norm.get(bg, 0)
            sem_raw = bigram_semantic.get(bg, 0)
            sem_pct = bigram_sem_norm.get(bg, 0)
            f.write(f"{bg:<30} {score:<10.4f} {freq:<8} {freq_pct:<10.4f} {sem_raw:<12.4f} {sem_pct:<10.4f}\n")
        
        f.write(f"\n\nTRIGRAMS:\n")
        f.write(f"{'Term':<35} {'Combined':<10} {'Freq':<8} {'Freq%':<10} {'Semantic':<12} {'Sem%':<10}\n")
        f.write(f"{'-'*80}\n")
        
        trigram_sem_norm = percentile_normalize(trigram_semantic)
        
        for tg, score in top_trigrams:
            freq = trigram_freq.get(tg, 0)
            freq_pct = trigram_freq_norm.get(tg, 0)
            sem_raw = trigram_semantic.get(tg, 0)
            sem_pct = trigram_sem_norm.get(tg, 0)
            f.write(f"{tg:<35} {score:<10.4f} {freq:<8} {freq_pct:<10.4f} {sem_raw:<12.4f} {sem_pct:<10.4f}\n")
        
        f.write(f"\n\n{'='*80}\n")
        f.write(f"Formula: Combined = Alpha × Percentile(Freq or TF-IDF) + (1-Alpha) × Percentile(Semantic)\n")
        f.write(f"Freq% = Percentile rank of frequency (0-1 scale)\n")
        f.write(f"Sem% = Percentile rank of semantic similarity (0-1 scale)\n")
    
    print(f"\nResults saved to: {config.OUTPUT_KEYWORDS_FILE}")
    print(f"\nTop 10 words by combined score:")
    for word, score in top_words[:10]:
        print(f"  {word}: {score:.4f}")

if __name__ == "__main__":
    main()
