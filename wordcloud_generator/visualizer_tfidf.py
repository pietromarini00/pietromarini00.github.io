"""
Generate a visual word cloud from TF-IDF scored terms.
Creates a PNG/JPG image to replace the About section.
"""

import matplotlib.pyplot as plt
from wordcloud import WordCloud
import matplotlib.colors as mcolors
from pathlib import Path

def read_allowed_multigrams():
    """Read allowed multi-word phrases from single file."""
    try:
        with open('wordcloud_input/allowed_multigrams.txt', 'r') as f:
            return set(line.strip().lower() for line in f if line.strip())
    except FileNotFoundError:
        return set()

def load_tfidf_data():
    """Load TF-IDF data from the keywords file."""
    word_scores = {}
    
    try:
        with open('wordcloud_keywords.txt', 'r') as f:
            content = f.read()
            
        # Extract TF-IDF section
        if 'TF-IDF Words:' in content:
            tfidf_section = content.split('TF-IDF Words:')[1].split('\n\nBigrams:')[0]
            for line in tfidf_section.strip().split('\n'):
                if ':' in line:
                    word, score = line.strip().split(':', 1)
                    word = word.strip()
                    score = float(score.strip())
                    word_scores[word] = score
        
        # Load allowed multigrams once
        allowed_multigrams = read_allowed_multigrams()
        
        # Extract bigrams
        if 'Bigrams:' in content:
            bigrams_section = content.split('Bigrams:')[1].split('\n\nTrigrams:')[0]
            
            for line in bigrams_section.strip().split('\n'):
                if ':' in line:
                    phrase, count = line.strip().split(':', 1)
                    phrase = phrase.strip()
                    if phrase.lower() in allowed_multigrams:
                        # Weight bigrams higher
                        word_scores[phrase] = int(count.strip()) * 0.05
        
        # Extract trigrams
        if 'Trigrams:' in content:
            trigrams_section = content.split('Trigrams:')[1]
            
            for line in trigrams_section.strip().split('\n'):
                if ':' in line:
                    phrase, count = line.strip().split(':', 1)
                    phrase = phrase.strip()
                    if phrase.lower() in allowed_multigrams:
                        # Weight trigrams much higher (rarer, so need more prominence)
                        word_scores[phrase] = int(count.strip()) * 0.15
        
    except Exception as e:
        print(f"Error loading data: {e}")
    
    return word_scores

def generate_wordcloud_image(output_file='wordcloud.png'):
    """Generate and save word cloud visualization."""
    
    # Load data
    word_scores = load_tfidf_data()
    
    if not word_scores:
        print("No word data found!")
        return
    
    print(f"Generating word cloud with {len(word_scores)} terms...")
    
    # Create word cloud with minimalist style
    wordcloud = WordCloud(
        width=1400,
        height=600,
        background_color='white',
        color_func=lambda *args, **kwargs: '#333333',  # Dark gray text
        font_path=None,  # Use default system font
        relative_scaling=0.5,
        min_font_size=10,
        max_font_size=80,
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
    plt.savefig(output_file, dpi=150, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print(f"\nWord cloud saved to: {output_file}")
    
    # Also save as JPG
    jpg_file = output_file.replace('.png', '.jpg')
    plt.savefig(jpg_file, dpi=150, bbox_inches='tight', 
                facecolor='white', edgecolor='none', format='jpg')
    print(f"Word cloud also saved to: {jpg_file}")
    
    plt.close()

def suggest_ngrams():
    """Suggest additional bigrams and trigrams based on domain knowledge."""
    
    suggested_bigrams = [
        # Research & Academia
        "research assistant", "research engineer", "research center", "research infrastructure",
        "research workflows", "research reproducibility", "research collaboration", "research automation",
        "graduate student", "doctoral candidate", "academic research", "research paper",
        
        # Data Science & ML
        "machine learning", "deep learning", "neural networks", "natural language",
        "computer vision", "data analytics", "data mining", "data processing",
        "data visualization", "data warehousing", "data pipeline", "data quality",
        "model training", "model deployment", "model evaluation", "feature engineering",
        "transfer learning", "reinforcement learning", "supervised learning", "unsupervised learning",
        
        # Data Management
        "data management", "data integration", "data infrastructure", "data systems",
        "data architecture", "data governance", "data provenance", "data lineage",
        "data catalog", "data discovery", "data sharing", "data reuse",
        "metadata standards", "metadata extraction", "metadata generation", "metadata enrichment",
        
        # Scientific Computing
        "scientific data", "scientific computing", "scientific infrastructure", "scientific workflows",
        "computational biology", "computational methods", "computational analysis", "computational tools",
        "reproducible research", "reproducible science", "open science", "open data",
        
        # Technologies & Tools
        "semantic parsing", "semantic search", "semantic web", "semantic analysis",
        "language models", "large language", "transformer models", "attention mechanisms",
        "dataset metadata", "dataset discovery", "dataset linkage", "dataset ontologies",
        "knowledge graphs", "graph databases", "relational databases", "document databases",
        
        # Cloud & Engineering
        "cloud platforms", "cloud infrastructure", "cloud computing", "distributed systems",
        "software engineering", "data engineering", "system design", "api design",
        "etl pipelines", "code migration", "legacy systems", "system integration",
        
        # NYU & Institutions
        "new york", "york university", "nyu vida", "georgia institute",
        "institute technology", "georgia tech", "bocconi university", "milan italy",
        
        # Fair & Standards
        "fair data", "fair principles", "data principles", "data standards",
        "data quality", "data validation", "data curation", "data annotation",
        
        # Publications & Output
        "research publications", "conference papers", "workshop presentations", "scholarly documents",
        "literature mining", "citation analysis", "document processing", "text extraction",
        
        # Specific Domains
        "biomedical data", "clinical data", "genomic data", "experimental data",
        "time series", "spatial data", "graph data", "multimodal data"
    ]
    
    suggested_trigrams = [
        # Institutions
        "georgia institute technology", "new york university", "nyu vida center",
        "bocconi university milan", "atlanta georgia usa",
        
        # FAIR Principles
        "fair data principles", "findable accessible interoperable", "accessible interoperable reusable",
        "data findability accessibility", "interoperable reusable data",
        
        # Research Areas
        "large language models", "natural language processing", "computer vision systems",
        "machine learning models", "deep learning networks", "semantic parsing models",
        "transformer based models", "attention based models", "neural network architectures",
        
        # Data Management
        "data management systems", "data integration systems", "data infrastructure systems",
        "metadata extraction tools", "dataset discovery tools", "data provenance tracking",
        "data lineage tracking", "metadata generation pipeline", "dataset linkage methods",
        
        # Scientific Computing
        "reproducible research workflows", "scientific data infrastructure", "computational research methods",
        "open science practices", "research data management", "scientific workflow systems",
        "biomedical data integration", "clinical data analysis", "genomic data processing",
        
        # Technologies
        "llm powered systems", "data gatherer llm-powered", "semantic parsing techniques",
        "knowledge graph construction", "graph database systems", "document processing pipelines",
        
        # Methods & Techniques
        "bayesian additive regression", "statistical machine learning", "probabilistic graphical models",
        "transfer learning methods", "reinforcement learning algorithms", "supervised learning techniques",
        "feature extraction methods", "dimensionality reduction techniques", "clustering algorithms methods",
        
        # Cloud & Engineering
        "cloud platform migration", "etl pipeline development", "distributed computing systems",
        "microservices architecture design", "api gateway design", "database schema design",
        "system architecture patterns", "software design patterns", "code modernization projects",
        
        # Publications & Research
        "scholarly document processing", "scientific literature mining", "citation network analysis",
        "publication metadata extraction", "research paper analysis", "conference paper publication",
        
        # Specific Projects
        "data gatherer project", "nyu data management", "professor juliana freire",
        "vida center research", "acl workshop presentation",
        
        # Data Quality
        "data quality assessment", "data validation methods", "data cleaning techniques",
        "data normalization procedures", "data transformation pipelines", "data enrichment processes",
        
        # Collaboration
        "research team collaboration", "interdisciplinary research teams", "cross institutional collaboration",
        "industry academic partnerships", "open source contributions", "community driven development",
        
        # Infrastructure
        "scalable data infrastructure", "distributed storage systems", "high performance computing",
        "containerized application deployment", "kubernetes orchestration systems", "docker container management",
        
        # Standards & Protocols
        "metadata schema standards", "data exchange protocols", "api specification standards",
        "ontology development standards", "semantic web standards", "linked data principles",
        
        # Analysis & Processing
        "exploratory data analysis", "statistical data analysis", "predictive model development",
        "time series forecasting", "anomaly detection methods", "pattern recognition techniques"
    ]
    
    return suggested_bigrams[:100], suggested_trigrams[:100]

def main():
    print("="*70)
    print("WORD CLOUD VISUALIZATION GENERATOR")
    print("="*70 + "\n")
    
    # Generate word cloud image
    generate_wordcloud_image('wordcloud.png')
    
    # Suggest additional n-grams
    print("\n" + "="*70)
    print("SUGGESTED N-GRAMS TO ADD")
    print("="*70 + "\n")
    
    bigrams, trigrams = suggest_ngrams()
    
    print("SUGGESTED BIGRAMS (100):")
    print("-" * 70)
    for i, bg in enumerate(bigrams, 1):
        print(f"{i:3d}. {bg}")
    
    print("\n" + "="*70)
    print("\nSUGGESTED TRIGRAMS (100):")
    print("-" * 70)
    for i, tg in enumerate(trigrams, 1):
        print(f"{i:3d}. {tg}")
    
    # Save suggestions to file
    with open('suggested_ngrams.txt', 'w') as f:
        f.write("SUGGESTED BIGRAMS:\n\n")
        f.write('\n'.join(bigrams))
        f.write("\n\n" + "="*70 + "\n\n")
        f.write("SUGGESTED TRIGRAMS:\n\n")
        f.write('\n'.join(trigrams))
    
    print(f"\n{'='*70}")
    print("Suggestions saved to: suggested_ngrams.txt")

if __name__ == "__main__":
    main()
