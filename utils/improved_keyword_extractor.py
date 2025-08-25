import pandas as pd
import re
import json
from pathlib import Path
from collections import Counter, defaultdict
import difflib

# Configuration
CSV_DIR = Path("data/csv")
OUTPUT_PATH = Path("data/processed/keywords_map_grouped.json")
TERJEMAH_FIELD = "terjemah"
MIN_FREQUENCY = 15
MAX_NGRAM = 3

# Enhanced Indonesian stopwords including hadits-specific terms
STOPWORDS = {
    'yang', 'dan', 'di', 'ke', 'dari', 'pada', 'dengan', 'untuk', 'adalah', 'akan',
    'telah', 'sudah', 'atau', 'juga', 'tidak', 'bila', 'jika', 'ketika', 'saat',
    'itu', 'ini', 'mereka', 'kita', 'kami', 'dia', 'ia', 'saya', 'anda', 'engkau',
    'beliau', 'kepada', 'oleh', 'bagi', 'antara', 'dalam', 'atas', 'bawah', 'luar',
    'maka', 'meski', 'namun', 'tetapi', 'hingga', 'sampai', 'sejak', 'sebab',
    'karena', 'akibat', 'sehingga', 'agar', 'supaya', 'demi', 'guna', 'biar',
    'walau', 'meskipun', 'walaupun', 'seandainya', 'andai', 'sekiranya', 'semoga',
    'mudah', 'barangkali', 'mungkin', 'agaknya', 'rasanya', 'sepertinya', 'tentang',
    'mengenai', 'perihal', 'soal', 'hal', 'masalah', 'perkara', 'ada', 'adanya',
    'terdapat', 'berada', 'terletak', 'bertempat', 'berdomisili', 'dapat', 'bisa',
    'mampu', 'sanggup', 'kuasa', 'boleh', 'diperbolehkan', 'berkata', 'berucap',
    'mengatakan', 'mengucapkan', 'berbicara', 'bersabda', 'menceritakan', 'mengabarkan',
    'pernah', 'mendengar', 'bahwa', 'semua', 'bagi', 'tiap', 'setiap', 'barangsiapa',
    'siapa', 'siapapun', 'mana', 'manapun', 'dimana', 'kemana', 'darimana', 'bagaimana',
    'mengapa', 'kenapa', 'kapan', 'bilamana', 'apakah', 'berapa', 'seberapa',
    'bin', 'abu', 'ibnu', 'al', 'an', 'as', 'ad', 'ar', 'az', 'ats', 'ath',
    'saw', 'ra', 'rah', 'radhiyallahu', 'anhu', 'anha', 'anhum', 'anhuma',
    'shallallahu', 'alaihi', 'wasallam', 'sallallahu', 'alaih'
}

SEMANTIC_GROUPS = {
    'shalat': {
        'keywords': ['shalat', 'salat', 'sholat', 'sembahyang'],
        'description': 'Prayer and worship'
    },
    'puasa': {
        'keywords': ['puasa', 'shaum', 'shiyam', 'berpuasa'],
        'description': 'Fasting'
    },
    'zakat': {
        'keywords': ['zakat', 'zakah', 'berzakat'],
        'description': 'Obligatory charity'
    },
    'haji': {
        'keywords': ['haji', 'hajj', 'ibadah haji', 'berhaji'],
        'description': 'Pilgrimage to Mecca'
    },
    'umrah': {
        'keywords': ['umrah', 'ibadah umrah', 'berumrah'],
        'description': 'Lesser pilgrimage'
    },
    'halal': {
        'keywords': ['halal', 'dihalalkan', 'menghalalkan', 'halalkan'],
        'description': 'Permissible in Islam'
    },
    'haram': {
        'keywords': ['haram', 'diharamkan', 'mengharamkan', 'dilarang'],
        'description': 'Forbidden in Islam'
    },
    'riba': {
        'keywords': ['riba', 'bertransaksi riba', 'mengambil riba'],
        'description': 'Usury/interest'
    },
    'zina': {
        'keywords': ['zina', 'berzina', 'perzinaan'],
        'description': 'Adultery/fornication'
    },
    'khamr': {
        'keywords': ['khamr', 'minuman keras', 'arak', 'mabuk'],
        'description': 'Intoxicants'
    },
    'nikah': {
        'keywords': ['nikah', 'menikah', 'pernikahan', 'perkawinan', 'menikahi'],
        'description': 'Marriage'
    },
    'jihad': {
        'keywords': ['jihad', 'berjihad', 'perjuangan'],
        'description': 'Struggle in the path of Allah'
    },
    'hijrah': {
        'keywords': ['hijrah', 'berhijrah', 'hijra'],
        'description': 'Migration for faith'
    },
    'syahid': {
        'keywords': ['syahid', 'mati syahid', 'kesyahidan'],
        'description': 'Martyrdom'
    },
    'niat': {
        'keywords': ['niat', 'berniat', 'maksud', 'diniatkan'],
        'description': 'Intention'
    },
    'ikhlas': {
        'keywords': ['ikhlas', 'keikhlasan', 'mengikhlaskan'],
        'description': 'Sincerity'
    },
    'sabar': {
        'keywords': ['sabar', 'bersabar', 'kesabaran'],
        'description': 'Patience'
    },
    'taubat': {
        'keywords': ['taubat', 'bertaubat', 'taubah'],
        'description': 'Repentance'
    },
    'doa': {
        'keywords': ['doa', 'berdoa', 'permohonan', 'memohon'],
        'description': 'Supplication'
    },
    'dzikir': {
        'keywords': ['dzikir', 'berdzikir', 'mengingat allah', 'zikir'],
        'description': 'Remembrance of Allah'
    },
    'wudhu': {
        'keywords': ['wudhu', 'berwudhu', 'bersuci'],
        'description': 'Ablution'
    },
    'najis': {
        'keywords': ['najis', 'kenajisan'],
        'description': 'Ritual impurity'
    },
    'pahala': {
        'keywords': ['pahala', 'ganjaran', 'balasan baik'],
        'description': 'Divine reward'
    },
    'dosa': {
        'keywords': ['dosa', 'maksiat', 'perbuatan dosa'],
        'description': 'Sin'
    },
    'surga': {
        'keywords': ['surga', 'jannah', 'syurga'],
        'description': 'Paradise'
    },
    'neraka': {
        'keywords': ['neraka', 'jahannam'],
        'description': 'Hell'
    },
    'iman': {
        'keywords': ['iman', 'beriman', 'keimanan'],
        'description': 'Faith'
    },
    'islam': {
        'keywords': ['islam', 'keislaman'],
        'description': 'Submission to Allah'
    },
    'sedekah': {
        'keywords': ['sedekah', 'bersedekah', 'shadaqah'],
        'description': 'Voluntary charity'
    },
    'ilmu': {
        'keywords': ['ilmu', 'menuntut ilmu', 'pengetahuan'],
        'description': 'Knowledge'
    },
    'akhlak': {
        'keywords': ['akhlak', 'budi pekerti', 'adab'],
        'description': 'Character/morality'
    }
}

def normalize_text(text: str) -> str:
    """Enhanced text normalization for hadits content"""
    if pd.isna(text):
        return ""
    
    text = text.lower().strip()
    
    # Remove narrator chain markers and standardize
    text = re.sub(r'["""\'`]', '', text)
    text = re.sub(r'\s*-\s*', ' ', text)  # Clean up dashes
    
    # Normalize Islamic phrases
    text = re.sub(r'shallallahu\s+\'?alaihi\s+wa?\s*sallam', 'saw', text)
    text = re.sub(r'sallallahu\s+\'?alaihi\s+wa?\s*sallam', 'saw', text)
    text = re.sub(r'radhi\s*allahu\s+(anhu|anha|anhum)', 'ra', text)
    text = re.sub(r'radhiyallahu\s+(anhu|anha|anhum)', 'ra', text)
    
    # Remove excessive punctuation but preserve sentence structure
    text = re.sub(r'[^\w\s\'-]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def is_meaningful_term(term: str) -> bool:
    """Check if a term is meaningful for Islamic keyword extraction"""
    words = term.split()
    
    # Skip if all words are stopwords
    if all(word in STOPWORDS for word in words):
        return False
    
    # Skip pure numbers or very short terms
    if len(term.strip()) <= 2 or re.match(r'^\d+$', term.strip()):
        return False
    
    # Skip terms that are just narrator names or chains
    narrator_patterns = [
        r'^(bin|abu|ibnu|al|an|as|ad|ar|az|ath|ats)\s',
        r'\s(bin|abu|ibnu)\s',
        r'^(muhammad|ahmad|abdullah|umar|ali|hassan|hussain)\s',
        r'(berkata|menceritakan|mengabarkan|mendengar)\s+(bin|abu|ibnu)'
    ]
    
    for pattern in narrator_patterns:
        if re.search(pattern, term):
            return False
    
    return True

def generate_meaningful_ngrams(text: str, max_n: int = MAX_NGRAM) -> list:
    """Generate meaningful n-grams excluding narrator chains and stopwords"""
    words = text.split()
    ngrams = []
    
    for n in range(1, max_n + 1):
        for i in range(len(words) - n + 1):
            ngram = ' '.join(words[i:i+n])
            if is_meaningful_term(ngram):
                ngrams.append(ngram)
    
    return ngrams

def load_csv_texts(csv_dir: Path) -> list:
    """Load all terjemah texts from CSV files"""
    all_texts = []
    
    for csv_file in csv_dir.glob("*.csv"):
        try:
            print(f"📖 Reading {csv_file.name}...")
            df = pd.read_csv(csv_file)
            
            if TERJEMAH_FIELD in df.columns:
                texts = df[TERJEMAH_FIELD].dropna().tolist()
                all_texts.extend(texts)
                print(f"   ✓ {len(texts)} hadis loaded")
            else:
                print(f"   ⚠️  Column '{TERJEMAH_FIELD}' not found")
                
        except Exception as e:
            print(f"   ❌ Error reading {csv_file.name}: {e}")
    
    print(f"\n📊 Total hadis texts loaded: {len(all_texts)}")
    return all_texts

def extract_frequent_terms(texts: list, min_freq: int = MIN_FREQUENCY) -> Counter:
    """Extract meaningful terms that appear more than min_freq times"""
    print("🔍 Generating meaningful n-grams and counting frequencies...")
    
    all_ngrams = []
    for i, text in enumerate(texts):
        if i % 1000 == 0:
            print(f"   Processing text {i+1}/{len(texts)}")
        
        normalized = normalize_text(text)
        ngrams = generate_meaningful_ngrams(normalized)
        all_ngrams.extend(ngrams)
    
    print(f"📈 Generated {len(all_ngrams)} meaningful n-grams")
    
    # Count frequencies
    counter = Counter(all_ngrams)
    
    # Filter by minimum frequency
    frequent_terms = {term: count for term, count in counter.items() 
                     if count >= min_freq}
    
    print(f"🎯 Found {len(frequent_terms)} terms with frequency >= {min_freq}")
    return Counter(frequent_terms)

def group_by_semantic_categories(frequent_terms: Counter) -> dict:
    
    terms = list(frequent_terms.keys())
    grouped = defaultdict(list)
    categorized = set()
    
    for category, group_info in SEMANTIC_GROUPS.items():
        keywords = group_info['keywords']
        group_members = []
        
        for term in terms:
            for keyword in keywords:
                if keyword in term.lower():
                    group_members.append(term)
                    categorized.add(term)
                    break
        
        if group_members:
            grouped[category] = sorted(set(group_members))
    
    remaining_terms = [term for term in terms if term not in categorized]
    remaining_terms.sort(key=lambda x: frequent_terms[x], reverse=True)
    
    for term in remaining_terms[:100]:  
        if frequent_terms[term] >= MIN_FREQUENCY * 2:  
            grouped[term] = [term]
    
    print(f"📋 Created {len(grouped)} semantic groups")
    return dict(grouped)

def save_results(grouped_terms: dict, output_path: Path):
    """Save results to JSON file with metadata"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Add metadata
    result = {
        "metadata": {
            "description": "Islamic keywords extracted from hadits collections",
            "min_frequency": MIN_FREQUENCY,
            "max_ngram": MAX_NGRAM,
            "total_groups": len(grouped_terms),
            "extraction_method": "improved_semantic_grouping"
        },
        "keywords": grouped_terms
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"💾 Results saved to {output_path}")

def main():
    """Main extraction pipeline with improved Islamic terminology handling"""
    print("🚀 Starting Improved Islamic Keyword Extraction")
    print("=" * 50)
    
    # Load all CSV texts
    texts = load_csv_texts(CSV_DIR)
    
    if not texts:
        print("❌ No texts found. Please check the CSV files.")
        return
    
    # Extract frequent meaningful terms
    frequent_terms = extract_frequent_terms(texts, MIN_FREQUENCY)
    
    if not frequent_terms:
        print(f"❌ No terms found with frequency >= {MIN_FREQUENCY}")
        return
    
    # Group by Islamic semantic categories
    grouped_terms = group_by_semantic_categories(frequent_terms)
    
    # Save results
    save_results(grouped_terms, OUTPUT_PATH)
    
    # Print summary
    print("\n" + "=" * 50)
    print("📊 IMPROVED EXTRACTION SUMMARY")
    print("=" * 50)
    print(f"Total hadis processed: {len(texts):,}")
    print(f"Meaningful terms with frequency >= {MIN_FREQUENCY}: {len(frequent_terms):,}")
    print(f"Final semantic groups: {len(grouped_terms):,}")
    print(f"Output file: {OUTPUT_PATH}")
    
    # Show Islamic categories
    print("\n🕌 Islamic semantic categories found:")
    islamic_categories = [k for k in grouped_terms.keys() if k in SEMANTIC_GROUPS]
    for category in sorted(islamic_categories):
        count = len(grouped_terms[category])
        desc = SEMANTIC_GROUPS[category]['description']
        print(f"  • {category}: {count} terms ({desc})")

if __name__ == "__main__":
    main() 