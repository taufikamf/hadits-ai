import pandas as pd
import re
import json
from pathlib import Path
from collections import Counter, defaultdict
from itertools import combinations
import difflib

# Configuration
CSV_DIR = Path("data/csv")
OUTPUT_PATH = Path("data/processed/keywords_map_grouped.json")
TERJEMAH_FIELD = "terjemah"
MIN_FREQUENCY = 20
MAX_NGRAM = 3

# Indonesian stopwords
STOPWORDS = {
    'yang', 'dan', 'di', 'ke', 'dari', 'pada', 'dengan', 'untuk', 'adalah', 'akan',
    'telah', 'sudah', 'atau', 'juga', 'tidak', 'bila', 'jika', 'ketika', 'saat',
    'itu', 'ini', 'mereka', 'kita', 'kami', 'dia', 'ia', 'saya', 'anda', 'engkau',
    'beliau', 'kepada', 'oleh', 'bagi', 'antara', 'dalam', 'atas', 'bawah', 'luar',
    'maka', 'meski', 'namun', 'tetapi', 'namun', 'hingga', 'sampai', 'sejak',
    'sebab', 'karena', 'akibat', 'sehingga', 'agar', 'supaya', 'demi', 'guna',
    'biar', 'walau', 'meskipun', 'walaupun', 'seandainya', 'andai', 'sekiranya',
    'semoga', 'mudah', 'barangkali', 'mungkin', 'agaknya', 'rasanya', 'sepertinya',
    'tentang', 'mengenai', 'perihal', 'soal', 'hal', 'masalah', 'perkara',
    'ada', 'adanya', 'terdapat', 'berada', 'terletak', 'bertempat', 'berdomisili',
    'dapat', 'bisa', 'mampu', 'sanggup', 'kuasa', 'boleh', 'diperbolehkan',
    'berkata', 'berucap', 'mengatakan', 'mengucapkan', 'berbicara', 'bersabda',
    'telah', 'menceritakan', 'kepada', 'kami', 'dia', 'berkata', 'mengabarkan',
    'pernah', 'mendengar', 'bahwa', 'saya', 'shallallahu', 'alaihi', 'wasallam',
    'radhi', 'allahu', 'anhu', 'anha', 'anhum', 'semua', 'bagi', 'tiap', 'setiap',
    'barangsiapa', 'siapa', 'siapapun', 'mana', 'manapun', 'dimana', 'kemana',
    'darimana', 'bagaimana', 'mengapa', 'kenapa', 'kapan', 'bilamana', 'apakah',
    'berapa', 'seberapa'
}

# Semantic grouping patterns for Islamic terms
SEMANTIC_GROUPS = {
    'shalat': ['shalat', 'salat', 'sholat', 'sembahyang'],
    'puasa': ['puasa', 'shaum', 'shiyam'],
    'zakat': ['zakat', 'zakah'],
    'haji': ['haji', 'hajj', 'ibadah haji'],
    'umrah': ['umrah', 'ibadah umrah'],
    'jihad': ['jihad', 'berjihad'],
    'hijrah': ['hijrah', 'berhijrah'],
    'iman': ['iman', 'beriman', 'keimanan'],
    'islam': ['islam', 'keislaman'],
    'ihsan': ['ihsan', 'berbuat ihsan'],
    'tauhid': ['tauhid', 'mentauhidkan', 'ketauhidan'],
    'syirik': ['syirik', 'menyekutukan', 'persekutuan'],
    'halal': ['halal', 'dihalalkan', 'menghalalkan', 'diperbolehkan'],
    'haram': ['haram', 'diharamkan', 'mengharamkan', 'dilarang'],
    'riba': ['riba', 'bertransaksi riba', 'mengambil riba'],
    'zina': ['zina', 'berzina'],
    'khamr': ['khamr', 'minuman keras', 'arak'],
    'nikah': ['nikah', 'menikah', 'pernikahan', 'perkawinan'],
    'talak': ['talak', 'mencerai', 'perceraian'],
    'waris': ['waris', 'warisan', 'mewarisi'],
    'wasiat': ['wasiat', 'berwasiat'],
    'amanah': ['amanah', 'amanat', 'titipan'],
    'khianat': ['khianat', 'berkhianat', 'pengkhianatan'],
    'sabar': ['sabar', 'bersabar', 'kesabaran'],
    'syukur': ['syukur', 'bersyukur', 'mensyukuri'],
    'taubat': ['taubat', 'bertaubat', 'taubah'],
    'istighfar': ['istighfar', 'beristighfar', 'memohon ampun'],
    'doa': ['doa', 'berdoa', 'permohonan'],
    'dzikir': ['dzikir', 'berdzikir', 'mengingat allah'],
    'tasbih': ['tasbih', 'bertasbih'],
    'takbir': ['takbir', 'bertakbir'],
    'tahmid': ['tahmid', 'memuji allah'],
    'jihad': ['jihad', 'berjihad', 'perjuangan'],
    'syahid': ['syahid', 'mati syahid', 'kesyahidan'],
    'surga': ['surga', 'jannah', 'syurga'],
    'neraka': ['neraka', 'jahannam'],
    'akhirat': ['akhirat', 'hari akhir', 'kehidupan akhirat'],
    'dunia': ['dunia', 'kehidupan dunia', 'alam dunia'],
    'qiyamat': ['qiyamat', 'hari qiyamat', 'kiamat'],
    'malaikat': ['malaikat', 'angel'],
    'jin': ['jin', 'makhluk jin'],
    'syetan': ['syetan', 'setan', 'iblis'],
    'nabi': ['nabi', 'rasul', 'utusan allah'],
    'wudhu': ['wudhu', 'berwudhu', 'bersuci'],
    'tayammum': ['tayammum', 'bertayammum'],
    'mandi': ['mandi', 'mandi junub', 'mandi besar'],
    'najis': ['najis', 'kenajisan'],
    'suci': ['suci', 'kesucian', 'bersuci'],
    'quran': ['quran', 'alquran', 'kitab allah'],
    'hadis': ['hadis', 'hadits', 'sunnah'],
    'sunnah': ['sunnah', 'sunah'],
    'bid\'ah': ['bidah', 'perkara bidah'],
    'makruh': ['makruh', 'dimakruhkan'],
    'mubah': ['mubah', 'dibolehkan'],
    'wajib': ['wajib', 'diwajibkan', 'fardhu'],
    'mustahab': ['mustahab', 'disunahkan'],
    'akhlak': ['akhlak', 'budi pekerti', 'adab'],
    'ihlas': ['ikhlas', 'keikhlasan'],
    'niat': ['niat', 'berniat', 'maksud'],
    'pahala': ['pahala', 'ganjaran', 'balasan baik'],
    'dosa': ['dosa', 'maksiat', 'perbuatan dosa'],
    'ampun': ['ampun', 'ampunan', 'pengampunan'],
    'maaf': ['maaf', 'memaafkan'],
    'sedekah': ['sedekah', 'bersedekah', 'shadaqah'],
    'infaq': ['infaq', 'berinfaq'],
    'wakaf': ['wakaf', 'mewakafkan'],
    'fitrah': ['fitrah', 'zakat fitrah'],
    'ilmu': ['ilmu', 'ilmu pengetahuan', 'menuntut ilmu'],
    'ulama': ['ulama', 'ahli ilmu'],
    'fiqih': ['fiqih', 'fikih', 'hukum islam'],
    'fatwa': ['fatwa', 'keputusan hukum'],
    'ijtihad': ['ijtihad', 'berijtihad'],
    'taqlid': ['taqlid', 'mengikuti ulama'],
    'musyawarah': ['musyawarah', 'bermusyawarah'],
    'adil': ['adil', 'keadilan', 'berlaku adil'],
    'zalim': ['zalim', 'kezaliman', 'berbuat zalim'],
    'fasik': ['fasik', 'kefasikan'],
    'kafir': ['kafir', 'kekafiran'],
    'munafik': ['munafik', 'kemunafikan']
}

def normalize_text(text: str) -> str:
    """Normalize text for processing"""
    if pd.isna(text):
        return ""
    # Convert to lowercase and remove extra whitespace
    text = text.lower().strip()
    # Remove quotes and clean up
    text = re.sub(r'["""\'`]', '', text)
    # Normalize Arabic transliteration
    text = re.sub(r'shallallahu\s+\'?alaihi\s+wa?\s*sallam', 'saw', text)
    text = re.sub(r'radhi\s*allahu\s+(anhu|anha|anhum)', 'ra', text)
    # Remove excessive punctuation but keep basic sentence structure
    text = re.sub(r'[^\w\s\'-]', ' ', text)
    # Clean up multiple spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def generate_ngrams(text: str, max_n: int = MAX_NGRAM) -> list:
    """Generate n-grams from text"""
    words = text.split()
    ngrams = []
    
    for n in range(1, max_n + 1):
        for i in range(len(words) - n + 1):
            ngram = ' '.join(words[i:i+n])
            # Filter out ngrams that are just stopwords or very short
            if (not all(word in STOPWORDS for word in ngram.split()) and 
                len(ngram.strip()) > 2 and
                not re.match(r'^\d+$', ngram.strip())):  # exclude pure numbers
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
    """Extract terms that appear more than min_freq times"""
    print("🔍 Generating n-grams and counting frequencies...")
    
    all_ngrams = []
    for i, text in enumerate(texts):
        if i % 1000 == 0:
            print(f"   Processing text {i+1}/{len(texts)}")
        
        normalized = normalize_text(text)
        ngrams = generate_ngrams(normalized)
        all_ngrams.extend(ngrams)
    
    print(f"📈 Generated {len(all_ngrams)} total n-grams")
    
    # Count frequencies
    counter = Counter(all_ngrams)
    
    # Filter by minimum frequency
    frequent_terms = {term: count for term, count in counter.items() 
                     if count >= min_freq}
    
    print(f"🎯 Found {len(frequent_terms)} terms with frequency >= {min_freq}")
    return Counter(frequent_terms)

def calculate_similarity(term1: str, term2: str) -> float:
    """Calculate similarity between two terms"""
    # Check for substring relationships
    if term1 in term2 or term2 in term1:
        return 0.8
    
    # Use sequence matcher for string similarity
    similarity = difflib.SequenceMatcher(None, term1, term2).ratio()
    
    # Check for semantic similarity (same root words)
    words1 = set(term1.split())
    words2 = set(term2.split())
    
    if words1.intersection(words2):
        similarity = max(similarity, 0.6)
    
    return similarity

def group_similar_terms(frequent_terms: Counter, similarity_threshold: float = 0.7) -> dict:
    """Group semantically similar terms"""
    print("🔗 Grouping similar terms...")
    
    terms = list(frequent_terms.keys())
    grouped = defaultdict(list)
    processed = set()
    
    # First, apply predefined semantic groups
    for root_term, variants in SEMANTIC_GROUPS.items():
        group_members = []
        for term in terms:
            if any(variant in term for variant in variants):
                group_members.append(term)
                processed.add(term)
        
        if group_members:
            # Use the most frequent term as the key
            key_term = max(group_members, key=lambda x: frequent_terms[x])
            grouped[key_term] = sorted(set(group_members))
    
    # Then group remaining terms by similarity
    remaining_terms = [term for term in terms if term not in processed]
    
    for i, term1 in enumerate(remaining_terms):
        if term1 in processed:
            continue
            
        group = [term1]
        
        for term2 in remaining_terms[i+1:]:
            if term2 not in processed:
                if calculate_similarity(term1, term2) >= similarity_threshold:
                    group.append(term2)
                    processed.add(term2)
        
        if len(group) > 1:
            # Use the most frequent term as the key
            key_term = max(group, key=lambda x: frequent_terms[x])
            grouped[key_term] = sorted(set(group))
        else:
            # Single term group
            grouped[term1] = [term1]
        
        processed.add(term1)
    
    print(f"📋 Created {len(grouped)} term groups")
    return dict(grouped)

def save_results(grouped_terms: dict, output_path: Path):
    """Save results to JSON file"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(grouped_terms, f, ensure_ascii=False, indent=2)
    
    print(f"💾 Results saved to {output_path}")

def main():
    """Main extraction pipeline"""
    print("🚀 Starting Advanced Keyword Extraction for Hadits")
    print("=" * 50)
    
    # Load all CSV texts
    texts = load_csv_texts(CSV_DIR)
    
    if not texts:
        print("❌ No texts found. Please check the CSV files.")
        return
    
    # Extract frequent terms
    frequent_terms = extract_frequent_terms(texts, MIN_FREQUENCY)
    
    if not frequent_terms:
        print(f"❌ No terms found with frequency >= {MIN_FREQUENCY}")
        return
    
    # Group similar terms
    grouped_terms = group_similar_terms(frequent_terms)
    
    # Save results
    save_results(grouped_terms, OUTPUT_PATH)
    
    # Print summary
    print("\n" + "=" * 50)
    print("📊 EXTRACTION SUMMARY")
    print("=" * 50)
    print(f"Total hadis processed: {len(texts):,}")
    print(f"Terms with frequency >= {MIN_FREQUENCY}: {len(frequent_terms):,}")
    print(f"Final grouped terms: {len(grouped_terms):,}")
    print(f"Output file: {OUTPUT_PATH}")
    
    # Show top groups
    print("\n🔝 Top 10 groups by frequency:")
    sorted_groups = sorted(grouped_terms.items(), 
                          key=lambda x: frequent_terms.get(x[0], 0), 
                          reverse=True)
    
    for i, (key, variants) in enumerate(sorted_groups[:10]):
        freq = frequent_terms.get(key, 0)
        print(f"{i+1:2}. {key} ({freq}x): {variants[:3]}{'...' if len(variants) > 3 else ''}")

if __name__ == "__main__":
    main() 