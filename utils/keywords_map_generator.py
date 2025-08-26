"""
Keywords Map Generator for Hadith AI
====================================

This script generates a comprehensive keywords map from the hadith corpus using:
- N-gram candidate generation (freq ≥ 20)
- Embedding clustering with KMeans/HDBSCAN
- Noise filtering (sanad, conjunctions)
- Manual Islamic domain dictionary integration
- Final curation to data/keywords_map.json

Author: Hadith AI Team
Date: 2024
"""

import json
import re
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple
import os

# Try to import optional dependencies
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("Warning: pandas not available. Using basic CSV processing.")

try:
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not available. Clustering disabled.")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("Warning: numpy not available. Some features limited.")


class KeywordsMapGenerator:
    """
    Generates a comprehensive keywords map from hadith corpus.
    """
    
    def __init__(self, 
                 min_frequency: int = 20,
                 max_ngram: int = 3,
                 max_clusters: int = 50):
        self.min_frequency = min_frequency
        self.max_ngram = max_ngram
        self.max_clusters = max_clusters
        
        # Comprehensive Islamic terms dictionary
        self.islamic_terms = {
            # Prayer and worship
            'shalat', 'salat', 'sholat', 'solat', 'shalatnya', 'sujud', 'rukuk', 'takbir',
            'tahiyyat', 'taslim', 'qiblat', 'kiblat', 'jamaah', 'imam', 'makmum',
            'shalat subuh', 'shalat zhuhur', 'shalat ashar', 'shalat maghrib', 'shalat isya',
            'shalat jumat', 'shalat ied', 'shalat tarawih', 'shalat witir', 'shalat tahajud',
            'shalat dhuha', 'shalat istikharah', 'shalat jenazah', 'shalat gerhana',
            'shalat hujan', 'qiyamul lail', 'tadarus', 'tilawah', 'dhikir', 'doa',
            
            # Purification
            'wudhu', 'wudu', 'tayammum', 'ghusl', 'najis', 'suci', 'bersuci', 'thaharah',
            'istinja', 'istinjak', 'air', 'debu', 'tanah', 'batu', 'mani', 'madhi',
            'wadhi', 'haidh', 'nifas', 'istihadah', 'junub', 'hadats', 'kecil', 'besar',
            
            # Fasting
            'puasa', 'shaum', 'shiyam', 'sahur', 'iftar', 'berbuka', 'ramadan', 'fidyah',
            'kafarat', 'itikaf', 'lailatul qadr', 'puasa sunnah', 'puasa wajib',
            'puasa senin kamis', 'puasa daud', 'puasa arafah', 'puasa asyura',
            'puasa syawal', 'puasa muharram', 'qadha', 'nazar',
            
            # Charity and zakat
            'zakat', 'sadaqah', 'infaq', 'fitrah', 'mal', 'harta', 'nisab', 'haul',
            'mustahiq', 'asnaf', 'fakir', 'miskin', 'amil', 'muallaf', 'riqab',
            'gharim', 'fisabilillah', 'ibnu sabil', 'zakat emas', 'zakat perak',
            'zakat pertanian', 'zakat perdagangan', 'zakat ternak', 'zakat profesi',
            
            # Pilgrimage
            'haji', 'umrah', 'ihram', 'tawaf', 'sai', 'wukuf', 'arafah', 'muzdalifah',
            'mina', 'jumrah', 'tahallul', 'hady', 'dam', 'badal', 'mahram',
            'miqat', 'talbiyah', 'hajar aswad', 'multazam', 'hijr ismail',
            'maqam ibrahim', 'safar', 'marwah', 'haji tamattu', 'haji qiran',
            'haji ifrad', 'fidyah haji', 'ihsar',
            
            # Legal rulings
            'halal', 'haram', 'makruh', 'sunnah', 'mustahab', 'wajib', 'fardhu',
            'mubah', 'hukum', 'syariat', 'fiqih', 'fatwa', 'ijma', 'qiyas',
            'istihsan', 'maslahah', 'istishab', 'sadd dzariah', 'urf', 'ijtihat',
            'taklid', 'talfiq', 'tarjih', 'nasikh', 'mansukh', 'muhkam', 'mutasyabih',
            
            # Beliefs and theology
            'iman', 'islam', 'ihsan', 'tauhid', 'syirik', 'kufur', 'munafik',
            'allah', 'rasul', 'nabi', 'malaikat', 'kitab', 'akhirat', 'qadar',
            'takdir', 'rukun iman', 'rukun islam', 'asmaul husna', 'sifat allah',
            'af\'al allah', 'janji allah', 'ancaman allah', 'surga', 'neraka',
            'mahsyar', 'mizan', 'shirath', 'hisab', 'siksa kubur', 'azab',
            
            # Ethics and behavior
            'akhlaq', 'adab', 'birrul walidain', 'silaturahmi', 'amanah', 'jujur',
            'sabar', 'syukur', 'tawadhu', 'ikhlas', 'taqwa', 'takut', 'harap',
            'taubat', 'istighfar', 'muhasabah', 'muraqabah', 'ihsan', 'husnu khuluq',
            'mahmud', 'madzmum', 'sombong', 'iri', 'hasud', 'dengki', 'fitnah',
            
            # Marriage and family
            'nikah', 'kawin', 'talaq', 'rujuk', 'iddah', 'khulu', 'mubarat',
            'mahar', 'nafkah', 'wali', 'saksi', 'walimah', 'mut\'ah', 'tahlil',
            'zihar', 'ila', 'li\'an', 'qazaf', 'had', 'rajam', 'jilid', 'diyat',
            'qishash', 'ta\'zir', 'hikmah', 'poligami', 'khalwat',
            
            # Business and transactions
            'jual', 'beli', 'dagang', 'riba', 'gharar', 'tadlis', 'ijarah',
            'mudharabah', 'musyarakah', 'salam', 'istisna', 'wakalah', 'kafalah',
            'hiwalah', 'rahn', 'waqf', 'hibah', 'wasiat', 'waris', 'faraid',
            'ashobah', 'zawil furudh', 'mahjub', 'hajib', 'aqad', 'shighat',
            
            # Food and dietary laws
            'makanan', 'minuman', 'halal', 'haram', 'syubhat', 'zabihah',
            'sembelih', 'aqiqah', 'qurban', 'hewan', 'binatang', 'khamr',
            'arak', 'miras', 'babi', 'bangkai', 'darah', 'bisa', 'racun',
            
            # Time and calendar
            'waktu', 'miqat', 'fajar', 'subuh', 'syuruq', 'dhuha', 'zhuhur',
            'ashar', 'maghrib', 'isya', 'hijriah', 'qomariah', 'syamsiah',
            'bulan', 'tahun', 'hari', 'jumaat', 'sabtu', 'ahad', 'senin',
            'selasa', 'rabu', 'kamis', 'muharram', 'safar', 'rabiul awwal',
            'rabiul akhir', 'jumadil ula', 'jumadil akhir', 'rajab', 'syaban',
            'ramadan', 'syawal', 'dzulqadah', 'dzulhijjah',
            
            # Additional terms from Quranic Arabic and Wikidata
            'quraan', 'quran', 'ayat', 'surah', 'juz', 'hizb', 'ruku', 'sajdah',
            'tilawah', 'tahfizh', 'tadabbur', 'tafsir', 'ta\'wil', 'asbabun nuzul',
            'nasikh mansukh', 'makki', 'madani', 'muhkamat', 'mutasyabihat'
        }
        
        # Stopwords including Indonesian, Arabic, and hadith-specific terms
        self.stopwords = {
            # Indonesian stopwords
            'yang', 'dan', 'di', 'ke', 'dari', 'pada', 'dengan', 'untuk', 'adalah', 'akan',
            'telah', 'sudah', 'atau', 'juga', 'tidak', 'bila', 'jika', 'ketika', 'saat',
            'itu', 'ini', 'mereka', 'kita', 'kami', 'dia', 'ia', 'saya', 'anda', 'engkau',
            'kamu', 'kalian', 'beliau', 'ada', 'seperti', 'antara', 'semua', 'setiap',
            'bagi', 'oleh', 'karena', 'sebab', 'supaya', 'agar', 'hingga', 'sampai',
            'maka', 'lalu', 'kemudian', 'setelah', 'sebelum', 'selama', 'sambil', 'dalam',
            
            # Arabic stopwords and narrator chains
            'bin', 'abu', 'ibnu', 'ibn', 'al', 'an', 'as', 'ad', 'ar', 'az', 'ats', 'ath',
            'saw', 'ra', 'rah', 'radhiyallahu', 'anhu', 'anha', 'anhum', 'anhuma',
            'shallallahu', 'alaihi', 'wasallam', 'sallallahu', 'alaih', 'alayhi',
            
            # Hadith-specific narrator and transmission terms
            'hadits', 'hadist', 'riwayat', 'diriwayatkan', 'menceritakan', 'bercerita',
            'telah', 'kepada', 'kami', 'dari', 'hadathana', 'akhbarana', 'haddathana',
            'nabaa', 'anna', 'qala', 'qaala', 'anbaa', 'an', 'fi', 'min', 'ila',
            'wa', 'la', 'ma', 'li', 'bi', 'ala', 'inda', 'bayna', 'tahta', 'fauqa'
        }
        
        # Noise patterns to filter out
        self.noise_patterns = [
            r'.*\\bbin\\b.*',  # Narrator chains
            r'.*\\babu\\b.*',
            r'.*\\bibnu?\\b.*',
            r'.*\\bal[- ]\\w+',
            r'.*hadathana.*',
            r'.*akhbarana.*',
            r'.*bercerita.*',
            r'.*menceritakan.*',
            r'.*riwayat.*',
            r'^[0-9]+$',  # Pure numbers
            r'^[a-z]$',   # Single letters
            r'.*(ditulis|disebutkan|diriwayatkan).*'
        ]
    
    def normalize_text(self, text: str) -> str:
        """Enhanced text normalization."""
        if not text:
            return ""
        
        text = text.lower().strip()
        
        # Normalize Islamic phrases
        text = re.sub(r'shallallahu\\s+[\\\'\\"]?alaihi\\s+wa?\\s*sallam', 'saw', text)
        text = re.sub(r'sallallahu\\s+[\\\'\\"]?alaihi\\s+wa?\\s*sallam', 'saw', text)
        text = re.sub(r'radhi\\s*allahu\\s+(anhu|anha|anhum)', 'ra', text)
        text = re.sub(r'radhiyallahu\\s+(anhu|anha|anhum)', 'ra', text)
        
        # Remove punctuation but preserve apostrophes and hyphens in words
        text = re.sub(r'[^\\w\\s\\'-]', ' ', text)
        text = re.sub(r'\\s+', ' ', text)
        
        return text.strip()
    
    def is_noise(self, term: str) -> bool:
        """Check if term is noise that should be filtered out."""
        if not term or len(term.strip()) < 3:
            return True
        
        term = term.strip().lower()
        
        # Check stopwords
        if term in self.stopwords:
            return True
        
        # Check noise patterns
        for pattern in self.noise_patterns:
            if re.match(pattern, term):
                return True
        
        return False
    
    def generate_ngram_candidates(self, texts: List[str]) -> Counter:
        """Generate n-gram candidates from corpus with frequency filtering."""
        print("🔍 Generating n-gram candidates from corpus...")
        
        all_ngrams = []
        
        for i, text in enumerate(texts):
            if i % 1000 == 0 and i > 0:
                print(f"   Processed {i}/{len(texts)} texts...")
            
            normalized = self.normalize_text(text)
            words = normalized.split()
            
            # Generate n-grams
            for n in range(1, self.max_ngram + 1):
                for j in range(len(words) - n + 1):
                    ngram = ' '.join(words[j:j+n])
                    if not self.is_noise(ngram):
                        all_ngrams.append(ngram)
        
        # Count frequencies
        ngram_counts = Counter(all_ngrams)
        
        # Filter by minimum frequency
        frequent_ngrams = Counter({
            ngram: count for ngram, count in ngram_counts.items()
            if count >= self.min_frequency
        })
        
        print(f"✅ Generated {len(frequent_ngrams)} frequent n-grams")
        return frequent_ngrams
    
    def cluster_similar_terms(self, terms: List[str]) -> Dict[str, List[str]]:
        """Cluster similar terms using embedding similarity (mock implementation)."""
        print("🔄 Clustering similar terms...")
        
        if not SKLEARN_AVAILABLE:
            print("   Warning: Clustering disabled, using rule-based grouping")
            return self._rule_based_grouping(terms)
        
        try:
            # Use TF-IDF for basic clustering
            vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=1000)
            
            # Create corpus for TF-IDF
            corpus = [term.replace(' ', '_') for term in terms]
            if len(corpus) < 2:
                return self._rule_based_grouping(terms)
            
            tfidf_matrix = vectorizer.fit_transform(corpus)
            
            # Determine number of clusters
            n_clusters = min(self.max_clusters, len(terms) // 3, 50)
            if n_clusters < 2:
                return self._rule_based_grouping(terms)
            
            # Perform clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(tfidf_matrix)
            
            # Group terms by cluster
            clusters = defaultdict(list)
            for term, label in zip(terms, cluster_labels):
                clusters[f"cluster_{label}"].append(term)
            
            print(f"   Created {len(clusters)} clusters")
            return dict(clusters)
            
        except Exception as e:
            print(f"   Warning: Clustering failed ({e}), using rule-based grouping")
            return self._rule_based_grouping(terms)
    
    def _rule_based_grouping(self, terms: List[str]) -> Dict[str, List[str]]:
        """Fallback rule-based grouping of terms."""
        groups = defaultdict(list)
        
        # Group by Islamic categories
        categories = {
            'shalat': ['shalat', 'salat', 'sholat', 'solat', 'shalatnya'],
            'puasa': ['puasa', 'shaum', 'shiyam', 'sahur', 'iftar', 'berbuka'],
            'zakat': ['zakat', 'sadaqah', 'infaq', 'fitrah'],
            'haji': ['haji', 'umrah', 'ihram', 'tawaf', 'sai'],
            'wudhu': ['wudhu', 'wudu', 'tayammum', 'ghusl', 'bersuci'],
            'hukum': ['halal', 'haram', 'makruh', 'sunnah', 'wajib', 'mubah'],
            'iman': ['iman', 'islam', 'ihsan', 'tauhid', 'taqwa'],
            'nikah': ['nikah', 'kawin', 'talaq', 'rujuk', 'iddah'],
            'jual_beli': ['jual', 'beli', 'dagang', 'riba', 'mudharabah']
        }
        
        # Assign terms to categories
        uncategorized = []
        for term in terms:
            assigned = False
            for category, keywords in categories.items():
                if any(keyword in term.lower() for keyword in keywords):
                    groups[category].append(term)
                    assigned = True
                    break
            
            if not assigned:
                uncategorized.append(term)
        
        # Handle uncategorized terms
        if uncategorized:
            # Group by first word for multi-word terms
            for term in uncategorized:
                first_word = term.split()[0]
                groups[first_word].append(term)
        
        return dict(groups)
    
    def add_manual_islamic_terms(self, groups: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """Add manual Islamic terms dictionary to the groups."""
        print("📚 Adding manual Islamic terms dictionary...")
        
        # Create Islamic terms groups
        islamic_groups = {
            'shalat': [
                'shalat', 'salat', 'sholat', 'solat', 'shalatnya', 'sujud', 'rukuk', 'takbir',
                'shalat subuh', 'shalat zhuhur', 'shalat ashar', 'shalat maghrib', 'shalat isya',
                'shalat jumat', 'shalat ied', 'shalat tarawih', 'shalat witir', 'shalat tahajud'
            ],
            'wudhu': [
                'wudhu', 'wudu', 'tayammum', 'ghusl', 'najis', 'suci', 'bersuci', 'thaharah',
                'istinja', 'istinjak'
            ],
            'puasa': [
                'puasa', 'shaum', 'shiyam', 'sahur', 'iftar', 'berbuka', 'ramadan', 'fidyah',
                'kafarat', 'itikaf', 'lailatul qadr'
            ],
            'zakat': [
                'zakat', 'sadaqah', 'infaq', 'fitrah', 'mal', 'harta', 'nisab', 'haul',
                'mustahiq', 'asnaf', 'fakir', 'miskin'
            ],
            'haji': [
                'haji', 'umrah', 'ihram', 'tawaf', 'sai', 'wukuf', 'arafah', 'muzdalifah',
                'mina', 'jumrah', 'tahallul', 'hady', 'dam', 'badal'
            ],
            'hukum': [
                'halal', 'haram', 'makruh', 'sunnah', 'mustahab', 'wajib', 'fardhu',
                'mubah', 'hukum', 'syariat', 'fiqih', 'fatwa'
            ],
            'aqidah': [
                'iman', 'islam', 'ihsan', 'tauhid', 'syirik', 'kufur', 'munafik',
                'allah', 'rasul', 'nabi', 'malaikat', 'kitab', 'akhirat', 'qadar'
            ],
            'akhlaq': [
                'akhlaq', 'adab', 'birrul walidain', 'silaturahmi', 'amanah', 'jujur',
                'sabar', 'syukur', 'tawadhu', 'ikhlas', 'taqwa'
            ],
            'muamalah': [
                'jual', 'beli', 'dagang', 'riba', 'gharar', 'tadlis', 'ijarah',
                'mudharabah', 'musyarakah', 'salam', 'istisna', 'wakalah'
            ],
            'munakahat': [
                'nikah', 'kawin', 'talaq', 'rujuk', 'iddah', 'khulu', 'mubarat',
                'mahar', 'nafkah', 'wali', 'saksi', 'walimah'
            ]
        }
        
        # Merge with existing groups
        for category, terms in islamic_groups.items():
            if category in groups:
                # Combine and deduplicate
                groups[category] = list(set(groups[category] + terms))
            else:
                groups[category] = terms
        
        print(f"   Added {len(islamic_groups)} Islamic term categories")
        return groups
    
    def create_final_keywords_map(self, groups: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """Create final keywords map with canonical terms and variants."""
        print("🎯 Creating final keywords map...")
        
        keywords_map = {}
        
        for group_name, terms in groups.items():
            if not terms:
                continue
            
            # Choose canonical term (prefer Islamic terms, then longest)
            islamic_terms_in_group = [t for t in terms if t in self.islamic_terms]
            if islamic_terms_in_group:
                canonical = max(islamic_terms_in_group, key=len)
            else:
                canonical = max(terms, key=len)
            
            # Add all terms as variants
            variants = sorted(list(set(terms)))
            
            # Only include if we have meaningful variants
            if len(variants) > 0:
                keywords_map[canonical] = variants
        
        print(f"✅ Created keywords map with {len(keywords_map)} canonical terms")
        return keywords_map
    
    def generate_keywords_map(self, csv_dir: str, output_path: str = None) -> Dict[str, List[str]]:
        """
        Main method to generate keywords map from corpus.
        
        Args:
            csv_dir: Directory containing hadith CSV files
            output_path: Path to save the final keywords map
            
        Returns:
            Final keywords map
        """
        print("🚀 Starting keywords map generation...")
        
        # Load texts from CSV files
        texts = self._load_texts_from_csv(csv_dir)
        
        # Generate n-gram candidates
        frequent_ngrams = self.generate_ngram_candidates(texts)
        
        # Get terms for clustering
        terms = list(frequent_ngrams.keys())
        
        # Cluster similar terms
        clustered_groups = self.cluster_similar_terms(terms)
        
        # Add manual Islamic terms
        enhanced_groups = self.add_manual_islamic_terms(clustered_groups)
        
        # Create final keywords map
        final_map = self.create_final_keywords_map(enhanced_groups)
        
        # Save results
        if output_path:
            self._save_keywords_map(final_map, frequent_ngrams, output_path)
        
        print("🎉 Keywords map generation completed!")
        return final_map
    
    def _load_texts_from_csv(self, csv_dir: str) -> List[str]:
        """Load texts from CSV files."""
        print("📁 Loading texts from CSV files...")
        
        csv_path = Path(csv_dir)
        if not csv_path.exists():
            raise ValueError(f"CSV directory not found: {csv_dir}")
        
        all_texts = []
        
        for csv_file in csv_path.glob("*.csv"):
            try:
                if PANDAS_AVAILABLE:
                    df = pd.read_csv(csv_file)
                    if 'terjemah' in df.columns:
                        texts = df['terjemah'].dropna().tolist()
                        all_texts.extend(texts)
                        print(f"   ✅ Loaded {len(texts)} texts from {csv_file.name}")
                    else:
                        print(f"   ⚠️ No 'terjemah' column in {csv_file.name}")
                else:
                    # Fallback CSV processing without pandas
                    with open(csv_file, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        if lines:
                            header = lines[0].strip().split(',')
                            if 'terjemah' in header:
                                terjemah_idx = header.index('terjemah')
                                for line in lines[1:]:
                                    parts = line.strip().split(',')
                                    if len(parts) > terjemah_idx and parts[terjemah_idx]:
                                        all_texts.append(parts[terjemah_idx])
                                print(f"   ✅ Loaded {len(lines)-1} texts from {csv_file.name}")
                            else:
                                print(f"   ⚠️ No 'terjemah' column in {csv_file.name}")
            except Exception as e:
                print(f"   ❌ Error loading {csv_file.name}: {e}")
        
        if not all_texts:
            raise ValueError("No texts found in CSV files")
        
        print(f"📄 Total texts loaded: {len(all_texts)}")
        return all_texts
    
    def _save_keywords_map(self, keywords_map: Dict[str, List[str]], 
                          frequent_ngrams: Counter, output_path: str):
        """Save keywords map with metadata."""
        print(f"💾 Saving keywords map to {output_path}...")
        
        output_data = {
            'metadata': {
                'total_keywords': len(keywords_map),
                'min_frequency': self.min_frequency,
                'max_ngram': self.max_ngram,
                'max_clusters': self.max_clusters,
                'total_ngrams_analyzed': len(frequent_ngrams),
                'generation_method': 'hybrid_clustering_manual_curation',
                'islamic_terms_count': len(self.islamic_terms)
            },
            'keywords': keywords_map,
            'statistics': {
                'top_frequent_ngrams': dict(frequent_ngrams.most_common(20))
            }
        }
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Keywords map saved successfully!")


def main():
    """Main function to generate keywords map."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate keywords map from hadith corpus')
    parser.add_argument('--csv-dir', default='data/csv', 
                       help='Directory containing CSV files (default: data/csv)')
    parser.add_argument('--output', default='data/keywords_map.json',
                       help='Output path for keywords map (default: data/keywords_map.json)')
    parser.add_argument('--min-freq', type=int, default=20,
                       help='Minimum frequency for n-grams (default: 20)')
    parser.add_argument('--max-ngram', type=int, default=3,
                       help='Maximum n-gram size (default: 3)')
    parser.add_argument('--max-clusters', type=int, default=50,
                       help='Maximum number of clusters (default: 50)')
    
    args = parser.parse_args()
    
    # Create generator
    generator = KeywordsMapGenerator(
        min_frequency=args.min_freq,
        max_ngram=args.max_ngram,
        max_clusters=args.max_clusters
    )
    
    # Generate keywords map
    try:
        keywords_map = generator.generate_keywords_map(args.csv_dir, args.output)
        print(f"\\n🎉 Successfully generated keywords map with {len(keywords_map)} entries!")
        
        # Show sample entries
        print("\\n📋 Sample entries:")
        for i, (canonical, variants) in enumerate(list(keywords_map.items())[:5]):
            print(f"   {canonical}: {variants}")
        
    except Exception as e:
        print(f"❌ Error generating keywords map: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())