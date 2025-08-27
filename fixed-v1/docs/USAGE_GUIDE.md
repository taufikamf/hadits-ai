# Usage Guide - Enhanced Hadith AI Fixed V1

This guide provides practical examples and best practices for using the Enhanced Hadith AI system in various scenarios.

## 🎯 Common Use Cases

### 1. Islamic Education Website

**Scenario**: Integrate hadith search into an Islamic learning platform.

**Implementation**:
```javascript
class HadithSearchWidget {
  constructor(apiUrl) {
    this.apiUrl = apiUrl;
    this.sessionId = null;
  }
  
  async initialize() {
    // Create session for user
    const response = await fetch(`${this.apiUrl}/session/create`, {
      method: 'POST'
    });
    const data = await response.json();
    this.sessionId = data.data.session_id;
    
    // Show greeting
    this.showGreeting();
  }
  
  async searchHadith(query) {
    const response = await fetch(`${this.apiUrl}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query: query,
        session_id: this.sessionId,
        max_results: 5
      })
    });
    
    const data = await response.json();
    this.displayResults(data.data);
  }
  
  displayResults(response) {
    const resultsDiv = document.getElementById('hadith-results');
    
    if (response.success && response.results.length > 0) {
      resultsDiv.innerHTML = `
        <div class="response-message">${response.message}</div>
        <div class="results-container">
          ${response.results.map((result, index) => `
            <div class="hadith-result">
              <h4>Hadits ${index + 1}</h4>
              <p class="hadith-text">${result.document.terjemah}</p>
              <div class="hadith-meta">
                <span class="source">${result.document.source || ''}</span>
                <span class="keywords">Keywords: ${result.matched_keywords.join(', ')}</span>
              </div>
            </div>
          `).join('')}
        </div>
      `;
    } else {
      resultsDiv.innerHTML = `<div class="no-results">${response.message}</div>`;
    }
  }
}

// Usage
const hadithWidget = new HadithSearchWidget('http://localhost:5000');
hadithWidget.initialize();
```

### 2. Mobile App Integration

**Scenario**: Add hadith search to an Islamic mobile application.

**Flutter/Dart Example**:
```dart
class HadithService {
  final String baseUrl;
  String? sessionId;
  
  HadithService(this.baseUrl);
  
  Future<void> createSession() async {
    final response = await http.post(
      Uri.parse('$baseUrl/session/create'),
    );
    
    final data = json.decode(response.body);
    sessionId = data['data']['session_id'];
  }
  
  Future<HadithResponse> searchHadith(String query) async {
    final response = await http.post(
      Uri.parse('$baseUrl/chat'),
      headers: {'Content-Type': 'application/json'},
      body: json.encode({
        'query': query,
        'session_id': sessionId,
        'max_results': 5,
      }),
    );
    
    final data = json.decode(response.body);
    return HadithResponse.fromJson(data['data']);
  }
}

// Usage in Flutter widget
class HadithSearchPage extends StatefulWidget {
  @override
  _HadithSearchPageState createState() => _HadithSearchPageState();
}

class _HadithSearchPageState extends State<HadithSearchPage> {
  final HadithService service = HadithService('http://your-api-server:5000');
  final TextEditingController queryController = TextEditingController();
  List<HadithResult> results = [];
  bool isLoading = false;
  
  @override
  void initState() {
    super.initState();
    service.createSession();
  }
  
  void searchHadith() async {
    setState(() { isLoading = true; });
    
    try {
      final response = await service.searchHadith(queryController.text);
      setState(() {
        results = response.results;
        isLoading = false;
      });
    } catch (e) {
      setState(() { isLoading = false; });
      // Handle error
    }
  }
  
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: Text('Pencarian Hadits')),
      body: Column(
        children: [
          Padding(
            padding: EdgeInsets.all(16),
            child: TextField(
              controller: queryController,
              decoration: InputDecoration(
                hintText: 'Cari hadits tentang...',
                suffixIcon: IconButton(
                  icon: Icon(Icons.search),
                  onPressed: searchHadith,
                ),
              ),
            ),
          ),
          Expanded(
            child: isLoading
              ? Center(child: CircularProgressIndicator())
              : ListView.builder(
                  itemCount: results.length,
                  itemBuilder: (context, index) {
                    final result = results[index];
                    return Card(
                      margin: EdgeInsets.all(8),
                      child: ListTile(
                        title: Text(result.document.terjemah),
                        subtitle: Text('Kata kunci: ${result.matchedKeywords.join(', ')}'),
                        trailing: Text('${(result.score * 100).toInt()}%'),
                      ),
                    );
                  },
                ),
          ),
        ],
      ),
    );
  }
}
```

### 3. Command Line Tool

**Scenario**: Create a CLI tool for hadith research.

**Python Implementation**:
```python
#!/usr/bin/env python3
"""
Hadith AI CLI Tool - Enhanced Fixed V1
"""

import argparse
import requests
import json
from typing import Optional

class HadithCLI:
    def __init__(self, api_url: str = "http://localhost:5000"):
        self.api_url = api_url
        self.session_id: Optional[str] = None
    
    def create_session(self):
        """Create a new session."""
        response = requests.post(f"{self.api_url}/session/create")
        data = response.json()
        self.session_id = data['data']['session_id']
        print(f"📱 Session created: {self.session_id}")
    
    def search(self, query: str, max_results: int = 5, show_scores: bool = False):
        """Search for hadith."""
        payload = {
            "query": query,
            "max_results": max_results
        }
        
        if self.session_id:
            payload["session_id"] = self.session_id
        
        response = requests.post(f"{self.api_url}/chat", json=payload)
        data = response.json()
        
        if data['success']:
            self.display_results(data['data'], show_scores)
        else:
            print(f"❌ Error: {data['error']}")
    
    def display_results(self, response_data, show_scores: bool = False):
        """Display search results."""
        print(f"\n🔍 Query: {response_data.get('query_analysis', {}).get('original_query', 'N/A')}")
        print(f"⏱️  Response time: {response_data['response_time_ms']:.1f}ms")
        print(f"📊 Found {len(response_data['results'])} results")
        print("=" * 80)
        
        for i, result in enumerate(response_data['results'], 1):
            doc = result['document']
            print(f"\n[{i}] {doc.get('source', 'Unknown Source')}")
            print(f"📖 {doc['terjemah']}")
            
            if result['matched_keywords']:
                print(f"🔑 Keywords: {', '.join(result['matched_keywords'])}")
            
            if show_scores:
                print(f"📈 Score: {result['score']:.3f} "
                      f"(Semantic: {result['semantic_score']:.3f}, "
                      f"Keyword: {result['keyword_score']:.3f}, "
                      f"Literal: {result['literal_overlap_score']:.3f})")
            
            print("-" * 80)
        
        # Show query analysis if available
        if 'query_analysis' in response_data:
            analysis = response_data['query_analysis']
            print(f"\n🧠 Query Analysis:")
            print(f"   Processed: {analysis.get('processed_query', 'N/A')}")
            print(f"   Key terms: {', '.join(analysis.get('key_terms', []))}")
            print(f"   Islamic context: {analysis.get('islamic_context_strength', 0):.2f}")
            print(f"   Has question: {analysis.get('has_question', False)}")
    
    def interactive_mode(self):
        """Run in interactive mode."""
        print("🕌 Hadith AI CLI - Interactive Mode")
        print("Type 'quit' to exit, 'help' for commands")
        
        self.create_session()
        
        while True:
            try:
                query = input("\n💬 Ask about hadith: ").strip()
                
                if query.lower() in ['quit', 'exit', 'q']:
                    print("👋 Goodbye!")
                    break
                elif query.lower() == 'help':
                    self.show_help()
                elif query.lower() == 'stats':
                    self.show_session_stats()
                elif query:
                    self.search(query)
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def show_help(self):
        """Show help information."""
        print("""
📋 Available Commands:
  help    - Show this help
  stats   - Show session statistics
  quit    - Exit the program
  
🔍 Search Examples:
  "hukum shalat jumat"
  "cara berwudhu yang benar"
  "zakat fitrah"
  "puasa ramadan"
  "berbakti kepada orang tua"
        """)
    
    def show_session_stats(self):
        """Show session statistics."""
        if not self.session_id:
            print("❌ No active session")
            return
        
        response = requests.get(f"{self.api_url}/session/{self.session_id}/stats")
        data = response.json()
        
        if data['success']:
            stats = data['data']
            print(f"""
📊 Session Statistics:
   Session ID: {stats['session_id']}
   Queries: {stats['query_count']}
   Total results: {stats['total_results_returned']}
   Avg results per query: {stats['avg_results_per_query']:.1f}
   Created: {stats['created_at']}
   Last activity: {stats['last_activity']}
            """)
        else:
            print(f"❌ Error getting stats: {data['error']}")

def main():
    parser = argparse.ArgumentParser(description="Hadith AI CLI Tool")
    parser.add_argument("--api-url", default="http://localhost:5000", 
                       help="API server URL")
    parser.add_argument("--query", "-q", help="Single query to execute")
    parser.add_argument("--max-results", "-n", type=int, default=5,
                       help="Maximum results to return")
    parser.add_argument("--show-scores", "-s", action="store_true",
                       help="Show relevance scores")
    parser.add_argument("--interactive", "-i", action="store_true",
                       help="Run in interactive mode")
    
    args = parser.parse_args()
    
    cli = HadithCLI(args.api_url)
    
    # Check API health
    try:
        response = requests.get(f"{args.api_url}/health", timeout=5)
        if response.status_code != 200:
            print(f"⚠️  API health check failed: {response.status_code}")
            return 1
    except requests.RequestException as e:
        print(f"❌ Cannot connect to API at {args.api_url}: {e}")
        return 1
    
    if args.interactive:
        cli.interactive_mode()
    elif args.query:
        cli.create_session()
        cli.search(args.query, args.max_results, args.show_scores)
    else:
        parser.print_help()
    
    return 0

if __name__ == "__main__":
    exit(main())
```

**Usage**:
```bash
# Single query
python hadith_cli.py -q "hukum zakat fitrah" -n 3 -s

# Interactive mode
python hadith_cli.py -i

# Different API server
python hadith_cli.py --api-url http://production-server:5000 -i
```

### 4. Web Chat Interface

**Scenario**: Create a web-based chat interface.

**HTML + JavaScript**:
```html
<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Hadith AI Chat</title>
    <style>
        .chat-container {
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            font-family: Arial, sans-serif;
        }
        
        .messages {
            height: 400px;
            overflow-y: auto;
            border: 1px solid #ddd;
            padding: 10px;
            margin-bottom: 20px;
            background-color: #f9f9f9;
        }
        
        .message {
            margin-bottom: 15px;
            padding: 10px;
            border-radius: 8px;
        }
        
        .user-message {
            background-color: #007bff;
            color: white;
            text-align: right;
        }
        
        .bot-message {
            background-color: white;
            border: 1px solid #ddd;
        }
        
        .hadith-result {
            background-color: #f8f9fa;
            border-left: 4px solid #28a745;
            padding: 10px;
            margin: 5px 0;
        }
        
        .input-container {
            display: flex;
            gap: 10px;
        }
        
        .input-container input {
            flex: 1;
            padding: 10px;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        
        .input-container button {
            padding: 10px 20px;
            background-color: #007bff;
            color: white;
            border: none;
            border-radius: 4px;
            cursor: pointer;
        }
        
        .loading {
            text-align: center;
            color: #6c757d;
            font-style: italic;
        }
    </style>
</head>
<body>
    <div class="chat-container">
        <h1>🕌 Hadith AI Chat</h1>
        <div id="messages" class="messages"></div>
        <div class="input-container">
            <input type="text" id="query-input" placeholder="Tanyakan tentang hadits..." />
            <button onclick="sendMessage()">Kirim</button>
        </div>
    </div>

    <script>
        class HadithChat {
            constructor() {
                this.apiUrl = 'http://localhost:5000';
                this.sessionId = null;
                this.messagesContainer = document.getElementById('messages');
                this.queryInput = document.getElementById('query-input');
                
                this.init();
            }
            
            async init() {
                // Create session
                await this.createSession();
                
                // Get greeting
                const greeting = await this.getGreeting();
                this.addMessage(greeting.message, 'bot');
                
                // Setup enter key listener
                this.queryInput.addEventListener('keypress', (e) => {
                    if (e.key === 'Enter') {
                        this.sendMessage();
                    }
                });
            }
            
            async createSession() {
                try {
                    const response = await fetch(`${this.apiUrl}/session/create`, {
                        method: 'POST'
                    });
                    const data = await response.json();
                    this.sessionId = data.data.session_id;
                } catch (error) {
                    console.error('Error creating session:', error);
                }
            }
            
            async getGreeting() {
                try {
                    const response = await fetch(`${this.apiUrl}/greeting?session_id=${this.sessionId}`);
                    const data = await response.json();
                    return data.data;
                } catch (error) {
                    console.error('Error getting greeting:', error);
                    return { message: 'Selamat datang di Hadith AI!' };
                }
            }
            
            async sendMessage() {
                const query = this.queryInput.value.trim();
                if (!query) return;
                
                // Add user message
                this.addMessage(query, 'user');
                this.queryInput.value = '';
                
                // Show loading
                const loadingElement = this.addMessage('Mencari hadits...', 'bot', true);
                
                try {
                    const response = await fetch(`${this.apiUrl}/chat`, {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({
                            query: query,
                            session_id: this.sessionId,
                            max_results: 3
                        })
                    });
                    
                    const data = await response.json();
                    
                    // Remove loading message
                    loadingElement.remove();
                    
                    if (data.success) {
                        this.addHadithResponse(data.data);
                    } else {
                        this.addMessage(data.error || 'Terjadi kesalahan', 'bot');
                    }
                    
                } catch (error) {
                    loadingElement.remove();
                    this.addMessage('Tidak dapat terhubung ke server', 'bot');
                }
            }
            
            addMessage(text, sender, isLoading = false) {
                const messageDiv = document.createElement('div');
                messageDiv.className = `message ${sender}-message`;
                if (isLoading) messageDiv.className += ' loading';
                messageDiv.textContent = text;
                
                this.messagesContainer.appendChild(messageDiv);
                this.messagesContainer.scrollTop = this.messagesContainer.scrollHeight;
                
                return messageDiv;
            }
            
            addHadithResponse(responseData) {
                const messageDiv = document.createElement('div');
                messageDiv.className = 'message bot-message';
                
                let html = `<p>${responseData.message}</p>`;
                
                if (responseData.results && responseData.results.length > 0) {
                    html += '<div>';
                    responseData.results.forEach((result, index) => {
                        html += `
                            <div class="hadith-result">
                                <strong>Hadits ${index + 1}:</strong><br>
                                ${result.document.terjemah}
                                ${result.matched_keywords.length > 0 ? 
                                  `<br><small><strong>Kata kunci:</strong> ${result.matched_keywords.join(', ')}</small>` : 
                                  ''}
                            </div>
                        `;
                    });
                    html += '</div>';
                }
                
                messageDiv.innerHTML = html;
                this.messagesContainer.appendChild(messageDiv);
                this.messagesContainer.scrollTop = this.messagesContainer.scrollHeight;
            }
        }
        
        // Initialize chat
        let chat;
        document.addEventListener('DOMContentLoaded', () => {
            chat = new HadithChat();
        });
        
        function sendMessage() {
            chat.sendMessage();
        }
    </script>
</body>
</html>
```

## 🎛️ Configuration Examples

### 1. High-Precision Configuration

For academic research requiring high accuracy:

```python
from service import ServiceConfig, RetrievalConfig

academic_config = ServiceConfig(
    retrieval_config=RetrievalConfig(
        top_k=100,  # More candidates
        min_score_threshold=0.3,  # Higher threshold
        semantic_weight=0.8,  # Favor semantic similarity
        keyword_weight=0.15,
        literal_overlap_weight=0.05,
        auto_adaptive_min_match=False,  # Fixed filtering
        base_min_match=0.6  # High precision
    ),
    max_results_display=3,  # Fewer, higher quality results
    min_confidence_threshold=0.3,
    high_confidence_threshold=0.7
)
```

### 2. High-Recall Configuration

For general education where finding something is better than nothing:

```python
educational_config = ServiceConfig(
    retrieval_config=RetrievalConfig(
        top_k=50,
        min_score_threshold=0.05,  # Lower threshold
        semantic_weight=0.6,
        keyword_weight=0.25,
        literal_overlap_weight=0.15,
        auto_adaptive_min_match=True,  # Adaptive filtering
        base_min_match=0.2  # Lower precision, higher recall
    ),
    max_results_display=7,  # More results
    min_confidence_threshold=0.05
)
```

### 3. Fast Response Configuration

For real-time chat applications:

```python
fast_config = ServiceConfig(
    retrieval_config=RetrievalConfig(
        top_k=20,  # Fewer candidates for speed
        enable_reranking=False,  # Skip reranking
        enable_query_analysis=False,  # Skip analysis
    ),
    max_results_display=3,
    enable_analytics=False  # Skip logging for speed
)
```

## 📊 Performance Optimization

### 1. Batch Processing

For processing multiple queries efficiently:

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

class BatchHadithProcessor:
    def __init__(self, api_url):
        self.api_url = api_url
        self.session = requests.Session()
    
    def process_single_query(self, query):
        response = self.session.post(f"{self.api_url}/query", json={
            "query": query,
            "max_results": 5
        })
        return response.json()
    
    def process_batch(self, queries, max_workers=5):
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(self.process_single_query, queries))
        return results

# Usage
processor = BatchHadithProcessor("http://localhost:5000")
queries = [
    "hukum shalat jumat",
    "cara berwudhu",
    "zakat fitrah"
]
results = processor.process_batch(queries)
```

### 2. Caching Strategy

For frequently asked questions:

```python
import hashlib
import json
from functools import lru_cache

class CachedHadithClient:
    def __init__(self, api_url, cache_size=100):
        self.api_url = api_url
        self.cache = {}
        self.cache_size = cache_size
    
    def _cache_key(self, query, max_results):
        return hashlib.md5(f"{query}:{max_results}".encode()).hexdigest()
    
    def search(self, query, max_results=5):
        cache_key = self._cache_key(query, max_results)
        
        # Check cache first
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Make API call
        response = requests.post(f"{self.api_url}/query", json={
            "query": query,
            "max_results": max_results
        })
        result = response.json()
        
        # Cache result
        if len(self.cache) >= self.cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        self.cache[cache_key] = result
        return result
```

## 🔍 Query Best Practices

### 1. Effective Query Patterns

**Good Queries**:
```
✅ "hukum shalat jumat bagi wanita"
✅ "cara berwudhu yang benar"
✅ "zakat fitrah dan zakat mal"
✅ "puasa ramadan bagi muslimah"
✅ "berbakti kepada orang tua"
```

**Less Effective Queries**:
```
❌ "shalat" (too vague)
❌ "bagaimana islam?" (too broad)
❌ "hadits nabi" (not specific)
❌ "hukum" (needs context)
```

### 2. Query Enhancement Tips

For better results, help users formulate better queries:

```javascript
function enhanceQuery(query) {
    const suggestions = {
        'shalat': 'Try: "hukum shalat", "cara shalat", "shalat berjamaah"',
        'puasa': 'Try: "hukum puasa", "puasa ramadan", "batal puasa"',
        'zakat': 'Try: "zakat fitrah", "zakat mal", "cara menghitung zakat"',
        'nikah': 'Try: "hukum nikah", "syarat nikah", "akad nikah"'
    };
    
    const words = query.toLowerCase().split(' ');
    for (const word of words) {
        if (suggestions[word]) {
            return {
                enhanced: false,
                suggestion: suggestions[word]
            };
        }
    }
    
    return { enhanced: true };
}
```

### 3. Handling No Results

```javascript
function handleNoResults(query, response) {
    const alternatives = [
        'Try using more specific Islamic terms',
        'Check spelling of Arabic/Islamic words',
        'Use Indonesian terms instead of Arabic',
        'Try broader or related topics'
    ];
    
    return {
        message: response.message,
        suggestions: alternatives,
        relatedQueries: generateRelatedQueries(query)
    };
}

function generateRelatedQueries(query) {
    const commonTopics = {
        'shalat': ['shalat berjamaah', 'shalat sunnah', 'waktu shalat'],
        'puasa': ['puasa sunnah', 'berbuka puasa', 'sahur'],
        'zakat': ['nisab zakat', 'mustahiq zakat', 'zakat profesi']
    };
    
    for (const [topic, related] of Object.entries(commonTopics)) {
        if (query.toLowerCase().includes(topic)) {
            return related;
        }
    }
    
    return ['hukum shalat', 'cara berwudhu', 'zakat fitrah'];
}
```

## 📈 Analytics and Monitoring

### 1. Track Usage Patterns

```python
import json
from collections import Counter
from datetime import datetime, timedelta

class HadithAnalytics:
    def __init__(self, log_file):
        self.log_file = log_file
    
    def parse_logs(self, days=7):
        cutoff_date = datetime.now() - timedelta(days=days)
        queries = []
        
        with open(self.log_file, 'r') as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    if entry['event_type'] == 'query_processed':
                        timestamp = datetime.fromisoformat(entry['timestamp'])
                        if timestamp > cutoff_date:
                            queries.append(entry)
                except:
                    continue
        
        return queries
    
    def get_popular_queries(self, limit=10):
        queries = self.parse_logs()
        query_texts = [q['query']['original'] for q in queries]
        return Counter(query_texts).most_common(limit)
    
    def get_performance_stats(self):
        queries = self.parse_logs()
        response_times = [q['performance']['response_time_ms'] for q in queries]
        
        return {
            'total_queries': len(queries),
            'avg_response_time': sum(response_times) / len(response_times),
            'max_response_time': max(response_times),
            'min_response_time': min(response_times)
        }
    
    def get_success_rate(self):
        queries = self.parse_logs()
        successful = [q for q in queries if q['results']['count'] > 0]
        return len(successful) / len(queries) if queries else 0

# Usage
analytics = HadithAnalytics('logs/hadith_ai_analytics.jsonl')
print("Popular queries:", analytics.get_popular_queries())
print("Performance:", analytics.get_performance_stats())
print("Success rate:", f"{analytics.get_success_rate():.1%}")
```

### 2. Monitor System Health

```python
def health_check_script():
    """Automated health monitoring script."""
    api_url = "http://localhost:5000"
    
    try:
        # Basic health check
        response = requests.get(f"{api_url}/health", timeout=10)
        health_data = response.json()
        
        if health_data['data']['status'] != 'healthy':
            send_alert(f"Hadith AI service is {health_data['data']['status']}")
        
        # Test query
        test_response = requests.post(f"{api_url}/query", json={
            "query": "test shalat",
            "max_results": 1
        }, timeout=30)
        
        if test_response.status_code != 200:
            send_alert(f"Hadith AI query test failed: {test_response.status_code}")
        
        test_data = test_response.json()
        if test_data['data']['response_time_ms'] > 5000:  # 5 seconds
            send_alert(f"Hadith AI response time too slow: {test_data['data']['response_time_ms']}ms")
        
        print(f"✅ Health check passed at {datetime.now()}")
        
    except Exception as e:
        send_alert(f"Hadith AI health check failed: {e}")

def send_alert(message):
    """Send alert notification."""
    print(f"🚨 ALERT: {message}")
    # Implement your alerting mechanism here (email, Slack, etc.)
```

This usage guide provides practical examples for integrating and using the Enhanced Hadith AI Fixed V1 system across different platforms and scenarios.
