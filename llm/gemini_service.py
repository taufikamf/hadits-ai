"""
LLM service for Google Gemini API integration.
Inspired by Dify's LLM node implementation.
"""
import logging
from typing import List, Dict, Any, Optional
import google.generativeai as genai

from config import settings

logger = logging.getLogger(__name__)


class GeminiLLMService:
    """
    Google Gemini LLM service for generating responses.
    Inspired by Dify's LLM integration pattern.
    """
    
    def __init__(self):
        # Configure Gemini API
        genai.configure(api_key=settings.gemini_api_key)
        
        # Initialize model
        self.model = genai.GenerativeModel('gemini-1.5-flash')
        
        logger.info("Initialized Gemini LLM service")
    
    def generate_response(
        self, 
        query: str, 
        retrieved_docs: List[Dict[str, Any]], 
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Generate response based on query and retrieved documents.
        
        Args:
            query: User's question in Indonesian
            retrieved_docs: List of retrieved hadits documents
            system_prompt: Optional system prompt
            
        Returns:
            Generated response from Gemini
        """
        try:
            # Build context from retrieved documents
            context = self._build_context(retrieved_docs)
            
            # Create prompt
            prompt = self._create_prompt(query, context, system_prompt)
            
            # Generate response
            response = self.model.generate_content(prompt)
            
            return response.text
            
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            raise
    
    def _build_context(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        """Build context string from retrieved documents"""
        if not retrieved_docs:
            return "Tidak ada hadits yang ditemukan."
        
        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            kitab = doc.get('kitab', 'Unknown')
            arab = doc.get('arab', '')
            terjemah = doc.get('terjemah', '')
            score = doc.get('score', 0)
            
            context_part = f"Hadits {i} (Relevansi: {score:.3f}):\n"
            context_part += f"Kitab: {kitab}\n"
            context_part += f"Arab: {arab}\n"
            context_part += f"Terjemahan: {terjemah}\n"
            
            context_parts.append(context_part)
        
        return "\n".join(context_parts)
    
    def _create_prompt(
        self, 
        query: str, 
        context: str, 
        system_prompt: Optional[str] = None
    ) -> str:
        """Create prompt for Gemini"""
        if system_prompt is None:
            system_prompt = self._get_default_system_prompt()
        
        prompt = f"""{system_prompt}

KONTEKS HADITS:
{context}

PERTANYAAN USER:
{query}

INSTRUKSI:
Berikan jawaban yang komprehensif berdasarkan hadits-hadits di atas. Sertakan rujukan yang jelas ke hadits yang relevan."""
        
        return prompt
    
    def _get_default_system_prompt(self) -> str:
        """Get default system prompt for hadits Q&A"""
        return """Anda adalah asisten AI yang ahli dalam hadits dan ilmu agama Islam. Tugas Anda adalah menjawab pertanyaan berdasarkan hadits-hadits yang disediakan sebagai konteks.

PANDUAN MENJAWAB:
1. Gunakan HANYA hadits yang disediakan dalam konteks untuk menjawab
2. Berikan jawaban dalam bahasa Indonesia yang jelas dan mudah dipahami
3. Sertakan rujukan kitab hadits untuk setiap hadits yang Anda kutip
4. Jika pertanyaan tidak dapat dijawab berdasarkan hadits yang ada, jelaskan dengan sopan
5. Berikan penjelasan yang mendalam tentang makna dan implikasi hadits
6. Hindari interpretasi yang tidak didukung oleh teks hadits

FORMAT JAWABAN:
- Mulai dengan jawaban singkat
- Berikan penjelasan detail berdasarkan hadits
- Sertakan kutipan hadits yang relevan dengan rujukan kitab
- Akhiri dengan kesimpulan atau hikmah dari hadits tersebut"""


# Global LLM service instance
_llm_service: Optional[GeminiLLMService] = None


def get_llm_service() -> GeminiLLMService:
    """Get global LLM service instance"""
    global _llm_service
    if _llm_service is None:
        _llm_service = GeminiLLMService()
    return _llm_service


# Example usage
if __name__ == "__main__":
    # Test the LLM service
    llm_service = get_llm_service()
    
    # Test documents
    test_docs = [
        {
            'kitab': 'Shahih Bukhari',
            'arab': 'إِنَّمَا الْأَعْمَالُ بِالنِّيَّاتِ',
            'terjemah': 'Sesungguhnya setiap perbuatan tergantung niatnya',
            'score': 0.95
        }
    ]
    
    # Test query
    query = "Apa pentingnya niat dalam Islam?"
    
    response = llm_service.generate_response(query, test_docs)
    print(f"Response: {response}") 