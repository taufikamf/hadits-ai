"""
LLM service for Google Gemini API integration.
Inspired by Dify's LLM node implementation.
"""
import logging
import time
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
        
        # Initialize model with safety settings
        generation_config = {
            'temperature': 0.7,
            'top_p': 0.9,
            'top_k': 40,
            'max_output_tokens': 2048,
        }
        
        # Initialize model
        self.model = genai.GenerativeModel(
            model_name='gemini-1.5-flash',
            generation_config=generation_config
        )
        
        logger.info("Initialized Gemini LLM service")
    
    def generate_response(
        self, 
        query: str, 
        retrieved_docs: List[Dict[str, Any]], 
        system_prompt: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0
    ) -> str:
        """
        Generate response based on query and retrieved documents.
        
        Args:
            query: User's question in Indonesian
            retrieved_docs: List of retrieved hadits documents
            system_prompt: Optional system prompt
            max_retries: Maximum number of retries on failure
            retry_delay: Delay between retries in seconds
            
        Returns:
            Generated response from Gemini
        """
        if not query.strip():
            return "Mohon maaf, pertanyaan tidak boleh kosong."
            
        if not retrieved_docs:
            return "Mohon maaf, tidak ditemukan hadits yang relevan dengan pertanyaan Anda."
        
        retries = 0
        last_error = None
        
        while retries < max_retries:
            try:
                # Build context from retrieved documents
                context = self._build_context(retrieved_docs)
                
                # Create prompt
                prompt = self._create_prompt(query, context, system_prompt)
                
                # Generate response
                response = self.model.generate_content(prompt)
                
                # Check if response was blocked
                if not response.text:
                    logger.warning("Response was empty or blocked")
                    return "Mohon maaf, pertanyaan tidak dapat diproses karena alasan keamanan."
                
                return response.text
                
            except Exception as e:
                last_error = e
                logger.warning(f"Attempt {retries + 1} failed: {e}")
                retries += 1
                if retries < max_retries:
                    time.sleep(retry_delay)
                continue
        
        logger.error(f"Failed to generate response after {max_retries} attempts: {last_error}")
        return "Mohon maaf, terjadi kesalahan saat memproses pertanyaan. Silakan coba lagi nanti."
    
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
            
            # Format each hadits with clear structure
            context_part = f"[HADITS {i}]\n"
            context_part += f"Kitab: {kitab}\n"
            context_part += f"Relevansi: {score:.3f}\n"
            context_part += f"Arab: {arab}\n"
            context_part += f"Terjemahan: {terjemah}\n"
            
            context_parts.append(context_part)
        
        return "\n\n".join(context_parts)
    
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

[KONTEKS HADITS]
{context}

[PERTANYAAN]
{query}

[INSTRUKSI]
Berikan jawaban yang komprehensif berdasarkan hadits-hadits di atas. Sertakan rujukan yang jelas ke hadits yang relevan. Jika pertanyaan tidak dapat dijawab sepenuhnya dengan hadits yang tersedia, jelaskan keterbatasan tersebut dengan sopan."""
        
        return prompt
    
    def _get_default_system_prompt(self) -> str:
        """Get default system prompt for hadits Q&A"""
        return """Anda adalah asisten AI yang ahli dalam hadits dan ilmu agama Islam. Tugas Anda adalah menjawab pertanyaan berdasarkan hadits-hadits yang disediakan sebagai konteks.

[PANDUAN MENJAWAB]
1. Gunakan HANYA hadits yang disediakan dalam konteks untuk menjawab
2. Berikan jawaban dalam bahasa Indonesia yang jelas dan mudah dipahami
3. Sertakan rujukan kitab hadits untuk setiap hadits yang Anda kutip
4. Jika pertanyaan tidak dapat dijawab berdasarkan hadits yang ada, jelaskan dengan sopan
5. Berikan penjelasan yang mendalam tentang makna dan implikasi hadits
6. Hindari interpretasi yang tidak didukung oleh teks hadits
7. Jika ada perbedaan pendapat ulama, sebutkan dengan jelas
8. Jika pertanyaan sensitif atau kontroversial, berikan jawaban dengan bijak

[FORMAT JAWABAN]
1. Jawaban Singkat: Berikan jawaban langsung dan ringkas
2. Penjelasan Detail: Uraikan makna dan konteks hadits
3. Rujukan Hadits: Kutip bagian relevan dengan sumber kitab
4. Kesimpulan: Berikan hikmah atau pelajaran utama
5. Catatan: Jika ada keterbatasan atau hal yang perlu diperhatikan"""


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