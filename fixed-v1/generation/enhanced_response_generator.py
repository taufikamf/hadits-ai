"""
Enhanced Response Generator - Fixed V1
====================================

Advanced response generation module with LLM integration for natural hadith conversations.
Supports multiple LLM providers, streaming responses, and adaptive formatting.

Features:
- Google Gemini AI integration
- OpenAI ChatGPT integration (optional)
- Streaming response generation
- Context-aware prompting
- Islamic response formatting
- Multiple response modes
- Response quality monitoring

Author: Hadith AI Team - Fixed V1
Date: 2024
"""

import json
import logging
import time
import asyncio
from typing import Dict, List, Optional, Any, AsyncGenerator, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

# LLM integrations
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Import retrieval results
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from retrieval.enhanced_retrieval_system import RetrievalResult

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMProvider(Enum):
    """Available LLM providers."""
    GEMINI = "gemini"
    OPENAI = "openai"
    LOCAL = "local"
    NONE = "none"


class ResponseMode(Enum):
    """Response generation modes."""
    COMPREHENSIVE = "comprehensive"  # Full detailed response with all hadits
    SUMMARY = "summary"              # Summarized response with key hadits
    DIRECT = "direct"                # Direct answer with minimal context
    CONVERSATIONAL = "conversational" # Natural conversation style


@dataclass
class GenerationConfig:
    """Configuration for response generation."""
    # LLM provider settings
    llm_provider: LLMProvider = LLMProvider.GEMINI
    gemini_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    gemini_model: str = "gemini-2.0-flash-exp"
    openai_model: str = "gpt-4o-mini"
    
    # Generation parameters
    temperature: float = 0.3
    max_tokens: int = 2500
    top_p: float = 0.9
    top_k: int = 40
    
    # Response formatting
    response_mode: ResponseMode = ResponseMode.COMPREHENSIVE
    max_hadits_display: int = 5
    max_arabic_length: int = 300
    max_translation_length: int = 800
    max_context_length: int = 8000
    
    # Streaming settings
    enable_streaming: bool = True
    chunk_delay_ms: int = 10
    
    # Quality controls
    min_response_length: int = 50
    max_response_length: int = 4000
    filter_inappropriate_content: bool = True
    
    # Language settings
    response_language: str = "id"  # Indonesian
    include_arabic: bool = True
    include_transliteration: bool = False


@dataclass
class GenerationResult:
    """Result from response generation."""
    success: bool
    content: str = ""
    response_mode: ResponseMode = ResponseMode.COMPREHENSIVE
    generation_time_ms: float = 0
    token_count: int = 0
    provider_used: LLMProvider = LLMProvider.NONE
    metadata: Dict[str, Any] = field(default_factory=dict)
    error_message: str = ""


class EnhancedResponseGenerator:
    """
    Advanced response generator with multiple LLM providers and streaming support.
    """
    
    def __init__(self, config: GenerationConfig = None):
        """
        Initialize the response generator.
        
        Args:
            config (GenerationConfig): Generation configuration
        """
        self.config = config or GenerationConfig()
        
        # Initialize LLM providers
        self.gemini_model = None
        self.openai_client = None
        
        self._initialize_providers()
        
        # System prompts
        self.system_prompts = self._load_system_prompts()
        
        logger.info(f"Enhanced Response Generator initialized with provider: {self.config.llm_provider.value}")
    
    def _initialize_providers(self):
        """Initialize available LLM providers."""
        try:
            # Initialize Gemini
            if (self.config.llm_provider == LLMProvider.GEMINI and 
                GEMINI_AVAILABLE and self.config.gemini_api_key):
                
                genai.configure(api_key=self.config.gemini_api_key)
                self.gemini_model = genai.GenerativeModel(self.config.gemini_model)
                logger.info("Gemini AI initialized successfully")
            
            # Initialize OpenAI
            if (self.config.llm_provider == LLMProvider.OPENAI and 
                OPENAI_AVAILABLE and self.config.openai_api_key):
                
                self.openai_client = openai.OpenAI(api_key=self.config.openai_api_key)
                logger.info("OpenAI initialized successfully")
                
        except Exception as e:
            logger.error(f"Error initializing LLM providers: {e}")
            self.config.llm_provider = LLMProvider.NONE
    
    def _load_system_prompts(self) -> Dict[ResponseMode, str]:
        """Load system prompts for different response modes."""
        
        base_rules = """
Anda adalah asisten AI yang ahli dalam hadits Islam dengan pengetahuan mendalam tentang ajaran Nabi Muhammad shallallahu 'alaihi wa sallam.

ATURAN PENTING:
- Gunakan Bahasa Indonesia yang formal dan sopan
- Awali dengan "Assalamu'alaikum" jika sesuai konteks
- Selalu tampilkan SEMUA hadits yang tersedia dalam konteks
- Format hadits dengan jelas dan terstruktur
- Berikan ringkasan yang mudah dipahami
- Akhiri dengan pertanyaan follow-up
- JANGAN gunakan tabel markdown (gunakan format list)
- Jika tidak yakin, minta klarifikasi
"""
        
        comprehensive_prompt = base_rules + """
FORMAT RESPONS KOMPREHENSIF:

1. **Pembukaan**: "Berikut adalah hadits-hadits yang relevan dengan pertanyaan Anda:"

2. **Hadits** (tampilkan SEMUA hadits):
**Hadits [Nomor]:**
- **Kitab**: [Nama kitab]
- **ID**: [ID hadits]
- **Arab**: [Teks Arab - singkat jika terlalu panjang]
- **Terjemah**: [Terjemahan Indonesia]

3. **Ringkasan**: Penjelasan singkat dan jelas tentang inti ajaran

4. **Penutup**: Pertanyaan follow-up seperti:
   - "Apakah Anda ingin penjelasan lebih lanjut tentang hadits ini?"
   - "Ingin mencari hadits dengan topik lain?"

CONTOH FORMAT:
Assalamu'alaikum. Berikut adalah hadits-hadits yang relevan dengan pertanyaan Anda:

**Hadits 1:**
- **Kitab**: Shahih Bukhari
- **ID**: 123
- **Arab**: حَدَّثَنَا عَبْدُ اللَّهِ...
- **Terjemah**: Telah menceritakan kepada kami Abdullah...

**Ringkasan**: [Penjelasan inti ajaran]

**Pertanyaan lanjut**: Apakah Anda ingin penjelasan lebih detail?
"""
        
        summary_prompt = base_rules + """
FORMAT RESPONS RINGKAS:

1. **Jawaban langsung** untuk pertanyaan
2. **Hadits kunci** (1-2 hadits terpenting)
3. **Inti ajaran** dalam 2-3 kalimat
4. **Follow-up question**

Fokus pada inti ajaran tanpa detail berlebihan.
"""
        
        conversational_prompt = base_rules + """
FORMAT PERCAKAPAN NATURAL:

Berikan respons seperti guru ngaji yang ramah dan berpengetahuan:
- Gunakan gaya bicara yang hangat dan personal
- Jelaskan dengan analogi yang mudah dipahami
- Kaitkan dengan kehidupan sehari-hari
- Dorong untuk pembelajaran lebih lanjut

Tetap tampilkan hadits dengan format yang jelas.
"""
        
        return {
            ResponseMode.COMPREHENSIVE: comprehensive_prompt,
            ResponseMode.SUMMARY: summary_prompt,
            ResponseMode.DIRECT: base_rules + "\nBerikan jawaban langsung dan tepat sasaran.",
            ResponseMode.CONVERSATIONAL: conversational_prompt
        }
    
    def prepare_context_from_results(self, results: List[RetrievalResult]) -> str:
        """
        Prepare context string from retrieval results with intelligent formatting.
        
        Args:
            results (List[RetrievalResult]): Retrieval results
            
        Returns:
            str: Formatted context string
        """
        if not results:
            return "Tidak ada hadits yang ditemukan untuk pertanyaan ini."
        
        context_parts = []
        total_length = 0
        
        for i, result in enumerate(results):
            if i >= self.config.max_hadits_display:
                break
                
            doc = result.document
            
            # Get and truncate Arabic text
            arab_text = doc.get('arab', '')
            if len(arab_text) > self.config.max_arabic_length:
                arab_text = arab_text[:self.config.max_arabic_length] + "... (teks diperpendek)"
            
            # Get and truncate translation
            terjemah_text = doc.get('terjemah', '')
            if len(terjemah_text) > self.config.max_translation_length:
                terjemah_text = terjemah_text[:self.config.max_translation_length] + "... (teks diperpendek)"
            
            # Format context entry
            context_part = f"""
Kitab: {doc.get('kitab', 'Unknown')}
ID: {doc.get('id', 'Unknown')}
Arab: {arab_text}
Terjemah: {terjemah_text}
Relevance Score: {result.score:.3f}
Matched Keywords: {', '.join(result.matched_keywords) if result.matched_keywords else 'None'}
---"""
            
            # Check context length limit
            if total_length + len(context_part) > self.config.max_context_length:
                logger.info(f"Context length limit reached, using {i} hadits")
                break
            
            context_parts.append(context_part)
            total_length += len(context_part)
        
        return "\n".join(context_parts)
    
    async def generate_response(self, query: str, results: List[RetrievalResult], 
                              context_info: Dict[str, Any] = None) -> GenerationResult:
        """
        Generate response for given query and results.
        
        Args:
            query (str): User query
            results (List[RetrievalResult]): Retrieval results
            context_info (Dict): Additional context information
            
        Returns:
            GenerationResult: Generated response
        """
        start_time = time.time()
        context_info = context_info or {}
        
        try:
            # Prepare context
            context = self.prepare_context_from_results(results)
            
            # Select system prompt based on mode
            system_prompt = self.system_prompts[self.config.response_mode]
            
            # Build full prompt
            full_prompt = f"{system_prompt}\n\nKONTEKS HADITS:\n{context}\n\nPERTANYAAN USER: {query}\n\nRESPONS ANDA:"
            
            # Generate response based on provider
            if self.config.llm_provider == LLMProvider.GEMINI and self.gemini_model:
                content = await self._generate_with_gemini(full_prompt)
                provider_used = LLMProvider.GEMINI
                
            elif self.config.llm_provider == LLMProvider.OPENAI and self.openai_client:
                content = await self._generate_with_openai(full_prompt)
                provider_used = LLMProvider.OPENAI
                
            else:
                # Fallback to simple formatting
                content = self._generate_fallback_response(query, results)
                provider_used = LLMProvider.NONE
            
            # Calculate metrics
            generation_time = (time.time() - start_time) * 1000
            token_count = len(content.split())
            
            # Validate response quality
            if not self._validate_response_quality(content):
                logger.warning("Generated response failed quality validation")
            
            return GenerationResult(
                success=True,
                content=content,
                response_mode=self.config.response_mode,
                generation_time_ms=generation_time,
                token_count=token_count,
                provider_used=provider_used,
                metadata={
                    "query_length": len(query.split()),
                    "results_count": len(results),
                    "context_length": len(context),
                    "avg_result_score": sum(r.score for r in results) / len(results) if results else 0
                }
            )
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            generation_time = (time.time() - start_time) * 1000
            
            return GenerationResult(
                success=False,
                error_message=str(e),
                generation_time_ms=generation_time,
                content=self._generate_fallback_response(query, results)
            )
    
    async def _generate_with_gemini(self, prompt: str) -> str:
        """Generate response using Google Gemini."""
        try:
            response = self.gemini_model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=self.config.temperature,
                    max_output_tokens=self.config.max_tokens,
                    top_p=self.config.top_p,
                    top_k=self.config.top_k,
                )
            )
            
            return response.text.strip()
            
        except Exception as e:
            logger.error(f"Gemini generation error: {e}")
            raise
    
    async def _generate_with_openai(self, prompt: str) -> str:
        """Generate response using OpenAI."""
        try:
            response = await self.openai_client.chat.completions.acreate(
                model=self.config.openai_model,
                messages=[
                    {"role": "system", "content": "You are a helpful Islamic hadith assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                top_p=self.config.top_p
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"OpenAI generation error: {e}")
            raise
    
    def _generate_fallback_response(self, query: str, results: List[RetrievalResult]) -> str:
        """Generate fallback response without LLM."""
        if not results:
            return "Maaf, tidak ditemukan hadits yang relevan untuk pertanyaan Anda. Silakan coba dengan kata kunci yang lebih spesifik."
        
        response_parts = []
        response_parts.append("Berikut adalah hadits-hadits yang relevan dengan pertanyaan Anda:")
        response_parts.append("")
        
        for i, result in enumerate(results[:self.config.max_hadits_display], 1):
            doc = result.document
            
            response_parts.append(f"**Hadits {i}:**")
            response_parts.append(f"- **Kitab**: {doc.get('kitab', 'Unknown')}")
            response_parts.append(f"- **ID**: {doc.get('id', 'Unknown')}")
            
            if self.config.include_arabic and doc.get('arab'):
                arab = doc.get('arab', '')
                if len(arab) > self.config.max_arabic_length:
                    arab = arab[:self.config.max_arabic_length] + "..."
                response_parts.append(f"- **Arab**: {arab}")
            
            terjemah = doc.get('terjemah', '')
            if len(terjemah) > self.config.max_translation_length:
                terjemah = terjemah[:self.config.max_translation_length] + "..."
            response_parts.append(f"- **Terjemah**: {terjemah}")
            response_parts.append("")
        
        if len(results) > self.config.max_hadits_display:
            remaining = len(results) - self.config.max_hadits_display
            response_parts.append(f"*Terdapat {remaining} hadits lainnya yang relevan.*")
            response_parts.append("")
        
        response_parts.append("Semoga bermanfaat untuk memahami ajaran Islam.")
        response_parts.append("Apakah Anda ingin penjelasan lebih lanjut atau mencari hadits dengan topik lain?")
        
        return "\n".join(response_parts)
    
    def _validate_response_quality(self, content: str) -> bool:
        """Validate response quality and appropriateness."""
        if not content or len(content.strip()) < self.config.min_response_length:
            return False
        
        if len(content) > self.config.max_response_length:
            logger.warning("Response exceeds maximum length")
        
        # Check for inappropriate content if enabled
        if self.config.filter_inappropriate_content:
            inappropriate_indicators = [
                "sorry", "i don't know", "i can't", "unable to",
                "maaf saya tidak bisa", "saya tidak tahu"
            ]
            
            content_lower = content.lower()
            if any(indicator in content_lower for indicator in inappropriate_indicators):
                logger.warning("Response contains inappropriate content indicators")
        
        return True
    
    async def generate_streaming_response(self, query: str, results: List[RetrievalResult]) -> AsyncGenerator[str, None]:
        """
        Generate streaming response for real-time display.
        
        Args:
            query (str): User query
            results (List[RetrievalResult]): Retrieval results
            
        Yields:
            str: Response chunks
        """
        try:
            # Generate full response first
            generation_result = await self.generate_response(query, results)
            
            if not generation_result.success:
                yield f"Error: {generation_result.error_message}"
                return
            
            # Stream the response word by word
            words = generation_result.content.split()
            current_chunk = ""
            
            for word in words:
                current_chunk += word + " "
                
                # Send chunk when it reaches reasonable size
                if len(current_chunk) > 50 or word.endswith(('.', '!', '?', ':')):
                    yield current_chunk.strip()
                    current_chunk = ""
                    
                    # Add small delay for realistic streaming
                    await asyncio.sleep(self.config.chunk_delay_ms / 1000)
            
            # Send remaining chunk
            if current_chunk.strip():
                yield current_chunk.strip()
                
        except Exception as e:
            logger.error(f"Streaming generation error: {e}")
            yield f"Terjadi kesalahan dalam generate respons: {str(e)}"


# Convenience functions
def create_generator(llm_provider: LLMProvider = LLMProvider.GEMINI, 
                    api_key: str = None, **kwargs) -> EnhancedResponseGenerator:
    """
    Create response generator with simplified configuration.
    
    Args:
        llm_provider (LLMProvider): LLM provider to use
        api_key (str): API key for the provider
        **kwargs: Additional configuration options
        
    Returns:
        EnhancedResponseGenerator: Configured generator
    """
    config = GenerationConfig(llm_provider=llm_provider, **kwargs)
    
    if llm_provider == LLMProvider.GEMINI:
        config.gemini_api_key = api_key
    elif llm_provider == LLMProvider.OPENAI:
        config.openai_api_key = api_key
    
    return EnhancedResponseGenerator(config)


async def quick_generate(query: str, results: List[RetrievalResult], 
                        api_key: str = None) -> str:
    """
    Quick response generation with default settings.
    
    Args:
        query (str): User query
        results (List[RetrievalResult]): Retrieval results
        api_key (str): Gemini API key
        
    Returns:
        str: Generated response
    """
    generator = create_generator(LLMProvider.GEMINI, api_key)
    result = await generator.generate_response(query, results)
    return result.content


# Main execution for testing
if __name__ == "__main__":
    import argparse
    import asyncio
    
    parser = argparse.ArgumentParser(description="Enhanced Response Generator - Fixed V1")
    parser.add_argument("--query", required=True, help="Test query")
    parser.add_argument("--api-key", help="Gemini API key")
    parser.add_argument("--mode", choices=[m.value for m in ResponseMode], 
                       default="comprehensive", help="Response mode")
    parser.add_argument("--provider", choices=[p.value for p in LLMProvider],
                       default="gemini", help="LLM provider")
    parser.add_argument("--streaming", action="store_true", help="Test streaming")
    
    args = parser.parse_args()
    
    async def test_generator():
        # Create mock results for testing
        mock_results = [
            # This would normally come from retrieval system
        ]
        
        # Create generator
        config = GenerationConfig(
            llm_provider=LLMProvider(args.provider),
            response_mode=ResponseMode(args.mode),
            gemini_api_key=args.api_key
        )
        
        generator = EnhancedResponseGenerator(config)
        
        if args.streaming:
            print("Streaming response:")
            async for chunk in generator.generate_streaming_response(args.query, mock_results):
                print(chunk, end=" ", flush=True)
            print()
        else:
            result = await generator.generate_response(args.query, mock_results)
            print(f"Generated response ({result.generation_time_ms:.1f}ms):")
            print(result.content)
    
    asyncio.run(test_generator())
