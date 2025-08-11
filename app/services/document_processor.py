import os
import requests
import tempfile
import hashlib
import re
from pathlib import Path
from typing import List, Dict, Any
import logging
from urllib.parse import urlparse

import PyPDF2
from docx import Document
import email
from email import policy

from app.models.schemas import DocumentChunk
from app.core.config import settings

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Service for processing various document formats"""
    
    def __init__(self):
        self.supported_formats = settings.SUPPORTED_FORMATS.split(',')
        self.chunk_size = settings.CHUNK_SIZE
        self.chunk_overlap = settings.CHUNK_OVERLAP
        self.max_chunk_size = 4000  # Much smaller limit for better semantic processing (4KB)
        
    def _validate_chunk_size(self, content: str) -> str:
        """Validate and truncate chunk if too large"""
        byte_size = len(content.encode('utf-8'))
        if byte_size > self.max_chunk_size:
            logger.warning(f"Chunk size {byte_size} bytes exceeds limit of {self.max_chunk_size}, truncating...")
            
            # Calculate safe character count (rough estimate)
            char_limit = int(self.max_chunk_size * 0.8)  # Conservative estimate
            
            if len(content) > char_limit:
                # Try to truncate at sentence boundary
                truncated = content[:char_limit]
                last_period = truncated.rfind('.')
                if last_period > char_limit * 0.7:  # If we can find a good sentence ending
                    content = truncated[:last_period + 1]
                else:
                    content = truncated
                
            logger.info(f"Chunk truncated to {len(content)} characters ({len(content.encode('utf-8'))} bytes)")
            
        return content
        
    async def download_document(self, url: str) -> str:
        """Download document from URL and return local file path"""
        try:
            logger.info(f"Downloading document from: {url}")
            
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=self._get_file_extension(url)) as tmp_file:
                for chunk in response.iter_content(chunk_size=8192):
                    tmp_file.write(chunk)
                temp_path = tmp_file.name
                
            logger.info(f"Document downloaded to: {temp_path}")
            return temp_path
            
        except Exception as e:
            logger.error(f"Error downloading document: {str(e)}")
            raise Exception(f"Failed to download document: {str(e)}")
    
    def _get_file_extension(self, url: str) -> str:
        """Extract file extension from URL"""
        parsed_url = urlparse(url)
        path = parsed_url.path.lower()
        
        if '.pdf' in path:
            return '.pdf'
        elif '.docx' in path:
            return '.docx'
        elif '.doc' in path:
            return '.doc'
        else:
            return '.pdf'  # Default to PDF
    
    async def process_document(self, file_path: str) -> List[DocumentChunk]:
        """Process document and return chunks"""
        try:
            logger.info(f"Processing document: {file_path}")
            
            # Determine file type and extract text
            if file_path.lower().endswith('.pdf'):
                text = self._extract_pdf_text(file_path)
            elif file_path.lower().endswith(('.docx', '.doc')):
                text = self._extract_docx_text(file_path)
            elif file_path.lower().endswith('.eml'):
                text = self._extract_email_text(file_path)
            else:
                text = self._extract_plain_text(file_path)
            
            # Create chunks
            chunks = self._create_chunks(text, file_path)
            
            # Clean up temporary file
            if os.path.exists(file_path):
                os.unlink(file_path)
                
            logger.info(f"Document processed successfully. Created {len(chunks)} chunks.")
            return chunks
            
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            raise Exception(f"Failed to process document: {str(e)}")
    
    def _extract_pdf_text(self, file_path: str) -> str:
        """Extract text from PDF file"""
        text = ""
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            logger.error(f"Error extracting PDF text: {str(e)}")
            raise Exception(f"Failed to extract PDF text: {str(e)}")
        
        return text.strip()
    
    def _extract_docx_text(self, file_path: str) -> str:
        """Extract text from DOCX file"""
        try:
            doc = Document(file_path)
            text = ""
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            return text.strip()
        except Exception as e:
            logger.error(f"Error extracting DOCX text: {str(e)}")
            raise Exception(f"Failed to extract DOCX text: {str(e)}")
    
    def _extract_email_text(self, file_path: str) -> str:
        """Extract text from email file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                msg = email.message_from_file(file, policy=policy.default)
                
            text = f"Subject: {msg.get('Subject', 'No Subject')}\n"
            text += f"From: {msg.get('From', 'Unknown')}\n"
            text += f"To: {msg.get('To', 'Unknown')}\n\n"
            
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        text += part.get_content()
            else:
                text += msg.get_content()
                
            return text.strip()
        except Exception as e:
            logger.error(f"Error extracting email text: {str(e)}")
            raise Exception(f"Failed to extract email text: {str(e)}")
    
    def _extract_plain_text(self, file_path: str) -> str:
        """Extract text from plain text file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read().strip()
        except Exception as e:
            logger.error(f"Error extracting plain text: {str(e)}")
            raise Exception(f"Failed to extract plain text: {str(e)}")
    
    def _create_chunks(self, text: str, source_path: str) -> List[DocumentChunk]:
        """Create semantic chunks with simple but effective approach"""
        chunks = []
        
        # Preprocess text - normalize whitespace and fix common issues
        text = self._preprocess_text(text)
        
        logger.info(f"Starting chunking process for text of length: {len(text)} characters")
        
        # Use simple paragraph-based chunking with overlap
        chunks = self._create_simple_overlapping_chunks(text, source_path)
        
        logger.info(f"Created {len(chunks)} chunks using simple overlapping method")
        
        return chunks
    
    def _create_simple_overlapping_chunks(self, text: str, source_path: str) -> List[DocumentChunk]:
        """Create overlapping chunks with enhanced numerical content preservation"""
        chunks = []
        chunk_size = self.chunk_size
        overlap_size = self.chunk_overlap
        
        # Split text into sentences for better chunk boundaries
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # PHASE 3: Identify numerical content for special handling
        numerical_sentences = []
        for i, sentence in enumerate(sentences):
            if self._contains_key_numerical_info(sentence):
                # Add context around numerical sentences
                start_idx = max(0, i-1)
                end_idx = min(len(sentences), i+2)
                context_sentences = sentences[start_idx:end_idx]
                numerical_sentences.extend(context_sentences)
        
        current_chunk = ""
        sentence_buffer = []
        chunk_index = 0
        
        for sentence in sentences:
            sentence_buffer.append(sentence)
            test_chunk = " ".join(sentence_buffer)
            
            # Preserve numerical content by creating special chunks
            if self._contains_key_numerical_info(sentence):
                # Ensure this sentence gets preserved with enough context
                if len(sentence_buffer) < 3:
                    continue  # Keep building context
            
            # If adding this sentence would exceed chunk size, create a chunk
            if len(test_chunk) > chunk_size and current_chunk:
                # Create chunk from current content
                chunk_content = current_chunk.strip()
                if chunk_content and len(chunk_content) > 50:
                    
                    chunk_id = self._generate_chunk_id(source_path, chunk_index)
                    
                    # Enhanced metadata for better search
                    metadata = {
                        "source": source_path,
                        "chunk_index": chunk_index,
                        "character_count": len(chunk_content),
                        "word_count": len(chunk_content.split()),
                        "section_title": self._extract_section_title(chunk_content),
                        "chunk_type": "numerical_enhanced" if self._contains_key_numerical_info(chunk_content) else "paragraph_based",
                        "has_numerical_data": self._contains_key_numerical_info(chunk_content),
                        "has_financial_terms": self._contains_financial_terms(chunk_content),
                        "has_eligibility_info": self._contains_eligibility_terms(chunk_content)
                    }
                    
                    chunk = DocumentChunk(
                        content=self._validate_chunk_size(chunk_content),
                        chunk_id=chunk_id,
                        metadata=metadata
                    )
                    chunks.append(chunk)
                    chunk_index += 1
                
                # Start new chunk with enhanced overlap for numerical content
                if self._contains_key_numerical_info(current_chunk):
                    overlap_sentences = sentence_buffer[-4:]  # More overlap for numerical content
                else:
                    overlap_sentences = sentence_buffer[-2:]  # Standard overlap
                    
                current_chunk = " ".join(overlap_sentences)
                sentence_buffer = overlap_sentences + [sentence]
            else:
                current_chunk = test_chunk
        
        # Create final chunk if there's remaining content
        if current_chunk.strip() and len(current_chunk.strip()) > 50:
            chunk_id = self._generate_chunk_id(source_path, chunk_index)
            
            metadata = {
                "source": source_path,
                "chunk_index": chunk_index,
                "character_count": len(current_chunk.strip()),
                "word_count": len(current_chunk.split()),
                "section_title": self._extract_section_title(current_chunk),
                "chunk_type": "numerical_enhanced" if self._contains_key_numerical_info(current_chunk) else "paragraph_based",
                "has_numerical_data": self._contains_key_numerical_info(current_chunk),
                "has_financial_terms": self._contains_financial_terms(current_chunk),
                "has_eligibility_info": self._contains_eligibility_terms(current_chunk)
            }
            
            chunk = DocumentChunk(
                content=self._validate_chunk_size(current_chunk.strip()),
                chunk_id=chunk_id,
                metadata=metadata
            )
            chunks.append(chunk)
        
        logger.info(f"Enhanced chunking created {len(chunks)} chunks with average size: {sum(len(c.content) for c in chunks) // len(chunks) if chunks else 0} characters")
        
        # Log special chunk types
        numerical_chunks = sum(1 for c in chunks if c.metadata.get("has_numerical_data"))
        financial_chunks = sum(1 for c in chunks if c.metadata.get("has_financial_terms"))
        logger.info(f"Special chunks: {numerical_chunks} numerical, {financial_chunks} financial")
        
        return chunks
    
    def _contains_key_numerical_info(self, text: str) -> bool:
        """Check if text contains important numerical information"""
        import re
        
        # Patterns for important numerical data
        patterns = [
            r'₹[\d,]+(?:\.\d+)?(?:\s*(?:lakh|crore|thousand))?',  # Indian currency amounts
            r'\d+(?:\.\d+)?%',  # Percentages
            r'\d+\s*(?:years?|months?|days?)',  # Time periods
            r'(?:minimum|maximum|up to|from|between)\s+₹?[\d,]+',  # Limits and ranges
            r'\d+(?:\.\d+)?\s*(?:lakh|crore)',  # Large amounts
            r'age\s+(?:limit|range|between|from|to)?\s*\d+',  # Age ranges
            r'sum\s+insured.*₹[\d,]+',  # Sum insured amounts
        ]
        
        return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)
    
    def _contains_financial_terms(self, text: str) -> bool:
        """Check if text contains financial/insurance terms"""
        financial_terms = [
            'co-payment', 'co-pay', 'deductible', 'premium', 'sum insured', 
            'coverage', 'reimbursement', 'claim', 'policy limit'
        ]
        text_lower = text.lower()
        return any(term in text_lower for term in financial_terms)
    
    def _contains_eligibility_terms(self, text: str) -> bool:
        """Check if text contains eligibility-related terms"""
        eligibility_terms = [
            'eligibility', 'eligible', 'age limit', 'entry age', 'renewal',
            'waiting period', 'pre-existing', 'exclusion'
        ]
        text_lower = text.lower()
        return any(term in text_lower for term in eligibility_terms)
    
    def _extract_section_title(self, content: str) -> str:
        """Extract a simple section title from chunk content"""
        lines = content.strip().split('\n')
        first_line = lines[0].strip()
        
        # If first line is short and looks like a title, use it
        if len(first_line) < 100 and any(word in first_line.upper() for word in ['POLICY', 'SECTION', 'CLAUSE', 'BENEFITS', 'COVERAGE', 'EXCLUSION']):
            return first_line
        
        # Otherwise, create title from first few words
        words = content.split()[:10]
        return " ".join(words) + "..." if len(words) == 10 else " ".join(words)
    
    def _extract_document_structure(self, text: str) -> List[Dict[str, Any]]:
        """Extract document structure with clause-level metadata"""
        import re
        
        sections = []
        
        # Patterns for different document structures
        patterns = [
            # Insurance policy patterns
            (r'(?i)^(\d+\.?\s+)([A-Z][^.\n]{10,})', 'numbered_clause'),
            (r'(?i)^([A-Z][A-Z\s]{5,}:)', 'titled_section'),
            (r'(?i)(exclusions?|exceptions?|limitations?|conditions?|benefits?|coverage)', 'special_clause'),
            (r'(?i)(co-?payment|co-?pay|deductible|waiting\s+period)', 'financial_clause'),
            (r'(?i)(pre-?existing|covid|daycare|ambulance|hospitalisation)', 'coverage_clause'),
            (r'(?i)(eligibility|age\s+limit|sum\s+insured|premium)', 'eligibility_clause'),
        ]
        
        # Split text into paragraphs
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        
        current_section = None
        current_content = []
        
        for para in paragraphs:
            para_type = 'general'
            clause_info = {}
            
            # Detect clause type and extract metadata
            for pattern, clause_type in patterns:
                match = re.search(pattern, para)
                if match:
                    para_type = clause_type
                    clause_info = {
                        'clause_type': clause_type,
                        'matched_text': match.group(),
                        'is_special': clause_type in ['special_clause', 'financial_clause', 'coverage_clause']
                    }
                    break
            
            # Start new section if we hit a major boundary
            if para_type in ['numbered_clause', 'titled_section'] and current_content:
                # Save current section
                if current_section:
                    sections.append(current_section)
                
                # Start new section
                current_section = {
                    'title': self._extract_section_title(para),
                    'content': para,
                    'clause_type': para_type,
                    'metadata': clause_info,
                    'paragraphs': [para]
                }
                current_content = [para]
            else:
                # Add to current section
                if current_section is None:
                    current_section = {
                        'title': 'Introduction',
                        'content': para,
                        'clause_type': 'general',
                        'metadata': clause_info,
                        'paragraphs': [para]
                    }
                    current_content = [para]
                else:
                    current_section['content'] += '\n\n' + para
                    current_section['paragraphs'].append(para)
                    current_content.append(para)
                    
                    # Update metadata if this paragraph has special info
                    if clause_info.get('is_special'):
                        current_section['metadata'].update(clause_info)
        
        # Add final section
        if current_section:
            sections.append(current_section)
        
        return sections
    
    def _extract_section_title(self, text: str) -> str:
        """Extract a meaningful title from section text"""
        import re
        
        # Try to find numbered sections
        match = re.match(r'^(\d+\.?\s+)([^.\n]{5,50})', text)
        if match:
            return match.group(2).strip()
        
        # Try to find titled sections
        match = re.match(r'^([A-Z][^:\n]{5,50})', text)
        if match:
            return match.group(1).strip()
        
        # Fallback to first few words
        words = text.split()[:8]
        return ' '.join(words) + ('...' if len(words) == 8 else '')
    
    def _create_semantic_chunks_from_section(self, section: Dict[str, Any], source_path: str, start_index: int) -> List[DocumentChunk]:
        """Create chunks from a structured section with semantic boundaries"""
        chunks = []
        content = section['content']
        
        # If section is small enough, keep as single chunk
        if len(content) <= self.chunk_size:
            # Validate chunk size before creating
            content = self._validate_chunk_size(content)
            
            chunk_id = self._generate_chunk_id(source_path, start_index)
            chunk = DocumentChunk(
                content=content,
                chunk_id=chunk_id,
                metadata={
                    "source": source_path,
                    "chunk_index": start_index,
                    "character_count": len(content),
                    "word_count": len(content.split()),
                    "section_title": section['title'],
                    "clause_type": section['clause_type'],
                    "is_special_clause": section['metadata'].get('is_special', False),
                    "semantic_type": "complete_section",
                    **section['metadata']
                }
            )
            chunks.append(chunk)
            return chunks
        
        # For larger sections, split semantically while preserving context
        paragraphs = section['paragraphs']
        current_chunk = ""
        chunk_paras = []
        
        for para in paragraphs:
            # Check if adding this paragraph would exceed chunk size
            test_chunk = current_chunk + ('\n\n' if current_chunk else '') + para
            
            if len(test_chunk) <= self.chunk_size:
                current_chunk = test_chunk
                chunk_paras.append(para)
            else:
                # Save current chunk if it has content
                if current_chunk:
                    # Validate chunk size before creating
                    current_chunk = self._validate_chunk_size(current_chunk)
                    
                    chunk_id = self._generate_chunk_id(source_path, start_index + len(chunks))
                    chunk = DocumentChunk(
                        content=current_chunk,
                        chunk_id=chunk_id,
                        metadata={
                            "source": source_path,
                            "chunk_index": start_index + len(chunks),
                            "character_count": len(current_chunk),
                            "word_count": len(current_chunk.split()),
                            "section_title": section['title'],
                            "clause_type": section['clause_type'],
                            "is_special_clause": section['metadata'].get('is_special', False),
                            "semantic_type": "section_part",
                            "paragraph_count": len(chunk_paras),
                            **section['metadata']
                        }
                    )
                    chunks.append(chunk)
                
                # Start new chunk with overlap from previous
                if chunks and self.chunk_overlap > 0:
                    overlap_text = self._get_semantic_overlap(current_chunk, self.chunk_overlap)
                    current_chunk = overlap_text + '\n\n' + para if overlap_text else para
                else:
                    current_chunk = para
                chunk_paras = [para]
        
        # Add final chunk
        if current_chunk:
            # Validate chunk size before creating
            current_chunk = self._validate_chunk_size(current_chunk)
            
            chunk_id = self._generate_chunk_id(source_path, start_index + len(chunks))
            chunk = DocumentChunk(
                content=current_chunk,
                chunk_id=chunk_id,
                metadata={
                    "source": source_path,
                    "chunk_index": start_index + len(chunks),
                    "character_count": len(current_chunk),
                    "word_count": len(current_chunk.split()),
                    "section_title": section['title'],
                    "clause_type": section['clause_type'],
                    "is_special_clause": section['metadata'].get('is_special', False),
                    "semantic_type": "section_final",
                    "paragraph_count": len(chunk_paras),
                    **section['metadata']
                }
            )
            chunks.append(chunk)
        
        return chunks
    
    def _get_semantic_overlap(self, text: str, overlap_size: int) -> str:
        """Get semantically meaningful overlap from previous chunk"""
        sentences = text.split('. ')
        if len(sentences) <= 1:
            return ""
        
        # Take the last complete sentence that fits in overlap size
        overlap_text = ""
        for sentence in reversed(sentences):
            test_overlap = sentence + ('. ' + overlap_text if overlap_text else '')
            if len(test_overlap) <= overlap_size:
                overlap_text = test_overlap
            else:
                break
        
        return overlap_text
    
    def _preprocess_text(self, text: str) -> str:
        """Clean and normalize text before chunking"""
        import re
        
        logger.info(f"Preprocessing text of length: {len(text)}")
        
        # Remove excessive whitespace but preserve paragraph structure
        text = re.sub(r'[ \t]+', ' ', text)              # Normalize spaces and tabs
        text = re.sub(r'\n\s*\n', '\n\n', text)          # Normalize paragraph breaks
        
        # Fix common PDF extraction issues
        text = re.sub(r'([a-z])-\s*\n\s*([a-z])', r'\1\2', text)  # Fix hyphenated words across lines
        text = re.sub(r'(\.)([A-Z])', r'\1 \2', text)     # Space after period before capital
        text = re.sub(r'(\d+)\.(\d+)', r'\1.\2', text)    # Preserve decimal numbers
        
        # Normalize sentence endings
        text = re.sub(r'\.{2,}', '.', text)               # Multiple periods to single
        text = re.sub(r'\s*\.\s*', '. ', text)            # Standardize period spacing
        
        # Ensure proper sentence structure
        text = re.sub(r'([a-z])\s*\n\s*([A-Z])', r'\1. \2', text)  # Add periods where missing
        
        logger.info(f"Text preprocessed, final length: {len(text)}")
        return text.strip()
    
    def _recursive_split(self, text: str, separators: List[str], chunk_size: int, chunk_overlap: int) -> List[str]:
        """Recursively split text using hierarchical separators"""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        
        # Try each separator in order of priority
        for separator in separators:
            if separator in text:
                # Split by current separator
                parts = text.split(separator)
                
                current_chunk = ""
                
                for i, part in enumerate(parts):
                    part = part.strip()
                    if not part:
                        continue
                    
                    # Test if adding this part would exceed chunk size
                    test_chunk = current_chunk + (separator if current_chunk else "") + part
                    
                    if len(test_chunk) <= chunk_size:
                        # Add to current chunk
                        if current_chunk:
                            current_chunk += separator + part
                        else:
                            current_chunk = part
                    else:
                        # Current chunk is ready, save it
                        if current_chunk:
                            chunks.append(current_chunk)
                        
                        # If single part is too large, recursively split it
                        if len(part) > chunk_size:
                            sub_chunks = self._recursive_split(part, separators[separators.index(separator)+1:], chunk_size, chunk_overlap)
                            chunks.extend(sub_chunks)
                        else:
                            # Start new chunk with overlap from previous
                            if chunks and chunk_overlap > 0:
                                overlap_text = self._get_overlap_text(chunks[-1], chunk_overlap)
                                current_chunk = overlap_text + separator + part if overlap_text else part
                            else:
                                current_chunk = part
                
                # Add final chunk
                if current_chunk:
                    chunks.append(current_chunk)
                
                return chunks
        
        # If no separators work, force split by characters
        return self._force_split(text, chunk_size, chunk_overlap)
    
    def _get_overlap_text(self, text: str, overlap_size: int) -> str:
        """Get overlap text from end of previous chunk"""
        words = text.split()
        if len(words) <= 5:  # If chunk is very short, use all of it
            return text
        
        # Take last N words where N gives us roughly overlap_size characters
        overlap_words = []
        char_count = 0
        
        for word in reversed(words):
            if char_count + len(word) + 1 <= overlap_size:  # +1 for space
                overlap_words.insert(0, word)
                char_count += len(word) + 1
            else:
                break
        
        return " ".join(overlap_words)
    
    def _force_split(self, text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """Force split text by characters when no good separators are found"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            if end >= len(text):
                # Last chunk
                chunks.append(text[start:])
                break
            
            # Try to end at a word boundary
            while end > start and text[end] != ' ':
                end -= 1
            
            if end == start:  # No word boundary found, force split
                end = start + chunk_size
            
            chunks.append(text[start:end])
            
            # Calculate next start with overlap
            overlap_chars = min(chunk_overlap, len(chunks[-1]) // 2)
            start = end - overlap_chars
        
        return chunks
    
    def _generate_chunk_id(self, source_path: str, chunk_index: int) -> str:
        """Generate unique chunk ID"""
        source_hash = hashlib.md5(source_path.encode()).hexdigest()[:8]
        return f"chunk_{source_hash}_{chunk_index:04d}"
