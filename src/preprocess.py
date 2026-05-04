# src/preprocess.py
import re
from typing import List, Iterator, Dict
from src.config import settings
import hashlib
import json

class TextCleaner:
    """Handles regex-based cleaning. No state."""
    
    @staticmethod
    def clean(text: str) -> str:
        """
        Implement cleaning pipeline.
        Steps:
        1. Replace multiple newlines with single space.
        2. Remove excessive whitespace (spaces/tabs).
        3. Remove HTML tags if present.
        4. Return stripped string.
        """
        # Remove multiple newlines with single space
        text = re.sub(r'\n+', ' ', text)
        
        # Remove [[File:...]], {{...}}, [1]
        text = re.sub(r'\[File:[^\]]+\]|\{\{[^\}]*\}\}|\[\d+\]', '', text)
        
        # Remove multiple spaces/tabs
        text = re.sub(r'\s+', ' ', text)
        
        #Remove HTML tags if present
        text = re.sub(r'<.*?>', '', text)
        
        return text.strip()

class Chunker:
    """Splits text into overlapping chunks based on config."""
    
    def __init__(self, chunk_size: int = None, overlap: int = None):
        # Use settings.CHUNK_SIZE if None, else chunk_size
        self.chunk_size = settings.CHUNK_SIZE if chunk_size is None else chunk_size
        self.overlap = settings.CHUNK_OVERLAP if overlap is None else overlap

    def split_text_generator(self, text: str):
        """
        Splits the text into chunks of words.
        
        Args:
            text (str): The input text to split.
        
        Yields:
            str: A chunk of text as a string.
        """
        # Clean and split text into words
        cleaned_text = TextCleaner.clean(text)
        words = cleaned_text.split()
        
        for window in self._sliding_window(words):
            chunk = ' '.join(window)
            yield chunk
   

    def _sliding_window(self, tokens: List[str]) -> Iterator[List[str]]:
        """
        Core logic generator.
        Yields windows of tokens based on self.chunk_size and self.overlap.
        """
        if not tokens or self.chunk_size <= 0 or self.overlap < 0:
            raise ValueError("Invalid input: chunk_size and overlap must be positive integers")

        step = self.chunk_size - self.overlap
        if step <= 0:
            raise ValueError("overlap must be smaller than chunk_size")
        for i in range(0, len(tokens), step):
            yield tokens[i:i+self.chunk_size]


class DocumentProcessor:
    """Handles loading JSONL and applying chunking with title context."""
    
    def __init__(self, chunker: Chunker):
        self.chunker = chunker

    def process_file(self, file_path: str) -> Iterator[Dict]:
        """
        Streams JSONL file line by line.
        For each document:
         1. Parse JSON (fields: 'title', 'text').
         2. For each chunk from chunker.split_text(text):
            - Create chunk_text = f"{title} : {chunk}"
            - Generate unique ID (e.g., f"{title}_{chunk_idx}")
            - Yield dict: {"id": id, "text": chunk_text, "title": title}
        """
        with open(file_path, 'r', encoding="utf-8") as file:
            for line in file:
                data = json.loads(line)
                title = data['title']
                text = data['text']
                if text is None:
                    continue
                
                chunks = self.chunker.split_text_generator(text)
                for i, chunk in enumerate(chunks):
                    chunk_text = f"{title} : {chunk}"
                    unique_id = f"{title}_chunk_{i}"
                    yield {"id": unique_id, "text": chunk_text, "title": title}
