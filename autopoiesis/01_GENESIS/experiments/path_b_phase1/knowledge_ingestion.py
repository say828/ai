"""
GENESIS Phase 3A: Knowledge Ingestion System

Collects knowledge from multiple sources and prepares it for
integration with the artificial life learning system.

Sources:
- Web scraping (articles, documents)
- Local files (text, markdown, code)
- APIs (structured data)
- User input (direct knowledge injection)

Output: Normalized knowledge units ready for graph integration
"""

import numpy as np
from typing import Dict, List, Optional, Union
from pathlib import Path
import json
import hashlib
from collections import defaultdict
from datetime import datetime


class KnowledgeUnit:
    """
    Atomic unit of knowledge

    Represents a single piece of information that can be:
    - Stored in knowledge graph
    - Learned by agents
    - Retrieved by queries
    """

    def __init__(self,
                 content: str,
                 source: str,
                 knowledge_type: str = "fact",
                 metadata: Optional[Dict] = None):
        """
        Args:
            content: The actual knowledge content
            source: Where this knowledge came from
            knowledge_type: Type of knowledge (fact, concept, rule, etc.)
            metadata: Additional information (timestamp, confidence, etc.)
        """
        self.content = content
        self.source = source
        self.knowledge_type = knowledge_type
        self.metadata = metadata or {}

        # Generate unique ID based on content
        self.id = self._generate_id()

        # Add timestamp
        if 'timestamp' not in self.metadata:
            self.metadata['timestamp'] = datetime.now().isoformat()

    def _generate_id(self) -> str:
        """Generate unique ID from content hash"""
        content_hash = hashlib.sha256(self.content.encode()).hexdigest()
        return f"ku_{content_hash[:16]}"

    def to_dict(self) -> Dict:
        """Convert to dictionary for storage"""
        return {
            'id': self.id,
            'content': self.content,
            'source': self.source,
            'knowledge_type': self.knowledge_type,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'KnowledgeUnit':
        """Load from dictionary"""
        return cls(
            content=data['content'],
            source=data['source'],
            knowledge_type=data.get('knowledge_type', 'fact'),
            metadata=data.get('metadata', {})
        )


class KnowledgeIngestionPipeline:
    """
    Pipeline for ingesting knowledge from multiple sources

    Processing stages:
    1. Collection: Gather raw data
    2. Normalization: Convert to standard format
    3. Quality filtering: Remove low-quality content
    4. Deduplication: Identify and merge duplicates
    5. Enrichment: Add metadata and context
    """

    def __init__(self, quality_threshold: float = 0.5):
        """
        Args:
            quality_threshold: Minimum quality score to accept (0-1)
        """
        self.quality_threshold = quality_threshold

        # Storage
        self.knowledge_units = {}  # {id: KnowledgeUnit}
        self.content_hashes = set()  # For fast deduplication

        # Statistics
        self.total_ingested = 0
        self.total_duplicates = 0
        self.total_filtered = 0

        # Source tracking
        self.sources = defaultdict(int)  # {source: count}

    def ingest_text(self, text: str, source: str = "user_input") -> List[str]:
        """
        Ingest raw text and extract knowledge units

        Simple version: Split into sentences and treat each as a knowledge unit

        Args:
            text: Raw text content
            source: Source identifier

        Returns:
            List of knowledge unit IDs
        """
        # Split into sentences (simple version)
        sentences = self._split_sentences(text)

        added_ids = []
        for sentence in sentences:
            # Skip very short or empty
            if len(sentence.strip()) < 10:
                continue

            # Create knowledge unit
            ku = KnowledgeUnit(
                content=sentence.strip(),
                source=source,
                knowledge_type="fact"
            )

            # Quality check
            quality = self._assess_quality(ku)
            if quality < self.quality_threshold:
                self.total_filtered += 1
                continue

            # Deduplication check
            if ku.id in self.knowledge_units:
                self.total_duplicates += 1
                continue

            # Add to storage
            self.knowledge_units[ku.id] = ku
            self.content_hashes.add(ku.id)
            self.sources[source] += 1
            self.total_ingested += 1
            added_ids.append(ku.id)

        return added_ids

    def ingest_file(self, filepath: Union[str, Path]) -> List[str]:
        """
        Ingest knowledge from a file

        Supported formats:
        - .txt: Plain text
        - .md: Markdown
        - .json: Structured knowledge

        Args:
            filepath: Path to file

        Returns:
            List of knowledge unit IDs
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        # Read file
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Process based on extension
        if filepath.suffix in ['.txt', '.md']:
            return self.ingest_text(content, source=str(filepath))

        elif filepath.suffix == '.json':
            return self._ingest_json(content, source=str(filepath))

        else:
            # Try as text
            return self.ingest_text(content, source=str(filepath))

    def ingest_structured(self, data: Dict, source: str = "structured") -> str:
        """
        Ingest structured knowledge (e.g., from API)

        Args:
            data: Dictionary with knowledge data
            source: Source identifier

        Returns:
            Knowledge unit ID
        """
        # Convert structured data to text
        content = self._dict_to_text(data)

        ku = KnowledgeUnit(
            content=content,
            source=source,
            knowledge_type=data.get('type', 'fact'),
            metadata=data.get('metadata', {})
        )

        # Check deduplication
        if ku.id in self.knowledge_units:
            self.total_duplicates += 1
            return ku.id

        # Store
        self.knowledge_units[ku.id] = ku
        self.sources[source] += 1
        self.total_ingested += 1

        return ku.id

    def get_knowledge_units(self,
                          source: Optional[str] = None,
                          knowledge_type: Optional[str] = None,
                          limit: Optional[int] = None) -> List[KnowledgeUnit]:
        """
        Retrieve knowledge units with optional filtering

        Args:
            source: Filter by source
            knowledge_type: Filter by type
            limit: Maximum number to return

        Returns:
            List of knowledge units
        """
        units = list(self.knowledge_units.values())

        # Filter by source
        if source:
            units = [u for u in units if u.source == source]

        # Filter by type
        if knowledge_type:
            units = [u for u in units if u.knowledge_type == knowledge_type]

        # Limit
        if limit:
            units = units[:limit]

        return units

    def get_statistics(self) -> Dict:
        """Get ingestion statistics"""
        return {
            'total_ingested': self.total_ingested,
            'total_duplicates': self.total_duplicates,
            'total_filtered': self.total_filtered,
            'unique_knowledge_units': len(self.knowledge_units),
            'sources': dict(self.sources),
            'knowledge_types': self._count_types()
        }

    def save(self, filepath: str):
        """Save ingested knowledge to file"""
        data = {
            'knowledge_units': [ku.to_dict() for ku in self.knowledge_units.values()],
            'statistics': self.get_statistics()
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self, filepath: str):
        """Load ingested knowledge from file"""
        with open(filepath, 'r') as f:
            data = json.load(f)

        for ku_data in data['knowledge_units']:
            ku = KnowledgeUnit.from_dict(ku_data)
            self.knowledge_units[ku.id] = ku
            self.sources[ku.source] += 1

    # Helper methods

    def _split_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences

        Simple version using punctuation
        Better version would use NLP library
        """
        import re

        # Split on sentence boundaries
        sentences = re.split(r'[.!?]+', text)

        # Clean up
        sentences = [s.strip() for s in sentences if s.strip()]

        return sentences

    def _assess_quality(self, ku: KnowledgeUnit) -> float:
        """
        Assess quality of knowledge unit

        Simple heuristics:
        - Length (too short = low quality)
        - Diversity (repeated words = low quality)
        - Structure (no punctuation = low quality)

        Returns:
            Quality score 0-1
        """
        content = ku.content

        # Length check
        if len(content) < 10:
            return 0.0
        if len(content) > 500:
            return 0.5  # Very long, might be low signal

        # Word diversity
        words = content.lower().split()
        if len(words) == 0:
            return 0.0

        unique_ratio = len(set(words)) / len(words)

        # Punctuation check
        has_punctuation = any(c in content for c in '.,!?;:')

        # Combine scores
        quality = (
            0.3 * unique_ratio +
            0.3 * (1.0 if has_punctuation else 0.0) +
            0.4 * min(len(content) / 100, 1.0)
        )

        return quality

    def _ingest_json(self, json_content: str, source: str) -> List[str]:
        """Ingest JSON-formatted knowledge"""
        data = json.loads(json_content)

        added_ids = []

        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    ku_id = self.ingest_structured(item, source=source)
                    added_ids.append(ku_id)
        elif isinstance(data, dict):
            ku_id = self.ingest_structured(data, source=source)
            added_ids.append(ku_id)

        return added_ids

    def _dict_to_text(self, data: Dict) -> str:
        """Convert dictionary to readable text"""
        parts = []

        for key, value in data.items():
            if key in ['metadata', 'type']:
                continue
            parts.append(f"{key}: {value}")

        return "; ".join(parts)

    def _count_types(self) -> Dict[str, int]:
        """Count knowledge units by type"""
        counts = defaultdict(int)
        for ku in self.knowledge_units.values():
            counts[ku.knowledge_type] += 1
        return dict(counts)
