"""
GENESIS Phase 3A: Universal Knowledge Graph

A scalable knowledge graph that stores entities, relations, and facts.
Designed to eventually scale to billions of entities.

Current implementation: In-memory graph with efficient indexing
Future: Database backend (Neo4j, etc.) for true web-scale

Architecture:
- Entities: Things (people, places, concepts, etc.)
- Relations: Connections between entities (is-a, part-of, etc.)
- Facts: Propositions about entities
- Deduplication: Entity resolution and merging
"""

import numpy as np
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict
import json
from datetime import datetime


class Entity:
    """
    Entity in knowledge graph

    Represents a "thing" - person, place, concept, event, etc.
    """

    def __init__(self,
                 entity_id: str,
                 entity_type: str,
                 name: str,
                 properties: Optional[Dict] = None):
        """
        Args:
            entity_id: Unique identifier
            entity_type: Type (person, place, concept, etc.)
            name: Human-readable name
            properties: Additional properties
        """
        self.id = entity_id
        self.type = entity_type
        self.name = name
        self.properties = properties or {}

        # Aliases for deduplication
        self.aliases = set([name.lower()])

        # Embedding for semantic similarity
        self.embedding = None  # Will be set later

    def add_alias(self, alias: str):
        """Add alternative name for entity"""
        self.aliases.add(alias.lower())

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'type': self.type,
            'name': self.name,
            'properties': self.properties,
            'aliases': list(self.aliases)
        }


class Relation:
    """
    Relation between two entities

    Represents a directed edge in the knowledge graph
    """

    def __init__(self,
                 relation_id: str,
                 head_entity: str,
                 relation_type: str,
                 tail_entity: str,
                 confidence: float = 1.0,
                 metadata: Optional[Dict] = None):
        """
        Args:
            relation_id: Unique identifier
            head_entity: Source entity ID
            relation_type: Type of relation (is-a, part-of, etc.)
            tail_entity: Target entity ID
            confidence: Confidence score (0-1)
            metadata: Additional information
        """
        self.id = relation_id
        self.head = head_entity
        self.type = relation_type
        self.tail = tail_entity
        self.confidence = confidence
        self.metadata = metadata or {}

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'id': self.id,
            'head': self.head,
            'type': self.type,
            'tail': self.tail,
            'confidence': self.confidence,
            'metadata': self.metadata
        }


class UniversalKnowledgeGraph:
    """
    Scalable knowledge graph

    Current capacity: ~1M entities in memory
    Future: Billions via database backend

    Features:
    - Entity deduplication
    - Relation inference
    - Semantic search
    - Graph reasoning
    """

    def __init__(self):
        """Initialize empty knowledge graph"""
        # Storage
        self.entities = {}  # {entity_id: Entity}
        self.relations = {}  # {relation_id: Relation}

        # Indexes for fast lookup
        self.entity_by_name = {}  # {name: entity_id}
        self.entity_by_type = defaultdict(set)  # {type: {entity_ids}}
        self.relations_by_head = defaultdict(list)  # {head_id: [relation_ids]}
        self.relations_by_tail = defaultdict(list)  # {tail_id: [relation_ids]}
        self.relations_by_type = defaultdict(list)  # {type: [relation_ids]}

        # Statistics
        self.entity_count = 0
        self.relation_count = 0
        self.merge_count = 0

    def add_entity(self,
                   name: str,
                   entity_type: str,
                   properties: Optional[Dict] = None,
                   deduplicate: bool = True) -> str:
        """
        Add entity to knowledge graph

        Args:
            name: Entity name
            entity_type: Entity type
            properties: Additional properties
            deduplicate: Check for duplicates before adding

        Returns:
            Entity ID (existing if duplicate found)
        """
        # Check for duplicate if requested
        if deduplicate:
            existing_id = self._find_duplicate_entity(name, entity_type)
            if existing_id:
                # Merge properties
                self._merge_entity_properties(existing_id, properties)
                return existing_id

        # Create new entity
        entity_id = f"e_{self.entity_count}"
        entity = Entity(entity_id, entity_type, name, properties)

        # Store
        self.entities[entity_id] = entity
        self.entity_by_name[name.lower()] = entity_id
        self.entity_by_type[entity_type].add(entity_id)
        self.entity_count += 1

        return entity_id

    def add_relation(self,
                    head_name: str,
                    relation_type: str,
                    tail_name: str,
                    confidence: float = 1.0,
                    create_entities: bool = True) -> Optional[str]:
        """
        Add relation to knowledge graph

        Args:
            head_name: Source entity name
            relation_type: Relation type
            tail_name: Target entity name
            confidence: Confidence score
            create_entities: Create entities if they don't exist

        Returns:
            Relation ID or None if entities not found
        """
        # Find or create entities
        head_id = self.entity_by_name.get(head_name.lower())
        if not head_id:
            if create_entities:
                head_id = self.add_entity(head_name, "unknown")
            else:
                return None

        tail_id = self.entity_by_name.get(tail_name.lower())
        if not tail_id:
            if create_entities:
                tail_id = self.add_entity(tail_name, "unknown")
            else:
                return None

        # Check for duplicate relation
        existing = self._find_duplicate_relation(head_id, relation_type, tail_id)
        if existing:
            # Update confidence (average)
            rel = self.relations[existing]
            rel.confidence = (rel.confidence + confidence) / 2
            return existing

        # Create new relation
        relation_id = f"r_{self.relation_count}"
        relation = Relation(relation_id, head_id, relation_type, tail_id, confidence)

        # Store
        self.relations[relation_id] = relation
        self.relations_by_head[head_id].append(relation_id)
        self.relations_by_tail[tail_id].append(relation_id)
        self.relations_by_type[relation_type].append(relation_id)
        self.relation_count += 1

        return relation_id

    def get_entity(self, name: str) -> Optional[Entity]:
        """Get entity by name"""
        entity_id = self.entity_by_name.get(name.lower())
        if entity_id:
            return self.entities[entity_id]
        return None

    def get_outgoing_relations(self, entity_name: str) -> List[Relation]:
        """Get all relations where entity is the head"""
        entity_id = self.entity_by_name.get(entity_name.lower())
        if not entity_id:
            return []

        relation_ids = self.relations_by_head.get(entity_id, [])
        return [self.relations[rid] for rid in relation_ids]

    def get_incoming_relations(self, entity_name: str) -> List[Relation]:
        """Get all relations where entity is the tail"""
        entity_id = self.entity_by_name.get(entity_name.lower())
        if not entity_id:
            return []

        relation_ids = self.relations_by_tail.get(entity_id, [])
        return [self.relations[rid] for rid in relation_ids]

    def query_path(self, start_name: str, end_name: str, max_hops: int = 3) -> List[List[str]]:
        """
        Find paths between two entities

        Args:
            start_name: Starting entity name
            end_name: Target entity name
            max_hops: Maximum path length

        Returns:
            List of paths (each path is list of entity names)
        """
        start_id = self.entity_by_name.get(start_name.lower())
        end_id = self.entity_by_name.get(end_name.lower())

        if not start_id or not end_id:
            return []

        # BFS to find paths
        paths = []
        queue = [(start_id, [start_id])]
        visited = set()

        while queue and len(paths) < 10:  # Limit to 10 paths
            current_id, path = queue.pop(0)

            if len(path) > max_hops:
                continue

            if current_id == end_id:
                # Convert IDs to names
                entity_names = [self.entities[eid].name for eid in path]
                paths.append(entity_names)
                continue

            if current_id in visited:
                continue
            visited.add(current_id)

            # Explore neighbors
            for rel_id in self.relations_by_head.get(current_id, []):
                rel = self.relations[rel_id]
                next_id = rel.tail
                if next_id not in path:  # Avoid cycles
                    queue.append((next_id, path + [next_id]))

        return paths

    def search_entities(self, query: str, entity_type: Optional[str] = None, limit: int = 10) -> List[Entity]:
        """
        Search entities by name

        Args:
            query: Search query
            entity_type: Filter by type (optional)
            limit: Maximum results

        Returns:
            List of matching entities
        """
        query_lower = query.lower()
        results = []

        for entity in self.entities.values():
            # Type filter
            if entity_type and entity.type != entity_type:
                continue

            # Name match (simple substring search)
            if query_lower in entity.name.lower() or any(query_lower in alias for alias in entity.aliases):
                results.append(entity)

            if len(results) >= limit:
                break

        return results

    def get_statistics(self) -> Dict:
        """Get knowledge graph statistics"""
        return {
            'entity_count': self.entity_count,
            'relation_count': self.relation_count,
            'merge_count': self.merge_count,
            'entity_types': {
                etype: len(entities)
                for etype, entities in self.entity_by_type.items()
            },
            'relation_types': {
                rtype: len(relations)
                for rtype, relations in self.relations_by_type.items()
            },
            'avg_relations_per_entity': self.relation_count / max(self.entity_count, 1)
        }

    def save(self, filepath: str):
        """Save knowledge graph to file"""
        data = {
            'entities': [e.to_dict() for e in self.entities.values()],
            'relations': [r.to_dict() for r in self.relations.values()],
            'statistics': self.get_statistics()
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self, filepath: str):
        """Load knowledge graph from file"""
        with open(filepath, 'r') as f:
            data = json.load(f)

        # Load entities
        for e_data in data['entities']:
            entity = Entity(
                e_data['id'],
                e_data['type'],
                e_data['name'],
                e_data.get('properties', {})
            )
            entity.aliases = set(e_data.get('aliases', [entity.name.lower()]))

            self.entities[entity.id] = entity
            self.entity_by_name[entity.name.lower()] = entity.id
            self.entity_by_type[entity.type].add(entity.id)

        # Load relations
        for r_data in data['relations']:
            relation = Relation(
                r_data['id'],
                r_data['head'],
                r_data['type'],
                r_data['tail'],
                r_data.get('confidence', 1.0),
                r_data.get('metadata', {})
            )

            self.relations[relation.id] = relation
            self.relations_by_head[relation.head].append(relation.id)
            self.relations_by_tail[relation.tail].append(relation.id)
            self.relations_by_type[relation.type].append(relation.id)

        # Update counts
        self.entity_count = len(self.entities)
        self.relation_count = len(self.relations)

    # Helper methods

    def _find_duplicate_entity(self, name: str, entity_type: str) -> Optional[str]:
        """Find existing entity with same name"""
        name_lower = name.lower()

        # Exact name match
        if name_lower in self.entity_by_name:
            return self.entity_by_name[name_lower]

        # Alias match
        for entity in self.entities.values():
            if entity.type == entity_type and name_lower in entity.aliases:
                return entity.id

        return None

    def _merge_entity_properties(self, entity_id: str, new_properties: Optional[Dict]):
        """Merge new properties into existing entity"""
        if not new_properties:
            return

        entity = self.entities[entity_id]
        for key, value in new_properties.items():
            if key not in entity.properties:
                entity.properties[key] = value

        self.merge_count += 1

    def _find_duplicate_relation(self, head_id: str, relation_type: str, tail_id: str) -> Optional[str]:
        """Find existing relation with same head, type, tail"""
        for rel_id in self.relations_by_head.get(head_id, []):
            rel = self.relations[rel_id]
            if rel.type == relation_type and rel.tail == tail_id:
                return rel_id
        return None
