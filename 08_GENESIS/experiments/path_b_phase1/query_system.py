"""
GENESIS Phase 3A: Query System

Real-time query interface for the knowledge system.
Allows users to ask questions while the system is actively learning.

Query types:
- Entity lookup: "What is X?"
- Relation query: "How is X related to Y?"
- Multi-hop reasoning: "What connects X and Y?"
- Semantic search: "Find similar to X"

Future: Natural language understanding, complex reasoning
"""

from typing import Dict, List, Optional, Tuple
from universal_knowledge_graph import UniversalKnowledgeGraph, Entity, Relation
from knowledge_ingestion import KnowledgeIngestionPipeline, KnowledgeUnit


class QueryResult:
    """
    Result of a query

    Contains answer, confidence, and supporting evidence
    """

    def __init__(self,
                 query: str,
                 answer: str,
                 confidence: float,
                 evidence: Optional[List[str]] = None,
                 reasoning_path: Optional[List[str]] = None):
        """
        Args:
            query: Original query
            answer: Answer text
            confidence: Confidence score (0-1)
            evidence: Supporting evidence (knowledge units, relations, etc.)
            reasoning_path: Steps taken to reach answer
        """
        self.query = query
        self.answer = answer
        self.confidence = confidence
        self.evidence = evidence or []
        self.reasoning_path = reasoning_path or []

    def __str__(self) -> str:
        """Human-readable representation"""
        lines = [
            f"Query: {self.query}",
            f"Answer: {self.answer}",
            f"Confidence: {self.confidence:.2%}",
        ]

        if self.evidence:
            lines.append(f"Evidence ({len(self.evidence)} items):")
            for i, ev in enumerate(self.evidence[:3], 1):
                lines.append(f"  {i}. {ev}")

        if self.reasoning_path:
            lines.append(f"Reasoning:")
            for step in self.reasoning_path:
                lines.append(f"  → {step}")

        return "\n".join(lines)


class QuerySystem:
    """
    Real-time query interface

    Integrates with:
    - Knowledge Graph (entities and relations)
    - Ingestion Pipeline (raw knowledge units)
    - Phase 2 Semantic Memory (learned concepts and rules)

    Query processing:
    1. Parse query
    2. Identify query type
    3. Execute appropriate query strategy
    4. Synthesize answer
    5. Return with confidence and evidence
    """

    def __init__(self,
                 knowledge_graph: UniversalKnowledgeGraph,
                 ingestion_pipeline: KnowledgeIngestionPipeline):
        """
        Args:
            knowledge_graph: Universal knowledge graph
            ingestion_pipeline: Knowledge ingestion pipeline
        """
        self.kg = knowledge_graph
        self.ingestion = ingestion_pipeline

        # Query statistics
        self.total_queries = 0
        self.query_types = {}

    def query(self, query_text: str) -> QueryResult:
        """
        Execute query and return result

        Args:
            query_text: Natural language query

        Returns:
            QueryResult with answer and metadata
        """
        self.total_queries += 1

        # Determine query type
        query_type = self._classify_query(query_text)
        self.query_types[query_type] = self.query_types.get(query_type, 0) + 1

        # Execute appropriate query strategy
        if query_type == "entity_lookup":
            return self._query_entity(query_text)

        elif query_type == "relation_query":
            return self._query_relation(query_text)

        elif query_type == "path_query":
            return self._query_path(query_text)

        elif query_type == "search":
            return self._query_search(query_text)

        else:
            # Default: general search
            return self._query_general(query_text)

    def _classify_query(self, query_text: str) -> str:
        """
        Classify query type based on keywords

        Returns:
            Query type string
        """
        query_lower = query_text.lower()

        # Entity lookup: "what is X", "who is X", "define X"
        if any(q in query_lower for q in ["what is", "who is", "define", "explain"]):
            return "entity_lookup"

        # Relation query: "how X relate to Y", "connection between X and Y"
        elif any(q in query_lower for q in ["relate", "connection", "relationship"]):
            return "relation_query"

        # Path query: "path from X to Y", "how to get from X to Y"
        elif any(q in query_lower for q in ["path", "connect", "link"]):
            return "path_query"

        # Search: "find X", "search for X"
        elif any(q in query_lower for q in ["find", "search", "look for"]):
            return "search"

        else:
            return "general"

    def _query_entity(self, query_text: str) -> QueryResult:
        """
        Answer entity lookup query

        Example: "What is artificial intelligence?"
        """
        # Extract entity name (simple extraction)
        entity_name = self._extract_entity_name(query_text)

        if not entity_name:
            return QueryResult(
                query=query_text,
                answer="Could not identify entity in query.",
                confidence=0.0
            )

        # Look up entity
        entity = self.kg.get_entity(entity_name)

        if not entity:
            # Try knowledge units
            units = self._search_knowledge_units(entity_name, limit=3)
            if units:
                answer = f"{entity_name}: " + " ".join([u.content for u in units[:1]])
                evidence = [u.content for u in units]
                return QueryResult(
                    query=query_text,
                    answer=answer,
                    confidence=0.6,
                    evidence=evidence,
                    reasoning_path=["Searched knowledge units", f"Found {len(units)} relevant units"]
                )
            else:
                return QueryResult(
                    query=query_text,
                    answer=f"No information found about '{entity_name}'.",
                    confidence=0.0
                )

        # Build answer from entity
        answer_parts = [f"{entity.name} ({entity.type})"]

        # Add properties
        if entity.properties:
            for key, value in list(entity.properties.items())[:3]:
                answer_parts.append(f"{key}: {value}")

        # Add relations
        relations = self.kg.get_outgoing_relations(entity.name)
        if relations:
            answer_parts.append(f"Related to: {', '.join([self.kg.entities[r.tail].name for r in relations[:3]])}")

        answer = ". ".join(answer_parts)

        return QueryResult(
            query=query_text,
            answer=answer,
            confidence=0.9,
            evidence=[f"Entity: {entity.name}"],
            reasoning_path=["Found entity in knowledge graph", "Retrieved properties and relations"]
        )

    def _query_relation(self, query_text: str) -> QueryResult:
        """
        Answer relation query

        Example: "How is X related to Y?"
        """
        # Extract two entities (simple extraction)
        entities = self._extract_two_entities(query_text)

        if len(entities) < 2:
            return QueryResult(
                query=query_text,
                answer="Could not identify two entities in query.",
                confidence=0.0
            )

        entity1, entity2 = entities[0], entities[1]

        # Find paths between entities
        paths = self.kg.query_path(entity1, entity2, max_hops=3)

        if not paths:
            return QueryResult(
                query=query_text,
                answer=f"No direct relationship found between '{entity1}' and '{entity2}'.",
                confidence=0.5
            )

        # Build answer from shortest path
        path = paths[0]
        answer = f"{entity1} is connected to {entity2} via: " + " → ".join(path)

        return QueryResult(
            query=query_text,
            answer=answer,
            confidence=0.8,
            evidence=[f"Path: {' → '.join(path)}"],
            reasoning_path=["Searched for paths", f"Found {len(paths)} paths", "Selected shortest path"]
        )

    def _query_path(self, query_text: str) -> QueryResult:
        """
        Answer path query

        Example: "What connects X and Y?"
        """
        # Similar to relation query but focus on path
        return self._query_relation(query_text)

    def _query_search(self, query_text: str) -> QueryResult:
        """
        Answer search query

        Example: "Find information about machine learning"
        """
        # Extract search term
        search_term = self._extract_search_term(query_text)

        if not search_term:
            return QueryResult(
                query=query_text,
                answer="Could not identify search term.",
                confidence=0.0
            )

        # Search entities
        entities = self.kg.search_entities(search_term, limit=5)

        # Search knowledge units
        ku = self._search_knowledge_units(search_term, limit=5)

        if not entities and not ku:
            return QueryResult(
                query=query_text,
                answer=f"No results found for '{search_term}'.",
                confidence=0.0
            )

        # Build answer
        answer_parts = []

        if entities:
            answer_parts.append(f"Found {len(entities)} entities:")
            for e in entities[:3]:
                answer_parts.append(f"- {e.name} ({e.type})")

        if ku:
            answer_parts.append(f"Found {len(ku)} knowledge units:")
            answer_parts.append(ku[0].content)

        answer = "\n".join(answer_parts)

        return QueryResult(
            query=query_text,
            answer=answer,
            confidence=0.7,
            evidence=[f"Entities: {len(entities)}", f"Knowledge units: {len(ku)}"],
            reasoning_path=["Searched knowledge graph", "Searched knowledge units"]
        )

    def _query_general(self, query_text: str) -> QueryResult:
        """
        Answer general query (fallback)

        Uses keyword search across all knowledge
        """
        # Extract keywords
        keywords = query_text.lower().split()

        # Search knowledge units
        all_units = self.ingestion.get_knowledge_units()
        relevant = []

        for unit in all_units:
            score = sum(1 for kw in keywords if kw in unit.content.lower())
            if score > 0:
                relevant.append((unit, score))

        # Sort by score
        relevant.sort(key=lambda x: x[1], reverse=True)

        if not relevant:
            return QueryResult(
                query=query_text,
                answer="No relevant information found.",
                confidence=0.0
            )

        # Return top result
        best_unit, score = relevant[0]

        return QueryResult(
            query=query_text,
            answer=best_unit.content,
            confidence=min(score / len(keywords), 1.0) * 0.6,
            evidence=[u.content for u, _ in relevant[:3]],
            reasoning_path=["Keyword search", f"Found {len(relevant)} matching units"]
        )

    # Helper methods

    def _extract_entity_name(self, query_text: str) -> Optional[str]:
        """
        Extract entity name from query

        Simple version: Take words after "what is" / "who is"
        """
        query_lower = query_text.lower()

        for phrase in ["what is", "who is", "define", "explain"]:
            if phrase in query_lower:
                parts = query_lower.split(phrase, 1)
                if len(parts) > 1:
                    name = parts[1].strip().rstrip('?').strip()
                    return name

        # Fallback: use whole query
        return query_text.strip().rstrip('?').strip()

    def _extract_two_entities(self, query_text: str) -> List[str]:
        """Extract two entity names from query"""
        # Simple version: split on "and", "to", "between"
        for separator in [" and ", " to ", " between "]:
            if separator in query_text.lower():
                parts = query_text.lower().split(separator)
                if len(parts) >= 2:
                    e1 = parts[0].strip().rstrip('?').strip()
                    e2 = parts[1].strip().rstrip('?').strip()
                    # Clean up
                    e1 = e1.replace("how", "").replace("is", "").replace("related", "").strip()
                    e2 = e2.replace("how", "").replace("is", "").replace("related", "").strip()
                    return [e1, e2]

        return []

    def _extract_search_term(self, query_text: str) -> Optional[str]:
        """Extract search term from query"""
        query_lower = query_text.lower()

        for phrase in ["find", "search for", "look for"]:
            if phrase in query_lower:
                parts = query_lower.split(phrase, 1)
                if len(parts) > 1:
                    term = parts[1].strip().rstrip('?').strip()
                    return term

        # Fallback
        return query_text.strip().rstrip('?').strip()

    def _search_knowledge_units(self, search_term: str, limit: int = 5) -> List[KnowledgeUnit]:
        """Search knowledge units by content"""
        all_units = self.ingestion.get_knowledge_units()
        matches = []

        search_lower = search_term.lower()

        for unit in all_units:
            if search_lower in unit.content.lower():
                matches.append(unit)

            if len(matches) >= limit:
                break

        return matches

    def get_statistics(self) -> Dict:
        """Get query system statistics"""
        return {
            'total_queries': self.total_queries,
            'query_types': self.query_types
        }
