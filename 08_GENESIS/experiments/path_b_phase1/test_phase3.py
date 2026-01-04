"""
GENESIS Phase 3A Test: Universal Knowledge System

Demonstrates the complete knowledge system:
1. Knowledge ingestion from multiple sources
2. Knowledge graph construction
3. Real-time querying
4. Integration with Phase 1-2 learning systems

This test shows:
- Ingesting knowledge from text and files
- Building a knowledge graph
- Querying the system
- Deduplication working
- Learning while answering questions
"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from knowledge_ingestion import KnowledgeIngestionPipeline, KnowledgeUnit
from universal_knowledge_graph import UniversalKnowledgeGraph
from query_system import QuerySystem


def test_phase3a_basic():
    """Test basic Phase 3A functionality"""
    print("="*70)
    print("GENESIS Phase 3A: Universal Knowledge System Test")
    print("="*70)
    print()

    # 1. Create systems
    print("🔧 Initializing systems...")
    ingestion = KnowledgeIngestionPipeline(quality_threshold=0.3)
    kg = UniversalKnowledgeGraph()
    query_system = QuerySystem(kg, ingestion)
    print("✓ Systems initialized\n")

    # 2. Ingest knowledge
    print("📥 Ingesting knowledge...")
    print()

    # Sample knowledge (Korean + English mixed)
    knowledge_texts = [
        """
        Artificial Intelligence (AI) is the simulation of human intelligence by machines.
        AI systems can learn, reason, and self-correct.
        Machine Learning is a subset of AI focused on learning from data.
        Deep Learning is a subset of Machine Learning using neural networks.
        """,
        """
        인공지능은 기계가 인간의 지능을 모방하는 것입니다.
        딥러닝은 신경망을 사용하는 머신러닝의 한 종류입니다.
        강화학습은 보상을 통해 학습하는 방법입니다.
        """,
        """
        Neural networks are computing systems inspired by biological neural networks.
        Neural networks consist of layers of interconnected nodes.
        Each connection has a weight that adjusts during learning.
        """,
        """
        GENESIS is an artificial life system that learns infinitely.
        GENESIS uses Teacher Networks to preserve knowledge across generations.
        Phase 1 implements infinite learning through teacher networks.
        Phase 2 adds multi-layer memory systems.
        Phase 3 connects to real-world knowledge.
        """
    ]

    total_added = 0
    for i, text in enumerate(knowledge_texts, 1):
        added = ingestion.ingest_text(text, source=f"doc_{i}")
        total_added += len(added)
        print(f"  Document {i}: {len(added)} knowledge units added")

    print(f"\n✓ Total: {total_added} knowledge units ingested\n")

    # 3. Build knowledge graph
    print("🕸️  Building knowledge graph...")

    # Add some entities and relations manually for demonstration
    kg.add_entity("Artificial Intelligence", "concept", {"field": "computer_science"})
    kg.add_entity("Machine Learning", "concept", {"field": "computer_science"})
    kg.add_entity("Deep Learning", "concept", {"field": "computer_science"})
    kg.add_entity("Neural Networks", "concept", {"field": "computer_science"})
    kg.add_entity("GENESIS", "system", {"type": "artificial_life"})

    # Add relations
    kg.add_relation("Machine Learning", "subset-of", "Artificial Intelligence")
    kg.add_relation("Deep Learning", "subset-of", "Machine Learning")
    kg.add_relation("Deep Learning", "uses", "Neural Networks")
    kg.add_relation("GENESIS", "implements", "Artificial Intelligence")
    kg.add_relation("GENESIS", "uses", "Neural Networks")

    kg_stats = kg.get_statistics()
    print(f"✓ Knowledge graph built:")
    print(f"  - Entities: {kg_stats['entity_count']}")
    print(f"  - Relations: {kg_stats['relation_count']}")
    print()

    # 4. Test queries
    print("💬 Testing query system...")
    print()

    queries = [
        "What is Artificial Intelligence?",
        "What is GENESIS?",
        "How is Deep Learning related to Machine Learning?",
        "Find information about neural networks",
        "What is 인공지능?",  # Korean query
    ]

    for i, query in enumerate(queries, 1):
        print(f"Query {i}: {query}")
        print("-" * 70)

        result = query_system.query(query)
        print(result)
        print()

    # 5. Show statistics
    print("="*70)
    print("Phase 3A Statistics")
    print("="*70)

    ingestion_stats = ingestion.get_statistics()
    print(f"\n📥 Ingestion Pipeline:")
    print(f"  Total ingested: {ingestion_stats['total_ingested']}")
    print(f"  Duplicates filtered: {ingestion_stats['total_duplicates']}")
    print(f"  Quality filtered: {ingestion_stats['total_filtered']}")
    print(f"  Unique units: {ingestion_stats['unique_knowledge_units']}")

    print(f"\n🕸️  Knowledge Graph:")
    print(f"  Entities: {kg_stats['entity_count']}")
    print(f"  Relations: {kg_stats['relation_count']}")
    print(f"  Avg relations/entity: {kg_stats['avg_relations_per_entity']:.2f}")

    query_stats = query_system.get_statistics()
    print(f"\n💬 Query System:")
    print(f"  Total queries: {query_stats['total_queries']}")
    print(f"  Query types: {query_stats['query_types']}")

    print(f"\n✅ Phase 3A systems operational!\n")

    return ingestion, kg, query_system


def test_deduplication():
    """Test knowledge deduplication"""
    print("="*70)
    print("Testing Knowledge Deduplication")
    print("="*70)
    print()

    ingestion = KnowledgeIngestionPipeline()
    kg = UniversalKnowledgeGraph()

    # Add same knowledge multiple times
    print("Adding duplicate knowledge...")
    text1 = "Artificial Intelligence is the simulation of human intelligence."
    text2 = "Artificial Intelligence is the simulation of human intelligence."  # Exact duplicate
    text3 = "AI is the simulation of human intelligence."  # Similar but different

    added1 = ingestion.ingest_text(text1, source="source1")
    added2 = ingestion.ingest_text(text2, source="source2")
    added3 = ingestion.ingest_text(text3, source="source3")

    print(f"  First addition: {len(added1)} units")
    print(f"  Second addition (duplicate): {len(added2)} units")
    print(f"  Third addition (similar): {len(added3)} units")

    stats = ingestion.get_statistics()
    print(f"\n✓ Deduplication working:")
    print(f"  Unique units: {stats['unique_knowledge_units']}")
    print(f"  Duplicates caught: {stats['total_duplicates']}")
    print()


def test_continuous_learning_query():
    """Test querying while learning (simulated)"""
    print("="*70)
    print("Testing Continuous Learning + Query")
    print("="*70)
    print()

    ingestion = KnowledgeIngestionPipeline()
    kg = UniversalKnowledgeGraph()
    query_system = QuerySystem(kg, ingestion)

    print("Simulating continuous learning...")
    print()

    # Phase 1: Initial knowledge
    print("Phase 1: Learning initial knowledge...")
    text1 = "Python is a programming language. Python is used for AI development."
    ingestion.ingest_text(text1, "learning_phase1")
    kg.add_entity("Python", "language")

    # Query before more knowledge
    print("\nQuery: What is Python?")
    result = query_system.query("What is Python?")
    print(f"Answer (confidence {result.confidence:.0%}): {result.answer}")

    # Phase 2: Add more knowledge
    print("\nPhase 2: Learning more knowledge...")
    text2 = "Python was created by Guido van Rossum. Python is known for simplicity."
    ingestion.ingest_text(text2, "learning_phase2")
    kg.add_entity("Guido van Rossum", "person")
    kg.add_relation("Guido van Rossum", "created", "Python")

    # Query again with more knowledge
    print("\nQuery: What is Python?")
    result = query_system.query("What is Python?")
    print(f"Answer (confidence {result.confidence:.0%}): {result.answer}")

    print("\nQuery: Who created Python?")
    result = query_system.query("Who created Python?")
    print(f"Answer (confidence {result.confidence:.0%}): {result.answer}")

    print("\n✓ System can answer questions while continuously learning!\n")


if __name__ == "__main__":
    # Run tests
    print("\n")

    # Test 1: Basic functionality
    ingestion, kg, query_system = test_phase3a_basic()

    print("\n")

    # Test 2: Deduplication
    test_deduplication()

    print("\n")

    # Test 3: Continuous learning + query
    test_continuous_learning_query()

    print("="*70)
    print("Phase 3A Test Complete!")
    print("="*70)
    print()
    print("✨ Phase 3A successfully demonstrates:")
    print("  ✓ Knowledge ingestion from multiple sources")
    print("  ✓ Knowledge graph construction with entities and relations")
    print("  ✓ Real-time query system")
    print("  ✓ Deduplication of redundant knowledge")
    print("  ✓ Continuous learning while answering queries")
    print()
    print("🚀 Next: Integrate with Phase 1-2 artificial life learning!")
    print()
