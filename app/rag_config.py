# """
# Improved RAG configuration for better accuracy
# """
# from dataclasses import dataclass
# from typing import Optional, List
# import os

# @dataclass
# class RAGConfig:
#     """Configuration for optimized RAG performance"""
    
#     # Chunking settings
#     chunk_size: int = 800
#     chunk_overlap: int = 150
    
#     # Embedding settings
#     embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
#     embedding_dimension: int = 384
    
#     # Retrieval settings
#     top_k_retrieve: int = 8
#     top_k_return: int = 5
#     fetch_k_multiplier: int = 3
#     similarity_threshold: float = 0.7
    
#     # LLM settings
#     use_llm: bool = True
#     llm_model: str = "llama3.2"
#     llm_temperature: float = 0.1
#     llm_max_tokens: int = 500
    
#     # Hybrid search settings
#     use_hybrid_search: bool = True
#     bm25_weight: float = 0.3
#     vector_weight: float = 0.7
    
#     # Context settings
#     max_context_length: int = 3000
#     min_relevant_chunks: int = 3
    
#     @property
#     def fetch_k(self) -> int:
#         return self.top_k_retrieve * self.fetch_k_multiplier
    
#     @classmethod
#     def get_optimized_for_ngo(cls) -> 'RAGConfig':
#         """Get configuration optimized for NGO proposal documents"""
#         return cls(
#             chunk_size=750,
#             chunk_overlap=120,
#             top_k_retrieve=10,
#             top_k_return=6,
#             similarity_threshold=0.65,
#             use_hybrid_search=True,
#             max_context_length=3500
#         )