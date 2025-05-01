import os
from typing import List, Optional, Dict, Any, Union
from langchain_community.vectorstores import FAISS, Chroma
# Güncellenmiş import - HuggingFaceEmbeddings'i doğru paketten alma
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    print("langchain_huggingface paketi kurulamadı, eski sürüm kullanılıyor")
from langchain_core.documents import Document
import numpy as np
from semantic_chunker.core import SemanticChunker

class VectorStore:
    """Belge vektörlerini yönetmek için sınıf"""
    
    def __init__(self, db_type: str = "faiss", embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        """
        Args:
            db_type: Vektör veritabanı tipi ("faiss" veya "chroma")
            embedding_model: Embedding modelinin adı
        """
        self.db_type = db_type.lower()
        self.embedding_model = embedding_model
        # Model yapılandırması - daha büyük maksimum uzunluk belirt
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": "mps"},
            encode_kwargs={"normalize_embeddings": True, "max_length": 1024}
        )
        self.vector_store = None
        
    def add_texts(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None, 
                  use_semantic_chunking: bool = False, max_tokens: int = 512, 
                  similarity_threshold: float = 0.5) -> None:
        """Metin parçalarını vektör veritabanına ekler, isteğe bağlı semantik parçalama ile
        
        Args:
            texts: Metin parçaları listesi
            metadatas: Her metin parçası için meta veri (opsiyonel)
            use_semantic_chunking: Semantik parçalama kullanılsın mı?
            max_tokens: Semantik parçalama için maksimum token sayısı
            similarity_threshold: Semantik benzerlik eşiği
        """
        if not texts:
            raise ValueError("Eklenecek metin bulunamadı.")
            
        processed_texts = texts
        processed_metadatas = metadatas
        
        # Semantik parçalama istendiyse uygula
        if use_semantic_chunking:
            processed_texts, processed_metadatas = self._apply_semantic_chunking(
                texts, metadatas, max_tokens, similarity_threshold
            )
            
        if self.vector_store is None:
            # İlk kez ekleme, veritabanını oluştur
            self._create_vector_store(processed_texts, processed_metadatas)
        else:
            # Var olan veritabanına ekle
            self.vector_store.add_texts(processed_texts, processed_metadatas)
    
    def _apply_semantic_chunking(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]], 
                                max_tokens: int = 512, similarity_threshold: float = 0.5) -> tuple:
        """Metinleri semantik olarak parçalar ve birleştirir
        
        Args:
            texts: Metin parçaları listesi
            metadatas: Meta veriler
            max_tokens: Maksimum token sayısı
            similarity_threshold: Benzerlik eşiği
            
        Returns:
            tuple: (Birleştirilmiş metinler, güncellenmiş meta veriler)
        """
        try:
            # Metin parçalarını formatla
            chunks = [{"text": text} for text in texts]
            
            # Semantic Chunker oluştur
            chunker = SemanticChunker(
                max_tokens=max_tokens, 
                similarity_threshold=similarity_threshold
            )
            
            # Parçaları semantik olarak birleştir
            merged_chunks = chunker.chunk(chunks)
            
            # Birleştirilmiş metinleri çıkar
            new_texts = [chunk["text"] for chunk in merged_chunks]
            
            # Meta verileri güncelle
            new_metadatas = None
            if metadatas:
                new_metadatas = []
                for i, merged_chunk in enumerate(merged_chunks):
                    # Hangi indekslerdeki parçalar birleştirildi?
                    merged_indices = []
                    for original_chunk in merged_chunk["metadata"]:
                        for j, text in enumerate(texts):
                            if original_chunk["text"] == text:
                                merged_indices.append(j)
                                break
                    
                    # Birleştirilmiş meta verileri oluştur
                    merged_metadata = {}
                    sources = set()
                    
                    for idx in merged_indices:
                        if idx < len(metadatas) and metadatas[idx]:
                            # Kaynak adlarını birleştir
                            if "source" in metadatas[idx]:
                                sources.add(metadatas[idx]["source"])
                            
                            # Diğer meta verileri birleştir
                            for key, value in metadatas[idx].items():
                                if key != "source":
                                    merged_metadata[key] = value
                    
                    # Kaynakları ekle
                    if sources:
                        merged_metadata["source"] = ", ".join(sources)
                    
                    # Semantik parçalama bilgisini ekle
                    merged_metadata["semantic_chunk"] = True
                    merged_metadata["original_chunks"] = len(merged_indices)
                    
                    new_metadatas.append(merged_metadata)
            
            return new_texts, new_metadatas
        except Exception as e:
            print(f"Semantik parçalama hatası: {e}. Orijinal metinler kullanılıyor.")
            return texts, metadatas
    
    def _create_vector_store(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None) -> None:
        """Vektör veritabanını oluşturur
        
        Args:
            texts: Metin parçaları listesi
            metadatas: Her metin parçası için meta veri
        """
        if self.db_type == "faiss":
            self.vector_store = FAISS.from_texts(texts, self.embeddings, metadatas=metadatas)
        elif self.db_type == "chroma":
            self.vector_store = Chroma.from_texts(texts, self.embeddings, metadatas=metadatas)
        else:
            raise ValueError(f"Desteklenmeyen vektör veritabanı tipi: {self.db_type}")
    
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Sorguya en benzer metin parçalarını bulur
        
        Args:
            query: Arama sorgusu
            k: Döndürülecek benzer belge sayısı
            
        Returns:
            List[Document]: En benzer belgelerin listesi
        """
        if self.vector_store is None:
            raise ValueError("Vektör veritabanı henüz oluşturulmadı.")
            
        return self.vector_store.similarity_search(query, k=k)
    
    def save(self, path: str) -> None:
        """Vektör veritabanını kaydeder
        
        Args:
            path: Kayıt klasörü
        """
        if self.vector_store is None:
            raise ValueError("Kaydedilecek vektör veritabanı bulunamadı.")
            
        os.makedirs(path, exist_ok=True)
        
        if self.db_type == "faiss":
            self.vector_store.save_local(path)
        elif self.db_type == "chroma":
            self.vector_store.persist(path)
    
    def load(self, path: str) -> None:
        """Kayıtlı vektör veritabanını yükler
        
        Args:
            path: Veritabanı klasörü
        """
        if not os.path.exists(path):
            raise ValueError(f"Veritabanı klasörü bulunamadı: {path}")
            
        if self.db_type == "faiss":
            self.vector_store = FAISS.load_local(path, self.embeddings)
        elif self.db_type == "chroma":
            self.vector_store = Chroma(persist_directory=path, embedding_function=self.embeddings)
    
    def clear(self) -> None:
        """Vektör veritabanını temizler"""
        self.vector_store = None 