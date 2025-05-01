import os
import fitz  # PyMuPDF
import pdfplumber
import docx
from typing import List, Union, Dict
from semantic_chunker.core import SemanticChunker

class DocumentLoader:
    """Farklı belge tiplerini yüklemek ve işlemek için sınıf"""
    
    @staticmethod
    def load_document(file_path: str) -> List[str]:
        """Belge tipine göre içeriği yükler ve metin parçalarına ayırır
        
        Args:
            file_path: Belge dosya yolu
            
        Returns:
            List[str]: Belge içeriğinin parçalanmış metin listesi
        """
        _, file_extension = os.path.splitext(file_path)
        file_extension = file_extension.lower()
        
        if file_extension == '.pdf':
            return DocumentLoader._load_pdf(file_path)
        elif file_extension == '.docx':
            return DocumentLoader._load_docx(file_path)
        elif file_extension in ['.txt', '.md', '.csv']:
            return DocumentLoader._load_text(file_path)
        else:
            raise ValueError(f"Desteklenmeyen dosya formatı: {file_extension}")
    
    @staticmethod
    def _load_pdf(file_path: str) -> List[str]:
        """PDF belgesini yükler
        
        İki farklı PDF kütüphanesi kullanıyoruz, birinde sorun olursa diğerine geçer
        """
        try:
            # PyMuPDF (fitz) ile deneyelim
            document = fitz.open(file_path)
            text_chunks = []
            
            for page_num in range(len(document)):
                page = document.load_page(page_num)
                text = page.get_text("text")
                text_chunks.append(text)
                
            return text_chunks
        except Exception as e:
            print(f"PyMuPDF hatası: {e}, pdfplumber deneniyor...")
            
            # pdfplumber ile deneyelim
            with pdfplumber.open(file_path) as pdf:
                text_chunks = []
                for page in pdf.pages:
                    text = page.extract_text() or ""
                    if text.strip():
                        text_chunks.append(text)
                        
            return text_chunks
    
    @staticmethod
    def _load_docx(file_path: str) -> List[str]:
        """Word belgesini yükler"""
        doc = docx.Document(file_path)
        text_chunks = []
        
        # Paragrafları al
        for para in doc.paragraphs:
            if para.text.strip():
                text_chunks.append(para.text)
                
        return text_chunks
    
    @staticmethod
    def _load_text(file_path: str) -> List[str]:
        """Düz metin belgesini yükler"""
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            
        # Paragrafları satır boşluklarına göre bölelim
        paragraphs = [p for p in content.split('\n\n') if p.strip()]
        if not paragraphs:
            # Eğer paragraf bulunamazsa, satırlara göre bölelim
            paragraphs = [p for p in content.split('\n') if p.strip()]
            
        return paragraphs
    
    @staticmethod
    def chunk_text(text_list: List[str], chunk_size: int = 1000, overlap: int = 200, use_semantic_chunker: bool = True) -> List[str]:
        """Metin parçalarını belirli boyutlarda ve semantik olarak anlamlı parçalara böler
        
        Args:
            text_list: Metin parçaları listesi
            chunk_size: Her bir parçanın maksimum karakter sayısı
            overlap: Parçalar arası örtüşme karakter sayısı
            use_semantic_chunker: Semantik parçalama kullanılsın mı?
            
        Returns:
            List[str]: Yeniden boyutlandırılmış metin parçaları
        """
        # Çok uzun metinleri önceden bölmek için önişleme - token uzunluğu hatasını önlemek için
        preprocessed_text_list = []
        max_length = 500  # Maksimum karakter uzunluğu
        
        for text in text_list:
            if len(text) > max_length:
                # Uzun metni daha küçük parçalara böl
                sentences = [s.strip() + "." for s in text.replace(".", ".\n").split("\n") if s.strip()]
                current_chunk = ""
                
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) <= max_length:
                        current_chunk += " " + sentence
                    else:
                        if current_chunk:
                            preprocessed_text_list.append(current_chunk.strip())
                        current_chunk = sentence
                
                if current_chunk:  # Son parçayı ekle
                    preprocessed_text_list.append(current_chunk.strip())
            else:
                preprocessed_text_list.append(text)
        
        if use_semantic_chunker:
            # Parçaları SemanticChunker formatına dönüştür
            chunks = [{"text": text} for text in preprocessed_text_list if text.strip()]
            
            # Maksimum token sayısını yaklaşık karakter sayısından hesapla
            # Yaklaşık 4 karakter = 1 token
            max_tokens = chunk_size // 4
            
            # SemanticChunker'ı yapılandır ve parçalama işlemini gerçekleştir
            # Sadece desteklenen parametreleri kullan
            semantic_chunker = SemanticChunker(
                max_tokens=max_tokens,
                similarity_threshold=0.5
            )
            
            try:
                merged_chunks = semantic_chunker.chunk(chunks)
                # Sadece metin kısmını çıkar
                return [chunk["text"] for chunk in merged_chunks]
            except Exception as e:
                print(f"Semantik parçalama hatası: {e}. Geleneksel parçalamaya geçiliyor...")
                # Hata durumunda geleneksel parçalamaya geri dön
                return DocumentLoader._traditional_chunking(preprocessed_text_list, chunk_size, overlap)
        else:
            # Geleneksel parçalama yöntemini kullan
            return DocumentLoader._traditional_chunking(preprocessed_text_list, chunk_size, overlap)
    
    @staticmethod
    def _traditional_chunking(text_list: List[str], chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Geleneksel metin parçalama yöntemi"""
        result = []
        current_chunk = ""
        
        for text in text_list:
            if len(current_chunk) + len(text) <= chunk_size:
                current_chunk += text + " "
            else:
                # Mevcut parçayı kaydet
                if current_chunk:
                    result.append(current_chunk.strip())
                
                # Örtüşme miktarı kadar eski metni sakla
                words = current_chunk.split()
                overlap_text = ""
                
                if words and len(' '.join(words[-30:])) < overlap:
                    overlap_text = ' '.join(words[-30:]) + " "
                
                current_chunk = overlap_text + text + " "
        
        # Son parçayı ekle
        if current_chunk:
            result.append(current_chunk.strip())
            
        return result 