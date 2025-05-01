import os
from typing import List, Dict, Any, Optional
from langchain.llms import LlamaCpp
from langchain_core.retrievers import BaseRetriever
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain_core.language_models.llms import LLM
from langchain.chains import LLMChain

class LLMInterface:
    """LLM ile iletişim kurmak için arayüz sınıfı"""
    
    def __init__(
        self, 
        model_path: str,
        temperature: float = 0.1,
        max_tokens: int = 2048,
        top_p: float = 0.9,
        context_window: int = 8192,
        verbose: bool = True
    ):
        """
        Args:
            model_path: LLaMA model dosyasının yolu
            temperature: Çıktı çeşitliliğini kontrol eden sıcaklık parametresi
            max_tokens: Oluşturulacak maksimum token sayısı
            top_p: Nucleus sampling parametresi
            context_window: Bağlam penceresi boyutu
            verbose: Ayrıntılı çıktı göster
        """
        try:
            self.llm = LlamaCpp(
                model_path=model_path,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p, 
                n_ctx=context_window,
                n_batch=512,  # Batch boyutunu artıralım
                f16_kv=True,  # Bellek optimizasyonu için
                verbose=verbose
            )
        except Exception as e:
            raise ValueError(f"LLM model yüklenemedi: {e}")
        
        # Kendi basit bellek yönetimimizi oluşturalım
        self.chat_history = []
        self.retriever = None
        self.qa_chain = None
    
    def setup_qa_chain(self, retriever: BaseRetriever) -> None:
        """Soru-Cevap zincirini kurar
        
        Args:
            retriever: Belge getirici (retriever)
        """
        self.retriever = retriever
        
        # Temel soru-cevap şablonu
        QA_TEMPLATE = """<s>
You are an exceptionally knowledgeable document analysis assistant with expertise in extracting, synthesizing, and explaining information from documents. Your responses should be comprehensive, well-structured, and insightful.

When answering the user's question:
1. Base your answer SOLELY on the context information provided below.
2. If the exact answer is not found in the context, respond with "This information could not be found in the documents."
3. Provide detailed explanations with specific examples from the document when available.
4. Structure your response with clear organization - use paragraphs, bullet points when appropriate.
5. Include direct quotes from the document when relevant, citing exactly where the information comes from.
6. If the question requires analysis, provide thoughtful interpretation while remaining faithful to the document's content.
7. If technical terms appear in the document, briefly explain them if necessary for understanding.
8. Ensure your answer is complete, addressing all aspects of the user's question.

Remember, your goal is to be as helpful and informative as possible while staying strictly within the bounds of the provided context.
</s>

<context>
{context}
</context>

Question: {question}

Answer:"""
        
        # Soru-Cevap zincirini oluştur
        qa_prompt = PromptTemplate.from_template(QA_TEMPLATE)
        self.qa_chain = LLMChain(llm=self.llm, prompt=qa_prompt)
    
    def answer_question(self, question: str) -> Dict[str, Any]:
        """Kullanıcı sorusunu cevaplar
        
        Args:
            question: Kullanıcı sorusu
            
        Returns:
            Dict[str, Any]: Cevap ve kaynak belgeler
        """
        if self.retriever is None or self.qa_chain is None:
            raise ValueError("QA zinciri henüz kurulmadı. setup_qa_chain() metodunu çağırın.")
        
        try:
            # Belge alıcıyı (retriever) kullanarak ilgili belgeleri al
            retrieved_docs = self.retriever.get_relevant_documents(question)
            
            # Belgeleri birleştirerek bağlam oluştur
            context = "\n\n".join([doc.page_content for doc in retrieved_docs])
            
            # Soru-cevap zincirini çalıştır
            answer = self.qa_chain.predict(context=context, question=question)
            
            # Kullanıcı sorusunu ve cevabı sohbet geçmişine ekle
            self.chat_history.append({"role": "user", "content": question})
            self.chat_history.append({"role": "assistant", "content": answer})
            
            # Sonuçları döndür
            return {
                "answer": answer,
                "source_documents": retrieved_docs
            }
                
        except Exception as e:
            # Hata durumunda
            import traceback
            error_msg = f"Soru yanıtlanırken hata oluştu: {str(e)}"
            print(traceback.format_exc())
            
            return {
                "answer": error_msg,
                "source_documents": []
            }
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Sohbet geçmişini döndürür
        
        Returns:
            List[Dict[str, str]]: Sohbet geçmişi
        """
        return self.chat_history
    
    def clear_conversation_history(self) -> None:
        """Sohbet geçmişini temizler"""
        self.chat_history = [] 