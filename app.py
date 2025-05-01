import os
import streamlit as st
import tempfile
import gc
from document_loader import DocumentLoader
from vector_store import VectorStore
from llm_interface import LLMInterface
from deep_translator import GoogleTranslator

# Belleği temizle - çakışan model örneklerini temizle
gc.collect()

# Sayfa düzeni ve stil
st.set_page_config(
    page_title="AI Belge Asistanı",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Özel CSS (renk kontrastı iyileştirmeleri)
st.markdown("""
    <style>
    /* Ana içerik alanı (beyaz arka plan) */
    section.main > div {
        background: #fff !important;
    }
    
    /* Ana içerik metinleri (koyu) */
    .main .stMarkdown h1, .main .stMarkdown h2, .main .stMarkdown h3,
    .main .stMarkdown h4, .main .stMarkdown h5, .main .stMarkdown h6 {
        color: #1e293b !important;
    }
    .main .stMarkdown p, .main .stMarkdown li, .main .stMarkdown span,
    .main .stMarkdown div {
        color: #222 !important;
    }
    
    /* Sekme başlıkları */
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        color: #1e293b !important;
        font-weight: 600 !important;
    }
    
    /* Ana içerik giriş kutuları */
    .main .stTextInput>div>div>input {
        background: #f3f4f6 !important;
        color: #222 !important;
        border: 1px solid #e5e7eb !important;
    }
    .main .stTextInput>div>div>input::placeholder {
        color: #6b7280 !important;
    }
    
    /* Ana içerik butonları */
    .main .stButton>button {
        background: linear-gradient(90deg, #2563eb 0%, #1e293b 100%) !important;
        color: #fff !important;
    }
    
    /* Sidebar (koyu arka plan) */
    [data-testid="stSidebar"] {
        background: #1a1d21 !important;
    }
    
    /* Sidebar metinleri (beyaz) */
    [data-testid="stSidebar"] .stMarkdown h1, [data-testid="stSidebar"] .stMarkdown h2,
    [data-testid="stSidebar"] .stMarkdown h3, [data-testid="stSidebar"] .stMarkdown h4,
    [data-testid="stSidebar"] .stMarkdown h5, [data-testid="stSidebar"] .stMarkdown h6 {
        color: #fff !important;
    }
    [data-testid="stSidebar"] .stMarkdown p, [data-testid="stSidebar"] .stMarkdown li,
    [data-testid="stSidebar"] .stMarkdown span, [data-testid="stSidebar"] .stMarkdown div {
        color: #fff !important;
    }
    
    /* Sidebar giriş kutuları */
    [data-testid="stSidebar"] .stTextInput>div>div>input {
        background: #23272e !important;
        color: #fff !important;
        border: 1px solid #353a40 !important;
    }
    [data-testid="stSidebar"] .stTextInput>div>div>input::placeholder {
        color: #b0b3b8 !important;
    }
    
    /* Sidebar butonları */
    [data-testid="stSidebar"] .stButton>button {
        background: #2563eb !important;
        color: #fff !important;
    }
    
    /* Genel stil ayarları */
    .stTextInput>div>div>input {
        border-radius: 8px !important;
        font-size: 1rem !important;
        padding: 0.6em 1em !important;
        height: 2.2em !important;
        min-height: 2.2em !important;
    }
    
    .stButton>button {
        width: 100%;
        border-radius: 8px !important;
        height: 2.5em !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        border: none !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
    }
    
    /* Focus durumları */
    .stTextInput>div>div>input:focus {
        outline: none !important;
        border-color: #2563eb !important;
    }
    
    /* Uyarı, bilgi ve hata mesajları için stil düzenlemeleri */
    .stAlert > div {
        padding: 0.8rem 1rem !important;
        border-radius: 8px !important;
    }
    
    /* Bilgi mesajları - daha koyu ve okunaklı */
    .stAlert.stInfo > div {
        background-color: #e0f2fe !important;
        color: #0c4a6e !important;
        border: 1px solid #bae6fd !important;
    }
    .stAlert.stInfo p, .stAlert.stInfo div {
        color: #0c4a6e !important;
        font-weight: 600 !important;
    }
    
    /* Başarı mesajları - daha koyu ve okunaklı */
    .stAlert.stSuccess > div {
        background-color: #dcfce7 !important;
        color: #166534 !important;
        border: 1px solid #86efac !important;
    }
    .stAlert.stSuccess p, .stAlert.stSuccess div {
        color: #166534 !important;
        font-weight: 600 !important;
    }
    
    /* Uyarı mesajları - daha koyu ve okunaklı */
    .stAlert.stWarning > div {
        background-color: #fef9c3 !important;
        color: #854d0e !important;
        border: 1px solid #fde047 !important;
    }
    .stAlert.stWarning p, .stAlert.stWarning div {
        color: #854d0e !important;
        font-weight: 600 !important;
    }
    
    /* Hata mesajları - daha koyu ve okunaklı */
    .stAlert.stError > div {
        background-color: #fee2e2 !important;
        color: #991b1b !important;
        border: 1px solid #fecaca !important;
    }
    .stAlert.stError p, .stAlert.stError div {
        color: #991b1b !important;
        font-weight: 600 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# Başlık ve açıklama (daha koyu ve okunaklı)
st.markdown("""
    <div style='text-align: center; margin-bottom: 2rem;'>
        <h1 style='color: #1e293b; font-size: 2.5rem; font-weight: 900; letter-spacing: -1px;'>🤖 AI Belge Asistanı</h1>
        <p style='font-size: 1.25rem; color: #222; font-weight: 500;'>
            Belgelerinizi yükleyin, <span style='color:#2563eb; font-weight:700;'>AI asistanınız</span> size yardımcı olsun!
        </p>
    </div>
    """, unsafe_allow_html=True)

# Yan menü
with st.sidebar:
    st.markdown("## ⚙️ Ayarlar")
    
    # Model yolu
    model_path = st.text_input(
        "LLaMA Model Yolu",
        "models/llama-2-7b-chat.Q4_K_M.gguf",
        help="LLaMA model dosyasının tam yolu"
    )
    
    # Veritabanı tipi
    db_type = st.selectbox(
        "Veritabanı Tipi",
        ["chroma", "faiss"],
        index=0,
        help="Vektör veritabanı tipini seçin"
    )
    
    # Gelişmiş ayarlar
    with st.expander("🎛️ Gelişmiş Ayarlar", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            temperature = st.slider("Sıcaklık", 0.0, 1.0, 0.1, help="Modelin yaratıcılık seviyesi")
            chunk_size = st.slider("Parça Boyutu", 200, 2000, 1000, help="Belge parçalama boyutu")
            
            # Semantik parçalama ayarı
            use_semantic_chunker = st.checkbox("Semantik Parçalama", True, 
                                             help="Metinleri anlamsal bütünlük koruyan parçalara böler")
        with col2:
            max_tokens = st.slider("Maksimum Token", 512, 4096, 2048, help="Maksimum yanıt uzunluğu")
            overlap = st.slider("Parça Örtüşmesi", 0, 500, 200, help="Parçalar arası örtüşme miktarı")
    
    st.markdown("---")
    st.markdown("## 📊 Durum")
    if "llm" in st.session_state:
        st.success("✅ Model hazır")
    else:
        st.warning("⚠️ Model başlatılmadı")

# Ana içerik
if "llm" not in st.session_state:
    # Model başlatma ekranı
    st.markdown("""
        <div style='text-align: center; padding: 2rem; background-color: white; border-radius: 20px;'>
            <h2>🎯 Başlamak için modeli başlatın</h2>
            <p>Belgelerinizi sorgulamak için önce AI modelini başlatmanız gerekiyor.</p>
        </div>
    """, unsafe_allow_html=True)
    
    if st.button("🚀 Modeli Başlat", key="start_model"):
        with st.spinner("Model yükleniyor..."):
            try:
                if not os.path.exists(model_path):
                    st.error(f"❌ Model dosyası bulunamadı: {model_path}")
                else:
                    st.session_state.llm = LLMInterface(
                        model_path=model_path,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                    st.session_state.vectordb = VectorStore(db_type=db_type)
                    st.success("✅ Model başarıyla başlatıldı!")
                    st.rerun()
            except Exception as e:
                st.error(f"❌ Model başlatılırken hata: {e}")
else:
    # Ana işlevsellik
    tab1, tab2 = st.tabs(["📥 Belge Yükleme", "💬 Sorgulama"])
    
    with tab1:
        st.markdown("""
            <div style='text-align: center; margin-bottom: 2rem;'>
                <h2>📥 Belgelerinizi Yükleyin</h2>
                <p>PDF, DOCX veya TXT formatındaki belgelerinizi yükleyin</p>
            </div>
        """, unsafe_allow_html=True)
        
        uploaded_files = st.file_uploader(
            "Belgelerinizi sürükleyip bırakın veya seçin",
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True
        )
        
        if uploaded_files and st.button("📤 Belgeleri İşle", key="process_docs"):
            with st.spinner("Belgeler işleniyor..."):
                try:
                    for uploaded_file in uploaded_files:
                        # Geçici dosya oluştur
                        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            tmp_path = tmp_file.name
                        
                        # Belgeyi yükle
                        text_chunks = DocumentLoader.load_document(tmp_path)
                        
                        # Belge içeriğini semantik olarak parçala
                        processed_chunks = DocumentLoader.chunk_text(
                            text_chunks, 
                            chunk_size=chunk_size,
                            overlap=overlap,
                            use_semantic_chunker=use_semantic_chunker
                        )
                        
                        # Vektör veritabanına ekle
                        st.session_state.vectordb.add_texts(
                            processed_chunks,
                            metadatas=[{"source": uploaded_file.name} for _ in processed_chunks],
                            use_semantic_chunking=use_semantic_chunker,
                            max_tokens=max_tokens,
                            similarity_threshold=0.5
                        )
                        
                        # Belge bilgisini session state'e ekle
                        if "documents" not in st.session_state:
                            st.session_state.documents = []
                        
                        # Mevcut belgede bu dosya yoksa ekle
                        doc_exists = False
                        for doc in st.session_state.documents:
                            if doc.get("name") == uploaded_file.name:
                                doc_exists = True
                                break
                        
                        if not doc_exists:
                            st.session_state.documents.append({
                                "name": uploaded_file.name,
                                "chunks": len(processed_chunks),
                                "type": uploaded_file.name.split(".")[-1].upper()
                            })
                        
                        # Geçici dosyayı temizle
                        os.unlink(tmp_path)
                    
                    # QA zincirini kur - Belgeleri yükleme tamamlandıktan sonra
                    try:
                        retriever = st.session_state.vectordb.vector_store.as_retriever()
                        st.session_state.llm.setup_qa_chain(retriever)
                    except Exception as e:
                        st.markdown(f"""
                            <div style="background-color:#fee2e2; padding:1rem; border-radius:0.5rem; border:1px solid #fecaca; margin-bottom:1rem;">
                                <p style="color:#991b1b; font-weight:700; margin:0; font-size:1rem;">
                                    ⚠️ QA sistemi kurulurken hata: {e}
                                </p>
                                <p style="color:#991b1b; margin:0.5rem 0 0 0; font-size:0.9rem;">
                                    Sorgu yapabilmeniz için QA sistemi gereklidir. Lütfen geliştiriciye başvurun.
                                </p>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    # Özel HTML ile daha koyu ve okunaklı başarı mesajı
                    st.markdown(f"""
                        <div style="background-color:#dcfce7; padding:1rem; border-radius:0.5rem; border:1px solid #86efac; margin-bottom:1rem;">
                            <p style="color:#166534; font-weight:700; margin:0; font-size:1rem;">
                                ✅ {len(uploaded_files)} belge başarıyla işlendi!
                            </p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Semantik parçalama bilgisi göster
                    if use_semantic_chunker:
                        # Özel HTML ile daha koyu ve okunaklı bilgi mesajı
                        st.markdown("""
                            <div style="background-color:#e0f2fe; padding:1rem; border-radius:0.5rem; border:1px solid #bae6fd; margin-bottom:1rem;">
                                <p style="color:#0c4a6e; font-weight:700; margin:0; font-size:1rem;">
                                    ℹ️ Belgeler, semantik parçalama ile işlendi. Bu, sorguları yanıtlarken daha iyi anlama sağlar.
                                </p>
                            </div>
                        """, unsafe_allow_html=True)
                    
                except Exception as e:
                    # Özel HTML ile daha koyu ve okunaklı hata mesajı
                    st.markdown(f"""
                        <div style="background-color:#fee2e2; padding:1rem; border-radius:0.5rem; border:1px solid #fecaca; margin-bottom:1rem;">
                            <p style="color:#991b1b; font-weight:700; margin:0; font-size:1rem;">
                                ❌ Belgeler işlenirken hata: {e}
                            </p>
                        </div>
                    """, unsafe_allow_html=True)
    
    with tab2:
        if "documents" not in st.session_state or not st.session_state.documents:
            st.markdown("""
                <div style='text-align: center; padding: 2rem; background-color: white; border-radius: 20px;'>
                    <h2>📚 Önce belge yükleyin</h2>
                    <p>Belgelerinizi sorgulamak için önce "Belge Yükleme" sekmesinden belge yüklemeniz gerekiyor.</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            if "chat_history" not in st.session_state:
                st.session_state.chat_history = []
            
            # Yüklenen belgelerin listesini göster
            st.markdown("""
                <div style="margin-bottom:1.5rem;">
                    <h3 style="font-size:1.2rem; color:#1e293b; margin-bottom:0.5rem;">📚 Yüklenen Belgeler</h3>
                </div>
            """, unsafe_allow_html=True)
            
            # Belge kartları
            doc_cols = st.columns(min(3, len(st.session_state.documents)))
            for i, doc in enumerate(st.session_state.documents):
                col_idx = i % len(doc_cols)
                with doc_cols[col_idx]:
                    st.markdown(f"""
                        <div style="background-color:#f8fafc; padding:1rem; border-radius:0.5rem; border:1px solid #e2e8f0; margin-bottom:1rem; height:100%;">
                            <p style="font-weight:600; color:#1e293b; margin:0; font-size:1rem;">
                                {doc.get("name", "Belge")}
                            </p>
                            <p style="color:#64748b; margin:0.2rem 0 0 0; font-size:0.9rem;">
                                <span style="background-color:#e2e8f0; padding:0.1rem 0.4rem; border-radius:0.3rem; font-size:0.7rem;">
                                    {doc.get("type", "DOC")}
                                </span> 
                                <span>{doc.get("chunks", 0)} parça</span>
                            </p>
                        </div>
                    """, unsafe_allow_html=True)
            
            st.markdown("<hr style='margin:1.5rem 0;'>", unsafe_allow_html=True)
            
            # Sohbet geçmişi
            for i, message in enumerate(st.session_state.chat_history):
                with st.chat_message(message["role"]):
                    if message["role"] == "user":
                        st.write(message["content"])
                    else:  # assistant
                        st.write(message["content"])
                        
                        # Çeviri butonları - küçük ve sağ köşeye hizalı
                        cols = st.columns([6, 0.5, 0.5])
                        with cols[1]:
                            if st.button("🇹🇷", key=f"tr_hist_{i}"):
                                try:
                                    tr_translator = GoogleTranslator(source='en', target='tr')
                                    tr_answer = tr_translator.translate(message["content"])
                                    st.session_state[f"answer_hist_{i}"] = tr_answer
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Çeviri hatası: {str(e)}")
                                
                        with cols[2]:
                            if st.button("🇬🇧", key=f"en_hist_{i}"):
                                st.session_state[f"answer_hist_{i}"] = message["content"]
                                st.rerun()
                        
                        # Eğer çevirisi varsa göster
                        if f"answer_hist_{i}" in st.session_state:
                            st.markdown("---")
                            st.write(st.session_state[f"answer_hist_{i}"])
                            
                        if "sources" in message and message["sources"]:
                            with st.expander("📚 Kaynaklar"):
                                for source in message["sources"]:
                                    st.markdown(f"""
                                        <div style='padding: 0.5rem; background-color: #f0f0f0; border-radius: 5px; margin-bottom: 0.5rem;'>
                                            <p style='margin: 0;'><strong>{source['source']}</strong></p>
                                            <p style='margin: 0; color: #666;'>{source['content']}</p>
                                        </div>
                                    """, unsafe_allow_html=True)
            
            # Sorgu girişi
            question = st.chat_input("Belgeleriniz hakkında bir soru sorun...")
            
            if question:
                with st.chat_message("user"):
                    st.write(question)
                
                with st.chat_message("assistant"):
                    with st.spinner("🤔 Düşünüyorum..."):
                        try:
                            result = st.session_state.llm.answer_question(question)
                            answer = result.get("answer", "Bu soruya cevap bulunamadı.")
                            source_docs = result.get("source_documents", [])
                            
                            sources = []
                            for doc in source_docs:
                                if hasattr(doc, 'metadata') and hasattr(doc, 'page_content'):
                                    sources.append({
                                        "source": doc.metadata.get("source", "Bilinmeyen Kaynak"),
                                        "content": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                                    })
                            
                            # Cevabı göster
                            st.write(answer)
                            
                            # Çeviri butonu için küçük bir alan
                            st.markdown("""
                                <style>
                                .translator-buttons {
                                    display: flex;
                                    justify-content: flex-end;
                                    gap: 5px;
                                    margin-top: 5px;
                                }
                                .translator-buttons button {
                                    font-size: 0.7rem !important;
                                    padding: 2px 8px !important;
                                    height: auto !important;
                                    min-height: 0 !important;
                                    width: auto !important;
                                }
                                </style>
                            """, unsafe_allow_html=True)
                            
                            # Dil seçenekleri - küçük butonlar olarak
                            cols = st.columns([6, 0.5, 0.5])
                            with cols[1]:
                                if st.button("🇹🇷", key=f"tr_btn_{len(st.session_state.chat_history)}"):
                                    # Türkçe çeviri
                                    try:
                                        tr_translator = GoogleTranslator(source='en', target='tr')
                                        tr_answer = tr_translator.translate(answer)
                                        st.session_state[f"answer_{len(st.session_state.chat_history)}"] = tr_answer
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"Çeviri hatası: {str(e)}")
                            
                            with cols[2]:
                                if st.button("🇬🇧", key=f"en_btn_{len(st.session_state.chat_history)}"):
                                    # İngilizce orijinal
                                    st.session_state[f"answer_{len(st.session_state.chat_history)}"] = answer
                                    st.rerun()
                            
                            # Eğer oturum durumunda çevirisi varsa onu göster
                            if f"answer_{len(st.session_state.chat_history)}" in st.session_state:
                                st.markdown("---")
                                st.write(st.session_state[f"answer_{len(st.session_state.chat_history)}"])
                            
                            if sources:
                                with st.expander("📚 Kaynaklar"):
                                    for source in sources:
                                        st.markdown(f"""
                                            <div style='padding: 0.5rem; background-color: #f0f0f0; border-radius: 5px; margin-bottom: 0.5rem;'>
                                                <p style='margin: 0;'><strong>{source['source']}</strong></p>
                                                <p style='margin: 0; color: #666;'>{source['content']}</p>
                                            </div>
                                        """, unsafe_allow_html=True)
                            
                            st.session_state.chat_history.append({
                                "role": "assistant",
                                "content": answer,
                                "sources": sources
                            })
                            
                        except Exception as e:
                            st.error(f"❌ Yanıt oluşturulurken hata: {str(e)}")
            
            if st.button("🗑️ Sohbeti Temizle"):
                st.session_state.chat_history = []
                st.rerun()

# Altbilgi
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Bu uygulama, belge sorgulaması için LLaMA, LangChain ve vektör veritabanları kullanır.</p>
    </div>
""", unsafe_allow_html=True)
