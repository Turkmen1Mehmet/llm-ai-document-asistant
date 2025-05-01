# Semantic Chunker Entegrasyonu

Bu belge, [Semantic Chunker](https://github.com/rango-ramesh/advanced-chunker) kütüphanesinin projemize nasıl entegre edildiğini açıklamaktadır.

## Entegrasyon Özeti

Semantic Chunker, metin belgelerini anlamsal olarak daha tutarlı parçalara bölmek için kullanılan bir Python kütüphanesidir. Geleneksel metin parçalama yöntemlerinden farklı olarak, metni anlamsal benzerliklerine göre gruplar, böylece ilişkili fikirler bir arada tutulur.

Bu projede Semantic Chunker'ı şu şekilde entegre ettik:

1. **requirements.txt** dosyasına kütüphaneyi ekledik.
2. **document_loader.py** dosyasında semantik parçalama özelliği ekledik.
3. **vector_store.py** dosyasında semantik parçalama desteği ekledik.
4. **app.py** dosyasında kullanıcı arayüzü güncellemeleri yaptık.
5. **demo_semantic_chunker.py** adında örnek bir demo dosyası oluşturduk.

## Özellikler

### 1. Semantik Parçalama

Document Loader sınıfı, belgeleri yükledikten sonra semantik parçalama yapabilme özelliği kazandı:

```python
chunks = DocumentLoader.chunk_text(
    text_chunks, 
    chunk_size=1000,
    overlap=200,
    use_semantic_chunker=True
)
```

### 2. Vektör Depo Entegrasyonu

Vector Store sınıfı, metin parçalarını semantik olarak birleştirip vektör deposuna ekleme yeteneği kazandı:

```python
vector_store.add_texts(
    texts,
    metadatas=metadatas,
    use_semantic_chunking=True,
    max_tokens=512,
    similarity_threshold=0.5
)
```

### 3. Meta Veri İzleme

Semantik parçalama sırasında, birleştirilmiş parçalara ait meta verileri de akıllıca birleştiren bir mekanizma geliştirdik. Bu sayede:

- Birleştirilen parçaların kaynakları korunur
- Kaç parçanın birleştirildiği bilgisi saklanır
- Semantik parçalama yapıldığı bilgisi meta verilere eklenir

## Kullanıcı Arayüzü

Streamlit arayüzünde semantik parçalama için yeni seçenekler ekledik:

1. **Semantik Parçalama** onay kutusu: İşlem etkinleştirilir veya devre dışı bırakılır
2. **Parça Boyutu** kaydırıcısı: Maksimum karakter sayısını belirler
3. **Parça Örtüşmesi** kaydırıcısı: Parçalar arası örtüşme miktarını belirler

## Demo

`demo_semantic_chunker.py` dosyası, kütüphanenin temel özelliklerini göstermek için oluşturulmuştur. Çalıştırıldığında:

1. Örnek metin parçaları gösterilir
2. Semantic Chunker ile birleştirilmiş parçalar gösterilir
3. Benzerlik matrisi ve semantik grafik görselleştirilir

## Performans Değerlendirmesi

Semantik parçalama, belgeleri daha anlamlı birimler halinde tutmaya yardımcı olur. Bu, şu avantajları sağlar:

1. **Daha İyi Anlama**: İlişkili içerikler bir arada tutulur
2. **Daha Az Parça**: Toplam belge parçası sayısını azaltır 
3. **Daha Kaliteli Yanıtlar**: Sorguya daha alakalı içerikler döndürür

## Sonuç

Semantic Chunker entegrasyonu, doküman anlama ve sorgulama sistemimizi daha güçlü hale getirmiştir. Kullanıcılar, belgelerindeki bilgilere daha anlamlı bir şekilde erişebilir ve daha doğru sorgu yanıtları alabilirler. 