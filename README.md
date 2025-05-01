# AI Belge Asistanı

Bu projede, belgelerinizi yükleyip yapay zeka yardımıyla sorgulayabileceğiniz bir uygulama geliştirilmiştir.

## Özellikler

- **Belge Yükleme**: PDF, DOCX ve TXT formatında belgeler desteklenir.
- **Semantik Parçalama**: Belge içeriği, anlamsal bütünlüğü koruyacak şekilde parçalara ayrılır.
- **Vektör Tabanlı Sorgulama**: Belgelerinizi doğal dilde sorgulayabilirsiniz.
- **LLaMA Tabanlı Yanıtlama**: Sorularınıza belgelerinizdeki bilgilere dayalı kapsamlı yanıtlar alabilirsiniz.

## Semantik Parçalama Nedir?

Geleneksel metin parçalama yöntemleri, metni sabit boyutlarda veya basit sınırlara göre ayırır. Bu durum, anlamsal bütünlüğün bozulmasına, ilişkili fikirlerin ayrılmasına veya bağlamsal tutarlılığın yakalanamamasına neden olabilir.

Semantik Parçalama ise, cümle gömmeleri ve kümeleme kullanarak anlamsal olarak benzer parçaları daha tutarlı birimler halinde birleştirir. Bu, daha verimli geri alma, daha net gömmeler ve daha etkili yanıtlar sağlar.

## Kullanım

1. Yan menüden model yolunu ve diğer ayarları yapılandırın
2. "Modeli Başlat" düğmesine tıklayın
3. "Belge Yükleme" sekmesinden belgelerinizi yükleyin
4. "Sorgulama" sekmesinden belgelerinizi sorgulayın

## Teknik Detaylar

Bu projede kullanılan başlıca teknolojiler:

- **Streamlit**: Kullanıcı arayüzü
- **LangChain**: LLM ile belge etkileşimi
- **ChromaDB/FAISS**: Vektör veritabanı
- **LLaMA**: Büyük dil modeli
- **Semantic Chunker**: Anlamsal metin parçalama

## Kurulum

```bash
# Depoyu klonlayın
git clone <repo-url>

# Dizine gidin
cd ai-belge-asistani

# Gerekli paketleri yükleyin
pip install -r requirements.txt

# Uygulamayı başlatın
streamlit run app.py
```

## Semantik Parçalama Ayarları

Arayüzde "Gelişmiş Ayarlar" bölümünden semantik parçalama özelliğini açıp kapatabilir ve şu parametreleri ayarlayabilirsiniz:

- **Parça Boyutu**: Oluşturulacak metin parçalarının maksimum boyutu
- **Parça Örtüşmesi**: Ardışık parçalar arasındaki örtüşme miktarı
- **Semantik Parçalama**: Anlamsal parçalama özelliğini açıp kapatma 