# Üretim Planlama Modelleri için Python Tabanlı Analiz Blueprint'i

## Ortak Parametreler 
### Kullanıcı Giriş Parametreleri
- `demand`: Aylık talep tahminleri `[1000, 1200, 1500, ...]` (adet)
- `working_days`: Aylık çalışma günleri `[22, 20, 23, ...]` (gün)
- `initial_inventory`: 0 adet (başlangıç stoğu)
- `safety_stock_ratio`: 0.05 (talep miktarı %5 kadar güvenlik stoğu)

### Maliyet Parametreleri
- `holding_cost`: 5 TL/ay (birim stok tutma maliyeti)
- `outsourcing_cost`: 75 TL/adet (ortalama fason imalat maliyeti)
- `stockout_cost`: 90 TL/adet (birim stokta bulundurmama maliyeti)
- `hiring_cost`: 1800 TL/kişi (işçi başına işe alım maliyeti)
- `firing_cost`: 2000 TL/kişi (işçi başına işten çıkarma maliyeti)
- `hourly_wage`: Saatlik işçi ücreti (TL/saat)
- `production_cost`: Birim üretim maliyeti (TL/adet)

### Üretim Parametreleri
- `labor_per_unit`: 0.5 saat/adet (üretim başına işgücü)
- `daily_hours`: Günlük çalışma saati

### Ortak Değişkenler
- `workers[t]`: t ayındaki işçi sayısı
- `production[t]`: t ayındaki üretim
- `inventory[t]`: t ayı sonundaki stok
- `stockout[t]`: t ayındaki karşılanmayan talep
- `hired[t]`: t ayındaki işe alınan işçi sayısı
- `fired[t]`: t ayındaki işten çıkarılan işçi sayısı
- `labor_cost[t]`: t ayındaki işçilik maliyeti
- `production_cost[t]`: t ayındaki üretim maliyeti
- `holding_cost[t]`: t ayındaki stok maliyeti
- `stockout_cost[t]`: t ayındaki stoksuzluk maliyeti

> Tüm modellerde işe alım/çıkarma maliyetleri dahil edilmiştir ve toplam maliyetlerin hesaplanmasında dikkate alınır.
> Ayrıca tüm modellerde güvenlik stoğu uygulanmıştır ve stokların hiçbir zaman güvenlik stok limitinin altına düşmemesi sağlanır.

## Model 1: Karma Planlama Modeli
### Model Özel Değişkenler
- `internal_production[t]`: t ayındaki iç üretim
- `outsourced_production[t]`: t ayındaki fason üretim
- `overtime_hours[t]`: t ayındaki fazla mesai saatleri

### Sabit Parametreler
- `min_internal_ratio`: 0.70 (toplam üretimin en az %70'i iç üretim olmalı)
- `max_workforce_change`: ±5 veya ±12 kişi (işgücü değişim sınırı)
- `max_outsourcing_ratio`: 0.30 (fason üretim oranı üst limiti)
- `min_workers`: Başlangıç minimum işçi sayısı
- `outsourcing_capacity`: Fason üretim kapasitesi (adet)
- `overtime_wage_multiplier`: Fazla mesai ücret çarpanı
- `max_overtime_per_worker`: İşçi başına maksimum fazla mesai saati/ay

### Python Yaklaşımı
- Kütüphane: `PuLP`
- Amaç: Toplam maliyeti minimize et
- Kısıtlar: İç üretim oranı, işgücü değişim sınırı, stok dengesi, fazla mesai ve fason kapasite limitleri, güvenlik stoğu limitleri
- Sonuçlar: Detaylı maliyet dökümü, birim maliyet analizi, tablo ve grafiksel çıktı
- Özellikler: İşçilik, üretim, stok, fason, işe alım/çıkarma ve fazla mesai maliyetlerinin ayrıntılı analizi ve görselleştirilmesi

## Model 2: Fazla Mesaili Üretim Modeli
### Model Özel Değişkenler
- `overtime_hours[t]`: t ayındaki toplam fazla mesai saatleri
- `normal_labor_cost[t]`: t ayındaki normal işçilik maliyeti
- `overtime_cost[t]`: t ayındaki fazla mesai maliyeti

### Sabit Parametreler
- `normal_hourly_wage`: Normal saatlik işçi ücreti (TL/saat)
- `overtime_wage_multiplier`: Fazla mesai ücret çarpanı (örn. 1.5)
- `max_overtime_per_worker`: İşçi başına maksimum fazla mesai (saat/ay)

### Python Yaklaşımı
- Kütüphaneler: `NumPy`, `Pandas`, `Matplotlib`, `tabulate`, `yaml`
- Amaç: Sabit işçi ve fazla mesaiyle talebin karşılanması, toplam maliyetin ve birim maliyetlerin ayrıntılı analizi
  - Kısıtlar: Normal kapasite ve fazla mesai kapasitesi limitleri, stok ve stoksuzluk yönetimi, Talep ve parametrelere göre optimum işçi sayısını otomatik hesaplama ve önerme, güvenlik stoğu limitleri
  - Sonuçlar: Detaylı maliyet tablosu, birim maliyet analizi, karşılanmayan talep, grafiksel çıktı (üretim, stok, fazla mesai)
  - Özellikler: Tüm maliyet kalemlerinin ayrıntılı dökümü, tablo ve grafiklerle görselleştirme

## Model 3: Toplu Üretim ve Stoklama Modeli
### Model Özel Değişkenler
- `real_inventory[t]`: t ayı sonundaki fiziksel stok seviyesi (her zaman güvenlik stoğu seviyesi veya üzeri)
- `unfilled[t]`: t ayında karşılanmayan talep

### Sabit Parametreler
- `worker_monthly_cost`: Aylık işçi maliyeti (TL/kişi/ay)
- `efficiency_factor`: Yüksek hacimli üretimlerde birim işgücü ihtiyacını azaltan verimlilik faktörü
- `scale_threshold`: Verimlilik iyileştirmesinin başladığı üretim hacmi eşiği

### Python Yaklaşımı
- Kütüphaneler: `NumPy`, `Pandas`, `Matplotlib`, `tabulate`, `yaml`
- Amaç: Sabit işçiyle toplu üretim ve stoklama, toplam maliyetin ve birim maliyetlerin ayrıntılı analizi
- Kısıtlar: Sabit işçi kapasitesi, stok ve stoksuzluk yönetimi, güvenlik stoğu limitleri
- Sonuçlar: Detaylı maliyet tablosu, birim maliyet analizi, karşılanmayan talep, grafiksel çıktı (üretim, stok)
- Özellikler: Fiziksel stok seviyeleri güvenlik stoğu sınırının altına düşmez, karşılanmayan talep ayrı olarak gösterilir, işe alım maliyeti toplam maliyete dahildir
- Verimlilik faktörü hesaplaması: Büyük hacimli üretimlerde verimlilik artışı
- İşçi sayısı optimizasyonu: optimum işçi sayısı %10 aralığında tutulur

## Model 4: Dinamik Programlama Tabanlı Model
### Sabit Parametreler
- `max_workers`: Maksimum işçi sayısı
- `max_workforce_change`: Maksimum işgücü değişimi (kişi/ay)

### Python Yaklaşımı
- Kütüphaneler: `NumPy`, `Pandas`, `Matplotlib`, `tabulate`, `yaml`
- Amaç: Dinamik programlama ile toplam maliyeti minimize etmek, işgücü değişimi ve stok yönetimini optimize etmek
- Kısıtlar: İşgücü, üretim, stok ve stoksuzluk denge denklemleri, işçi değişim sınırları, güvenlik stoğu limitleri
- Sonuçlar: Detaylı maliyet tablosu, birim maliyet analizi, karşılanmayan talep, grafiksal çıktı (işçi, üretim, stok, maliyetler)
-  Detay: İlk dönemde 0 işçiyle başlar ve optimum sayıda işçiyi işe alır, kapasiteye değil, gerçek üretim hacmine göre üretim maliyetlerini hesaplar
- Özellikler: Model artık talebe göre optimum işçi sayısını belirler ve ihtiyaç duyulan kadar işçiyi işe alır, güvenlik stok seviyelerini korur

## Model 5: Dış Kaynak Kullanımı Modelleri Karşılaştırması
### Model Özel Değişkenler
- `outsourced_supplier_A[t]`: t ayındaki Tedarikçi A'dan alınan üretim
- `outsourced_supplier_B[t]`: t ayındaki Tedarikçi B'den alınan üretim
- `cost_A[t]`: t ayındaki Tedarikçi A maliyeti
- `cost_B[t]`: t ayındaki Tedarikçi B maliyeti
- `total_cost[t]`: t ayındaki toplam maliyet

### Sabit Parametreler
- `cost_supplier_A`: Tedarikçi A birim maliyeti (TL/adet)
- `cost_supplier_B`: Tedarikçi B birim maliyeti (TL/adet)
- `capacity_supplier_A`: Tedarikçi A aylık kapasitesi (adet)
- `capacity_supplier_B`: Tedarikçi B aylık kapasitesi (adet)

### Python Yaklaşımı
- Kütüphaneler: `PuLP`, `NumPy`, `Pandas`, `Matplotlib`, `tabulate`, `yaml`
- Amaç: Tedarikçi seçimi ve kapasite kısıtları altında toplam maliyeti minimize etmek, karşılanamayan talebi ve stok seviyesini optimize etmek
- Kısıtlar: Tedarikçi kapasite limitleri, stok ve stoksuzluk denge denklemleri, güvenlik stoğu sınırları
- Sonuçlar: Detaylı maliyet tablosu, birim maliyet analizi, karşılanmayan talep, grafiksel çıktı (tedarikçi kullanımı, stok, karşılanmayan talep, kapasite limitleri)
- Özellikler: Tüm maliyet kalemlerinin ayrıntılı dökümü, tablo ve grafiklerle görselleştirme, tedarikçi bazında analiz

## Model 6: Mevsimsellik ve Talep Dalgaları Modeli
### Sabit Parametreler
- `max_workers`: Maksimum işçi sayısı
- `max_workforce_change`: Maksimum işgücü değişimi (kişi/ay)

### Python Yaklaşımı
- Kütüphaneler: `PuLP`, `NumPy`, `Pandas`, `Matplotlib`, `tabulate`, `yaml`
- Amaç: Mevsimsel talep dalgalanmalarında üretim, stok ve stoksuzluk maliyetlerinin toplamını minimize etmek
- Kısıtlar: Üretim kapasitesi, stok ve stoksuzluk denge denklemleri, işçi sayısı değişim kısıtları, güvenlik stoğu limitleri
- Sonuçlar: Detaylı maliyet tablosu, birim maliyet analizi, karşılanmayan talep, grafiksal çıktı (üretim, stok, stoksuzluk, maliyetler)
- Özellikler: Model sıfır işçiyle başlar ve talebe göre işçileri işe alır, işe alım maliyetleri toplam maliyete dahil edilir, güvenlik stok seviyeleri korunur

## Toplam Maliyet Hesaplaması ve Maliyet Analizi
Her model için aşağıdaki toplam maliyet kalemleri hesaplanır:
- İşçilik maliyeti (normal ve fazla mesai)
- Üretim maliyeti
- Stok maliyeti
- Stoksuzluk maliyeti
- İşe alım/çıkarma maliyetleri
- Fason üretim maliyeti (uygulanabilirse)

Birim maliyet analizi, tüm modeller için standartlaştırılmış karşılaştırma sağlar:
- Toplam talep ve karşılanma oranı
- Ortalama birim maliyet
- Maliyet kategorilerine göre birim maliyet dağılımı
- Stoksuzluk oranı ve maliyeti
- İşe alım/çıkarma birim maliyeti

## Streamlit Arayüzü ve Karar Destek Paneli
### Genel Özellikler
- Kullanıcıdan model parametrelerini (talep, işçilik, stok, kapasite vb.) girdi olarak alır
- Tüm modelleri tek tıkla çalıştırır ve sonuçları karşılaştırmalı olarak özet ve detaylı tablolar halinde sunar
- Sonuçları tablo ve grafiklerle (üretim, stok, maliyet, stoksuzluk vb.) görselleştirir
- Model fonksiyonları ayrı dosyalarda, arayüzde fonksiyon olarak çağrılır
- Hataları kullanıcıya açıkça bildirir
- Parametreleri yaml dosyasından yükler ve kullanıcı arayüzünden değiştirilebilir
- İşe alım maliyetleri dahil tüm maliyet kalemlerini hesaplar ve gösterir
- Güvenlik stok parametreleri kullanıcı tarafından ayarlanabilir

### Bellek Yönetimi ve Performans Optimizasyonu
- **Cache Yönetimi**: Model solver fonksiyonları `@st.cache_data` ile cache'lenir (TTL: 5 dakika, max 3 entry)
- **Bellek Temizliği**: Her model çalıştıktan sonra otomatik garbage collection ve bellek temizliği
- **Memory Monitoring**: Opsiyonel bellek kullanım izleme (psutil ile)
- **Progress Tracking**: Uzun işlemler için progress bar ve durum göstergesi
- **Cache Control**: Kullanıcı manuel cache temizleme özelliği
- **Lifecycle Management**: Büyük veri yapılarının yaşam döngüsü yönetimi
- **Error Recovery**: Hata durumlarında da bellek temizliği yapılır

### Temel Akış
1. **Parametre Girişi:**
   - Yan panelde (sidebar) kullanıcıdan talep, işçilik, stok, kapasite gibi parametreler alınır
   - Varsayılan değerler ve örnek veri yüklenir
   - Güvenlik stoğu oranı ayarlanabilir (varsayılan talep miktarının %5'i)
2. **Model Seçimi ve Çalıştırma:**
   - Kullanıcı tek model veya tüm modelleri seçip çalıştırabilir
   - Her modelin fonksiyonu çağrılır, sonuçlar toplanır
3. **Özet Karşılaştırma Tablosu:**
   - Toplam maliyet, işçilik maliyeti, toplam üretim, stoksuzluk oranı, işgücü esnekliği gibi ana metrikler tablo halinde sunulur
   - Sayısal değerler okunaklı formatlanır
4. **Detaylı Sonuçlar:**
   - Her modelin detaylı çıktısı (aylık üretim, stok, maliyet, stoksuzluk vb.) ayrı tabloda gösterilir
   - Hata oluşursa kullanıcıya bildirilir
5. **Görselleştirme:**
   - Üretim, stok, maliyet, stoksuzluk gibi metrikler için çizgi ve çubuk grafikler sunulur
   - Model bazında ve karşılaştırmalı grafikler desteklenir

### Teknik Notlar
- Ana dosya: `streamlit_app.py`
- Model fonksiyonları: `modelX_*.py` dosyalarında
- Parametre yönetimi: `parametreler.yaml` dosyasında
- Kullanılan kütüphaneler: `streamlit`, `pandas`, `numpy`, `matplotlib`, `tabulate`, `PuLP`, `yaml`, `gc`, `psutil`
- Kodda hata yönetimi ve kullanıcıya açıklama ön planda tutulur
- Bellek optimizasyonu için cache yönetimi ve garbage collection uygulanır
- Tüm modellerin çıktıları ortak formatta toplanır ve karşılaştırılır
- Her model için ayrı bir `maliyet_analizi` fonksiyonu uygulanmıştır ve tüm modelleri aynı standart formatta çıktı üretir
- Talep tipine göre (normal, yüksek, aşırı yüksek, mevsimsel) optimum işçi sayısı önerisi uygulanmıştır
- Güvenlik stoğu her modelde talep miktarının belirli bir yüzdesi olarak uygulanır, böylece stoksuzluk riski azaltılır
