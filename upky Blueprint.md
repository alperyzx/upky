# Üretim Planlama Modelleri için Python Tabanlı Analiz Blueprint'i

## Ortak Parametreler (Karşılaştırma Tablosu için Kullanıcı Girişi)
- `demand`: Aylık talep tahminleri `[1000, 1200, 1500, ...]` (adet).
- `working_days`: Aylık çalışma günleri `[22, 20, 23, ...]` (gün).
- `holding_cost`: 5 TL/ay (birim stok tutma maliyeti).
- `outsourcing_cost`: 85 TL/adet (ortalama fason imalat maliyeti).
- `labor_per_unit`: 0.5 saat/adet (üretim başına işgücü).
- `stockout_cost`: 20 TL/adet (birim stokta bulundurmama maliyeti).
- `hiring_cost`: 1800 TL/kişi (işçi başına işe alım maliyeti).
- `firing_cost`: 1500 TL/kişi (işçi başına işten çıkarma maliyeti).
- `initial_inventory`: 0 adet (başlangıç stoğu).
- `safety_stock_ratio`: 0.03 (talep miktarının %3'ü kadar güvenlik stoğu).

> Tüm modellerde işe alım/çıkarma maliyetleri dahil edilmiştir ve toplam maliyetlerin hesaplanmasında dikkate alınır.
> Ayrıca tüm modellerde güvenlik stoğu uygulanmıştır ve stokların hiçbir zaman güvenlik stok limitinin altına düşmemesi sağlanır.

## Model 1: Karma Planlama Modeli
### Değişkenler
- `workers[t]`: t ayındaki işçi sayısı.
- `internal_production[t]`: t ayındaki iç üretim.
- `outsourced_production[t]`: t ayındaki fason üretim.
- `inventory[t]`: t ayı sonundaki stok.
- `stockout[t]`: t ayındaki karşılanmayan talep (stokta bulundurmama miktarı).
- `overtime_hours[t]`: t ayındaki fazla mesai saatleri.
- `hired[t]`: t ayındaki işe alınan işçi sayısı.
- `fired[t]`: t ayındaki işten çıkarılan işçi sayısı.

### Sabit Parametreler
- `min_internal_ratio`: 0.70 (toplam üretimin en az %70'i iç üretim olmalı).
- `max_workforce_change`: ±5 veya ±12 kişi (işgücü değişim sınırı).
- `max_outsourcing_ratio`: 0.30 (fason üretim oranı üst limiti).
- `min_workers`: Başlangıç minimum işçi sayısı.
- `outsourcing_capacity`: Fason üretim kapasitesi (adet).
- `hiring_cost`: İşe alım maliyeti (TL/kişi).
- `firing_cost`: İşten çıkarma maliyeti (TL/kişi).
- `hourly_wage`: Saatlik işçi ücreti (TL/saat).
- `production_cost`: İç üretim birim maliyeti (TL/adet).
- `daily_hours`: Günlük çalışma saati.
- `overtime_wage_multiplier`: Fazla mesai ücret çarpanı.
- `max_overtime_per_worker`: İşçi başına maksimum fazla mesai saati/ay.
- `initial_inventory`: Başlangıç stok miktarı.
- `safety_stock_ratio`: Talep miktarının belirli bir yüzdesi olarak güvenlik stoğu.

### Python Yaklaşımı
- Kütüphane: `PuLP`.
- Amaç: Toplam maliyeti minimize et.
- Kısıtlar: İç üretim oranı, işgücü değişim sınırı, stok dengesi, fazla mesai ve fason kapasite limitleri, güvenlik stoğu limitleri.
- Sonuçlar: Detaylı maliyet dökümü, birim maliyet analizi, tablo ve grafiksel çıktı.
- Özellikler: İşçilik, üretim, stok, fason, işe alım/çıkarma ve fazla mesai maliyetlerinin ayrıntılı analizi ve görselleştirilmesi.

## Model 2: Fazla Mesaili Üretim Modeli
### Değişkenler
- `production[t]`: t ayındaki toplam üretim.
- `overtime_hours[t]`: t ayındaki toplam fazla mesai saatleri.
- `inventory[t]`: t ayı sonundaki stok.
- `holding[t]`: t ayındaki stok maliyeti.
- `stockout[t]`: t ayındaki stoksuzluk maliyeti.
- `normal_labor_cost[t]`: t ayındaki normal işçilik maliyeti.
- `overtime_cost[t]`: t ayındaki fazla mesai maliyeti.
- `production_cost[t]`: t ayındaki üretim maliyeti.

### Sabit Parametreler
- `fixed_workers`: Sabit işçi sayısı (örn. 18 kişi).
- `labor_per_unit`: Birim başına işçilik süresi (saat/adet).
- `daily_hours`: Günlük çalışma saati.
- `working_days[t]`: t ayındaki toplam çalışma günü.
- `normal_hourly_wage`: Normal saatlik işçi ücreti (TL/saat).
- `overtime_wage_multiplier`: Fazla mesai ücret çarpanı (örn. 1.5).
- `max_overtime_per_worker`: İşçi başına maksimum fazla mesai (saat/ay).
- `production_cost`: Birim üretim maliyeti (TL/adet).
- `holding_cost`: Stok tutma maliyeti (TL/adet/ay).
- `stockout_cost`: Stoksuzluk maliyeti (TL/adet/ay).
- `hiring_cost`: İşe alım maliyeti (TL/kişi).
- `initial_inventory`: Başlangıç stok miktarı.
- `safety_stock_ratio`: Talep miktarının belirli bir yüzdesi olarak güvenlik stoğu.

### Python Yaklaşımı
- Kütüphaneler: `NumPy`, `Pandas`, `Matplotlib`, `tabulate`, `yaml`.
- Amaç: Sabit işçi ve fazla mesaiyle talebin karşılanması, toplam maliyetin ve birim maliyetlerin ayrıntılı analizi.
  - Kısıtlar: Normal kapasite ve fazla mesai kapasitesi limitleri, stok ve stoksuzluk yönetimi, Talep ve parametrelere göre optimum işçi sayısını otomatik hesaplama ve önerme.
  - Sonuçlar: Detaylı maliyet tablosu, birim maliyet analizi, karşılanmayan talep, grafiksel çıktı (üretim, stok, fazla mesai).
  - Özellikler: Tüm maliyet kalemlerinin ayrıntılı dökümü, tablo ve grafiklerle görselleştirme.

## Model 3: Toplu Üretim ve Stoklama Modeli
### Değişkenler
- `production[t]`: t ayındaki üretim.
- `inventory[t]`: t ayı sonundaki stok (artık hiçbir zaman negatif değil).
- `real_inventory[t]`: t ayı sonundaki fiziksel stok seviyesi (her zaman 0 veya pozitif).
- `holding[t]`: t ayındaki stok maliyeti.
- `stockout[t]`: t ayındaki stoksuzluk maliyeti.
- `labor_cost[t]`: t ayındaki işçilik maliyeti.
- `production_cost[t]`: t ayındaki üretim maliyeti.
- `unfilled[t]`: t ayında karşılanmayan talep.

### Sabit Parametreler
- `fixed_workers`: Sabit işçi sayısı (örn. 18 kişi).
- `labor_per_unit`: Birim başına işçilik süresi (saat/adet).
- `daily_hours`: Günlük çalışma saati.
- `working_days[t]`: t ayındaki toplam çalışma günü.
- `hourly_wage`: Saatlik işçi ücreti (TL/saat).
- `production_cost`: Birim üretim maliyeti (TL/adet).
- `holding_cost`: Stok tutma maliyeti (TL/adet/ay).
- `stockout_cost`: Stoksuzluk maliyeti (TL/adet/ay).
- `worker_monthly_cost`: Aylık işçi maliyeti (TL/kişi/ay).
- `hiring_cost`: İşe alım maliyeti (TL/kişi).
- `efficiency_factor`: Yüksek hacimli üretimlerde birim işgücü ihtiyacını azaltan verimlilik faktörü.
- `scale_threshold`: Verimlilik iyileştirmesinin başladığı üretim hacmi eşiği.
- `initial_inventory`: Başlangıç stok miktarı.
- `safety_stock_ratio`: Talep miktarının belirli bir yüzdesi olarak güvenlik stoğu.

### Python Yaklaşımı
- Kütüphaneler: `NumPy`, `Pandas`, `Matplotlib`, `tabulate`, `yaml`.
- Amaç: Sabit işçiyle toplu üretim ve stoklama, toplam maliyetin ve birim maliyetlerin ayrıntılı analizi.
- Kısıtlar: Sabit işçi kapasitesi, stok ve stoksuzluk yönetimi.
- Sonuçlar: Detaylı maliyet tablosu, birim maliyet analizi, karşılanmayan talep, grafiksel çıktı (üretim, stok).
- Özellikler: Fiziksel stok seviyeleri negatif olmaz, karşılanmayan talep ayrı olarak gösterilir, işe alım maliyeti toplam maliyete dahildir.
- Verimlilik faktörü hesaplaması: Büyük hacimli üretimlerde verimlilik artışı.
- İşçi sayısı optimizasyonu: optimum işçi sayısının %10 aralığında tutulur.

## Model 4: Dinamik Programlama Tabanlı Model
### Değişkenler
- `workers[t]`: t ayındaki işçi sayısı.
- `production[t]`: t ayındaki üretim.
- `inventory[t]`: t ayı sonundaki stok.
- `stockout[t]`: t ayındaki karşılanmayan talep (stoksuzluk).
- `hired[t]`: t ayındaki işe alınan işçi sayısı.
- `fired[t]`: t ayındaki işten çıkarılan işçi sayısı.
- `labor_cost[t]`: t ayındaki işçilik maliyeti.
- `production_cost[t]`: t ayındaki üretim maliyeti.
- `holding_cost[t]`: t ayındaki stok maliyeti.
- `stockout_cost[t]`: t ayındaki stoksuzluk maliyeti.
- `hiring_cost[t]`: t ayındaki işe alım maliyeti.
- `firing_cost[t]`: t ayındaki işten çıkarma maliyeti.

### Sabit Parametreler
- `labor_per_unit`: Birim başına işçilik süresi (saat/adet).
- `hourly_wage`: Saatlik işçi ücreti (TL/saat).
- `production_cost`: Birim üretim maliyeti (TL/adet).
- `holding_cost`: Stok tutma maliyeti (TL/adet/ay).
- `stockout_cost`: Stoksuzluk maliyeti (TL/adet/ay).
- `hiring_cost`: İşe alım maliyeti (TL/kişi).
- `firing_cost`: İşten çıkarma maliyeti (TL/kişi).
- `daily_hours`: Günlük çalışma saati.
- `working_days[t]`: t ayındaki toplam çalışma günü.
- `max_workers`: Maksimum işçi sayısı.
- `max_workforce_change`: Maksimum işgücü değişimi (kişi/ay).
- `initial_inventory`: Başlangıç stok miktarı.
- `safety_stock_ratio`: Talep miktarının belirli bir yüzdesi olarak güvenlik stoğu.

### Python Yaklaşımı
- Kütüphaneler: `NumPy`, `Pandas`, `Matplotlib`, `tabulate`, `yaml`.
- Amaç: Dinamik programlama ile toplam maliyeti minimize etmek, işgücü değişimi ve stok yönetimini optimize etmek.
- Kısıtlar: İşgücü, üretim, stok ve stoksuzluk denge denklemleri, işçi değişim sınırları, güvenlik stoğu limitleri.
- Sonuçlar: Detaylı maliyet tablosu, birim maliyet analizi, karşılanmayan talep, grafiksal çıktı (işçi, üretim, stok, maliyetler).
-  Detay:  İlk dönemde 0 işçiyle başlar ve optimum sayıda işçiyi işe alır, kapasiteye değil, gerçek üretim hacmine göre üretim maliyetlerini hesaplar.
- Özellikler: Model artık talebe göre optimum işçi sayısını belirler ve ihtiyaç duyulan kadar işçiyi işe alır, güvenlik stok seviyelerini korur.

## Model 5: Dış Kaynak Kullanımı Modelleri Karşılaştırması
### Değişkenler
- `outsourced_supplier_A[t]`: t ayındaki Tedarikçi A'dan alınan üretim.
- `outsourced_supplier_B[t]`: t ayındaki Tedarikçi B'den alınan üretim.
- `inventory[t]`: t ayı sonundaki stok.
- `stockout[t]`: t ayındaki karşılanmayan talep (stokta bulundurmama).
- `cost_A[t]`: t ayındaki Tedarikçi A maliyeti.
- `cost_B[t]`: t ayındaki Tedarikçi B maliyeti.
- `holding_cost[t]`: t ayındaki stok maliyeti.
- `stockout_cost[t]`: t ayındaki stoksuzluk maliyeti.
- `total_cost[t]`: t ayındaki toplam maliyet.

### Sabit Parametreler
- `cost_supplier_A`: Tedarikçi A birim maliyeti (TL/adet).
- `cost_supplier_B`: Tedarikçi B birim maliyeti (TL/adet).
- `capacity_supplier_A`: Tedarikçi A aylık kapasitesi (adet).
- `capacity_supplier_B`: Tedarikçi B aylık kapasitesi (adet).
- `holding_cost`: Stok tutma maliyeti (TL/adet/ay).
- `stockout_cost`: Stoksuzluk maliyeti (TL/adet/ay).
- `initial_inventory`: Başlangıç stok miktarı.
- `safety_stock_ratio`: Talep miktarının belirli bir yüzdesi olarak güvenlik stoğu.

### Python Yaklaşımı
- Kütüphaneler: `PuLP`, `NumPy`, `Pandas`, `Matplotlib`, `tabulate`, `yaml`.
- Amaç: Tedarikçi seçimi ve kapasite kısıtları altında toplam maliyeti minimize etmek, karşılanamayan talebi ve stok seviyesini optimize etmek.
- Kısıtlar: Tedarikçi kapasite limitleri, stok ve stoksuzluk denge denklemleri.
- Sonuçlar: Detaylı maliyet tablosu, birim maliyet analizi, karşılanmayan talep, grafiksel çıktı (tedarikçi kullanımı, stok, karşılanmayan talep, kapasite limitleri).
- Özellikler: Tüm maliyet kalemlerinin ayrıntılı dökümü, tablo ve grafiklerle görselleştirme, tedarikçi bazında analiz.

## Model 6: Mevsimsellik ve Talep Dalgaları Modeli
### Değişkenler
- `production[t]`: t ayındaki üretim.
- `inventory[t]`: t ayı sonundaki stok.
- `stockout[t]`: t ayındaki karşılanmayan talep (stoksuzluk).
- `holding_cost[t]`: t ayındaki stok maliyeti.
- `stockout_cost[t]`: t ayındaki stoksuzluk maliyeti.
- `production_cost[t]`: t ayındaki üretim maliyeti.
- `labor_cost[t]`: t ayındaki işçilik maliyeti.
- `total_cost[t]`: t ayındaki toplam maliyet.
- `workers[t]`: t ayındaki işçi sayısı.
- `hire[t]`: t ayındaki işe alınan işçi sayısı.
- `fire[t]`: t ayındaki işten çıkarılan işçi sayısı.
- `hiring_cost[t]`: t ayındaki işe alım maliyeti.
- `firing_cost[t]`: t ayındaki işten çıkarma maliyeti.

### Sabit Parametreler
- `demand`: Mevsimsel talep desenleri (12 aylık dizi).
- `working_days[t]`: t ayındaki toplam çalışma günü.
- `holding_cost`: Stok maliyeti (TL/adet/ay).
- `stockout_cost`: Stoksuzluk maliyeti (TL/adet/ay).
- `production_cost`: Üretim maliyeti (TL/adet).
- `labor_per_unit`: Birim başına işçilik süresi (saat/adet).
- `hourly_wage`: Saatlik işçi ücreti (TL/saat).
- `daily_hours`: Günlük çalışma saati.
- `hiring_cost`: İşe alım maliyeti (TL/kişi).
- `firing_cost`: İşten çıkarma maliyeti (TL/kişi).
- `max_workers`: Maksimum işçi sayısı.
- `max_workforce_change`: Maksimum işgücü değişimi (kişi/ay).
- `initial_inventory`: Başlangıç stok miktarı.
- `safety_stock_ratio`: Talep miktarının belirli bir yüzdesi olarak güvenlik stoğu.

### Python Yaklaşımı
- Kütüphaneler: `PuLP`, `NumPy`, `Pandas`, `Matplotlib`, `tabulate`, `yaml`.
- Amaç: Mevsimsel talep dalgalanmalarında üretim, stok ve stoksuzluk maliyetlerinin toplamını minimize etmek.
- Kısıtlar: Üretim kapasitesi, stok ve stoksuzluk denge denklemleri, işçi sayısı değişim kısıtları, güvenlik stoğu limitleri.
- Sonuçlar: Detaylı maliyet tablosu, birim maliyet analizi, karşılanmayan talep, grafiksal çıktı (üretim, stok, stoksuzluk, maliyetler).
- Özellikler: Model sıfır işçiyle başlar ve talebe göre işçileri işe alır, işe alım maliyetleri toplam maliyete dahil edilir, güvenlik stok seviyeleri korunur.

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
- Kullanıcıdan model parametrelerini (talep, işçilik, stok, kapasite vb.) girdi olarak alır.
- Tüm modelleri tek tıkla çalıştırır ve sonuçları karşılaştırmalı olarak özet ve detaylı tablolar halinde sunar.
- Sonuçları tablo ve grafiklerle (üretim, stok, maliyet, stoksuzluk vb.) görselleştirir.
- Model fonksiyonları ayrı dosyalarda, arayüzde fonksiyon olarak çağrılır.
- Hataları kullanıcıya açıkça bildirir.
- Parametreleri yaml dosyasından yükler ve kullanıcı arayüzünden değiştirilebilir.
- İşe alım maliyetleri dahil tüm maliyet kalemlerini hesaplar ve gösterir.
- Güvenlik stok parametreleri kullanıcı tarafından ayarlanabilir.

### Temel Akış
1. **Parametre Girişi:**
   - Yan panelde (sidebar) kullanıcıdan talep, işçilik, stok, kapasite gibi parametreler alınır.
   - Varsayılan değerler ve örnek veri yüklenir.
   - Güvenlik stoğu oranı ayarlanabilir (varsayılan talep miktarının %5'i).
2. **Model Seçimi ve Çalıştırma:**
   - Kullanıcı tek model veya tüm modelleri seçip çalıştırabilir.
   - Her modelin fonksiyonu çağrılır, sonuçlar toplanır.
3. **Özet Karşılaştırma Tablosu:**
   - Toplam maliyet, işçilik maliyeti, toplam üretim, stoksuzluk oranı, işgücü esnekliği gibi ana metrikler tablo halinde sunulur.
   - Sayısal değerler okunaklı formatlanır.
4. **Detaylı Sonuçlar:**
   - Her modelin detaylı çıktısı (aylık üretim, stok, maliyet, stoksuzluk vb.) ayrı tabloda gösterilir.
   - Hata oluşursa kullanıcıya bildirilir.
5. **Görselleştirme:**
   - Üretim, stok, maliyet, stoksuzluk gibi metrikler için çizgi ve çubuk grafikler sunulur.
   - Model bazında ve karşılaştırmalı grafikler desteklenir.

### Teknik Notlar
- Ana dosya: `streamlit_app.py`
- Model fonksiyonları: `modelX_*.py` dosyalarında.
- Parametre yönetimi: `parametreler.yaml` dosyasında.
- Kullanılan kütüphaneler: `streamlit`, `pandas`, `numpy`, `matplotlib`, `tabulate`, `PuLP`, `yaml`.
- Kodda hata yönetimi ve kullanıcıya açıklama ön planda tutulur.
- Tüm modellerin çıktıları ortak formatta toplanır ve karşılaştırılır.
- Her model için ayrı bir `maliyet_analizi` fonksiyonu uygulanmıştır ve tüm modelleri aynı standart formatta çıktı üretir.
- Talep tipine göre (normal, yüksek, aşırı yüksek, mevsimsel) optimum işçi sayısı önerisi uygulanmıştır.
- Güvenlik stoğu her modelde talep miktarının belirli bir yüzdesi olarak uygulanır, böylece stoksuzluk riski azaltılır.
