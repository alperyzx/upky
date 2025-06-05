# Üretim Planlama Modelleri için Python Tabanlı Analiz Blueprint'i

## Ortak Parametreler (Karşılaştırma Tablosu için Kullanıcı Girişi)
- `demand`: Aylık talep tahminleri `[1000, 1200, 1500, ...]` (adet).
- `working_days`: Aylık çalışma günleri `[22, 20, 23, ...]` (gün).
- `holding_cost`: 5 TL/ay (birim stok tutma maliyeti).
- `outsourcing_cost`: 15 TL/adet (fason imalat maliyeti).
- `labor_per_unit`: 0.5 saat/adet (üretim başına işgücü).
- `stockout_cost`: 20 TL/adet (birim stokta bulundurmama maliyeti).

> Diğer parametreler modellerde sabit olarak tanımlıdır ve karşılaştırma tablosunda kullanıcı tarafından değiştirilemez.

## Model 1: Karma Planlama Modeli
### Değişkenler
- `workers[t]`: t ayındaki işçi sayısı.
- `internal_production[t]`: t ayındaki iç üretim.
- `outsourced_production[t]`: t ayındaki fason üretim.
- `inventory[t]`: t ayı sonundaki stok.
- `stockout[t]`: t ayındaki karşılanmayan talep (stokta bulundurmama miktarı).

### Sabit Parametreler
- `min_internal_ratio`: 0.70.
- `max_workforce_change`: ±5 kişi.
- `max_outsourcing_ratio`: 0.30.
- `min_workers`: Başlangıç minimum işçi sayısı.

### Python Yaklaşımı
- Kütüphane: `PuLP`.
- Amaç: Toplam maliyeti minimize et.
- Kısıtlar: İç üretim oranı, işgücü değişim sınırı, stok dengesi.

## Model 2: Fazla Mesaili Üretim Modeli
### Değişkenler
- `overtime_hours[t]`: t ayındaki fazla mesai saatleri.
- `production[t]`: t ayındaki üretim.
- `inventory[t]`: t ayı sonundaki stok.

### Sabit Parametreler
- `fixed_workers`: 50 kişi.
- `overtime_wage_multiplier`: 1.5.
- `max_overtime_per_worker`: 20 saat.

### Python Yaklaşımı
- Kütüphaneler: `NumPy`, `Pandas`.
- Hesaplama: Normal ve fazla mesai kapasitesi, stok dengesi.

## Model 3: Toplu Üretim ve Stoklama Modeli
### Değişkenler
- `production[t]`: t ayındaki üretim.
- `inventory[t]`: t ayı sonundaki stok.

### Sabit Parametreler
- `fixed_workers`: 50 kişi.
- `production_rate`: 2 adet/saat.

### Python Yaklaşımı
- Kütüphane: `Pandas`.
- Hesaplama: Sabit üretim, stok seviyesi.

## Model 4: Dinamik Programlama Tabanlı Model
### Değişkenler
- `workers[t]`: t ayındaki işçi sayısı.
- `production[t]`: t ayındaki üretim.
- `inventory[t]`: t ayı sonundaki stok.

### Sabit Parametreler
- `transition_costs`: İşçi değişim maliyetleri.
- `holding_cost`, `stockout_cost`.

### Python Yaklaşımı
- Kütüphane: `NumPy`.
- Modelleme: Dinamik programlama.

## Model 5: Dış Kaynak Kullanımı Modelleri Karşılaştırması
### Değişkenler
- `outsourced_supplier_A[t]`: Tedarikçi A’dan üretim.
- `outsourced_supplier_B[t]`: Tedarikçi B’den üretim.
- `internal_production[t]`: İç üretim.
- `inventory[t]`: Stok seviyesi.
- `stockout[t]`: Karşılanmayan talep (stokta bulundurmama).

### Sabit Parametreler
- `cost_supplier_A`: 15 TL/adet.
- `cost_supplier_B`: 18 TL/adet.
- `capacity_supplier_A`: 500 adet.
- `capacity_supplier_B`: 500 adet.
- `stockout_cost`: 20 TL/adet.

### Python Yaklaşımı
- Kütüphane: `Pandas`, `PuLP`.
- Modelleme: Lineer programlama ile tedarikçi ve iç üretim optimizasyonu.

## Model 6: Mevsimsellik ve Talep Dalgaları Modeli
### Değişkenler
- `production[t]`: t ayındaki üretim.
- `inventory[t]`: t ayı sonundaki stok.
- `stockout[t]`: t ayındaki karşılanmayan talep (stoksuzluk).

### Sabit Parametreler
- `seasonal_demand`: Mevsimsel talep desenleri.
- `holding_cost`: Stok maliyeti.
- `stockout_cost`: Stoksuzluk maliyeti.
- `production_cost`: Üretim maliyeti.
- `max_production`: Maksimum aylık üretim kapasitesi.

### Python Yaklaşımı
- Kütüphane: `PuLP`.
- Amaç: Üretim, stok ve stoksuzluk maliyetlerinin toplamını minimize et.
- Kısıtlar: Üretim kapasitesi, stok ve stoksuzluk denklemi.
- Model, düşük talep aylarında fazla üretim yapıp stokta tutarak yüksek talep aylarını karşılamaya çalışır.

## Toplam Maliyet Hesaplaması
- İşçilik, işçi değişim, stok, fason maliyetleri.

## Karar Destek Aracı
- Kütüphaneler: `Pandas`, `NumPy`, `PuLP`, `Matplotlib`.
- Özellikler: Parametre girişi, model çalıştırma, görselleştirme.
