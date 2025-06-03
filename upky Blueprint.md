# Üretim Planlama Modelleri için Python Tabanlı Analiz Blueprint'i

## Ortak Parametreler
- `demand`: Aylık talep tahminleri `[1000, 1200, 1500, ...]` (adet).
- `working_days`: Aylık çalışma günleri `[22, 20, 23, ...]` (gün).
- `holding_cost`: 5 TL/ay (birim stok tutma maliyeti).
- `stockout_cost`: 20 TL/adet (birim stokta bulundurmama maliyeti).
- `outsourcing_cost`: 15 TL/adet (fason imalat maliyeti).
- `labor_per_unit`: 0.5 saat/adet (üretim başına işgücü).
- `hiring_cost`: 1000 TL/kişi (işçi alım maliyeti).
- `firing_cost`: 800 TL/kişi (işçi çıkarma maliyeti).
- `daily_hours`: 8 saat/gün (günlük çalışma süresi).
- `overtime_limit`: 20 saat/ay (fazla mesai limiti).
- `outsourcing_capacity`: 500 adet/ay (fason kapasite).

## Model 1: Karma Planlama Modeli
### Değişkenler
- `workers[t]`: t ayındaki işçi sayısı.
- `internal_production[t]`: t ayındaki iç üretim.
- `outsourced_production[t]`: t ayındaki fason üretim.
- `inventory[t]`: t ayı sonundaki stok.

### Sabit Parametreler
- `min_internal_ratio`: 0.70.
- `max_workforce_change`: ±5 kişi.
- `max_outsourcing_ratio`: 0.30.

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

### Sabit Parametreler
- `cost_supplier_A`: 15 TL/adet.
- `cost_supplier_B`: 18 TL/adet.
- `capacity_supplier_A`: 500 adet.

### Python Yaklaşımı
- Kütüphane: `PuLP`.
- Amaç: Maliyet minimizasyonu.

## Model 6: Mevsimsellik ve Talep Dalgaları Modeli
### Değişkenler
- `production[t]`: t ayındaki üretim.
- `inventory[t]`: t ayı sonundaki stok.
- `workers[t]`: t ayındaki işçi sayısı (opsiyonel).

### Sabit Parametreler
- `seasonal_demand`: Mevsimsel talep desenleri.

### Python Yaklaşımı
- Kütüphaneler: `statsmodels`, `Pandas`.
- Hesaplama: Zaman serisi analizi, üretim planlama.

## Toplam Maliyet Hesaplaması
- İşçilik, işçi değişim, stok, fason maliyetleri.

## Karar Destek Aracı
- Kütüphaneler: `Pandas`, `NumPy`, `PuLP`, `Matplotlib`.
- Özellikler: Parametre girişi, model çalıştırma, görselleştirme.