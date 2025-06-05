import numpy as np
import pandas as pd
from tabulate import tabulate

# Parametreler
# Çok yüksek talep için örnek:
demand = np.array([300, 230, 200, 200, 500, 200, 400, 180, 100, 100, 120, 150])
working_days = np.array([22, 20, 23, 19, 21, 19, 22, 22, 22, 21, 21, 21])
holding_cost = 5
stockout_cost = 20
hiring_cost = 1800
firing_cost = 1500
daily_hours = 8
labor_per_unit = 4 # 1 birim üretim için gereken işgücü (saat)
max_workers = 100  # Daha yüksek üst sınır
min_workers = 12
max_workforce_change = 3  # Daha hızlı işçi artışı
months = len(demand)
hourly_wage = 10
production_cost = 30  # birim üretim maliyeti (TL)

# Check that demand and working_days have the same length
if len(demand) != len(working_days):
    raise ValueError(f"Length of demand ({len(demand)}) and working_days ({len(working_days)}) must be equal.")

# Üretim kapasitesi: işçi * gün * saat / birim işgücü
def prod_capacity(workers, t):
    return workers * working_days[t] * daily_hours / labor_per_unit

# DP tabloları
cost_table = np.full((months+1, max_workers+1), np.inf)
backtrack = np.full((months+1, max_workers+1), -1, dtype=int)

# Başlangıç: 0. ayda 0 stok ve her işçi sayısı için maliyet 0
cost_table[0, min_workers:max_workers+1] = 0

# DP algoritması
for t in range(months):
    for prev_w in range(min_workers, max_workers+1):
        if cost_table[t, prev_w] < np.inf:
            for w in range(max(min_workers, prev_w-max_workforce_change), min(max_workers, prev_w+max_workforce_change)+1):
                capacity = prod_capacity(w, t)
                # Karşılanamayan talep varsa ceza maliyeti uygula
                unmet = max(0, demand[t] - capacity)
                inventory = max(0, capacity - demand[t])
                hire = max(0, w - prev_w) * hiring_cost
                fire = max(0, prev_w - w) * firing_cost
                holding = inventory * holding_cost
                stockout = unmet * stockout_cost
                labor = w * working_days[t] * daily_hours * hourly_wage

                # Kapasiteye değil, gerçek üretim miktarına göre maliyet hesaplanmalı
                actual_prod = min(capacity, demand[t])
                prod_cost = actual_prod * production_cost  # Kapasite değil, gerçek üretim miktarı

                total_cost = cost_table[t, prev_w] + hire + fire + holding + stockout + labor + prod_cost
                if total_cost < cost_table[t+1, w]:
                    cost_table[t+1, w] = total_cost
                    backtrack[t+1, w] = prev_w

# En düşük maliyetli yolun sonundaki işçi sayısı
final_workers = np.argmin(cost_table[months])
min_cost = cost_table[months, final_workers]

# Geriye doğru optimal yolun çıkarılması
workers_seq = []
w = final_workers
for t in range(months, 0, -1):
    workers_seq.append(w)
    w = backtrack[t, w]
workers_seq = workers_seq[::-1]

# Üretim ve stok hesaplama
production_seq = []
inventory_seq = []
stockout_seq = []
for t, w in enumerate(workers_seq):
    cap = prod_capacity(w, t)
    prod = min(cap, demand[t])
    production_seq.append(prod)
    inventory_seq.append(max(0, cap - demand[t]))
    stockout_seq.append(max(0, demand[t] - cap))

# Sonuç tablosu
results = []
total_labor = 0
total_production = 0
total_holding = 0
total_stockout = 0
total_hiring = 0
total_firing = 0

for t in range(months):
    # İşçi sayısı değişiminden kaynaklanan maliyetler
    if t > 0:
        hire = max(0, workers_seq[t] - workers_seq[t-1]) * hiring_cost
        fire = max(0, workers_seq[t-1] - workers_seq[t]) * firing_cost
    else:
        hire = workers_seq[t] * hiring_cost
        fire = 0

    labor_cost = workers_seq[t] * working_days[t] * daily_hours * hourly_wage
    prod_cost = production_seq[t] * production_cost
    holding = inventory_seq[t] * holding_cost
    stockout = stockout_seq[t] * stockout_cost

    # Toplam maliyetleri hesapla
    total_labor += labor_cost
    total_production += prod_cost
    total_holding += holding
    total_stockout += stockout
    total_hiring += hire
    total_firing += fire

    results.append([
        t+1, workers_seq[t], production_seq[t], inventory_seq[t], stockout_seq[t],
        labor_cost, prod_cost, hire, fire, holding, stockout
    ])

df = pd.DataFrame(results, columns=[
    'Ay', 'İşçi', 'Üretim', 'Stok', 'Karşılanmayan Talep',
    'İşçilik Maliyeti (₺)', 'Üretim Maliyeti (₺)', 'İşe Alım Maliyeti (₺)',
    'İşten Çıkarma Maliyeti (₺)', 'Stok Maliyeti (₺)', 'Stoksuzluk Maliyeti (₺)'
])

print(tabulate(df, headers='keys', tablefmt='fancy_grid', showindex=False, numalign='right'))
print(f'\nToplam Maliyet: {min_cost:,.2f} TL')
print(f"\nAyrıntılı Toplam Maliyetler:")
print(f"- İşçilik Maliyeti Toplamı: {total_labor:,.2f} TL")
print(f"- Üretim Maliyeti Toplamı: {total_production:,.2f} TL")
print(f"- Stok Maliyeti Toplamı: {total_holding:,.2f} TL")
print(f"- Karşılanmayan Talep Maliyeti Toplamı: {total_stockout:,.2f} TL")
print(f"- İşe Alım Maliyeti Toplamı: {total_hiring:,.2f} TL")
print(f"- İşten Çıkarma Maliyeti Toplamı: {total_firing:,.2f} TL")

# Birim maliyet hesaplaması
total_demand = sum(demand)
total_produced = sum(production_seq)
total_unfilled = sum(stockout_seq)
total_cost = min_cost

print(f"\nBirim Maliyet Analizi:")
print(f"- Toplam Talep: {total_demand:,} birim")
print(f"- Toplam Üretim: {total_produced:,} birim ({total_produced/total_demand*100:.2f}%)")
print(f"- Karşılanmayan Talep: {total_unfilled:,} birim ({total_unfilled/total_demand*100:.2f}%)")
if total_produced > 0:
    print(f"- Ortalama Birim Maliyet (Toplam): {total_cost/total_produced:.2f} TL/birim")
    print(f"- Ortalama İşçilik Birim Maliyeti: {total_labor/total_produced:.2f} TL/birim")
    print(f"- Ortalama Üretim Birim Maliyeti: {total_production/total_produced:.2f} TL/birim")
    print(f"- Diğer Maliyetler (Stok, İşe Alım/Çıkarma): {(total_holding+total_hiring+total_firing)/total_produced:.2f} TL/birim")
else:
    print("- Ortalama Birim Maliyet: Hesaplanamadı (0 birim üretildi)")

# Grafiksel çıktı
try:
    import matplotlib.pyplot as plt
except ImportError:
    print('matplotlib kütüphanesi eksik. Kurmak için: pip install matplotlib')
    exit(1)
months_list = list(range(1, months+1))
fig, ax1 = plt.subplots(figsize=(10,6))
ax1.bar(months_list, workers_seq, color='skyblue', label='İşçi', alpha=0.7)
ax1.set_xlabel('Ay')
ax1.set_ylabel('İşçi Sayısı', color='skyblue')
ax1.tick_params(axis='y', labelcolor='skyblue')
ax2 = ax1.twinx()
ax2.plot(months_list, production_seq, marker='s', label='Üretim', color='green')
ax2.plot(months_list, inventory_seq, marker='d', label='Stok', color='red')
ax2.plot(months_list, stockout_seq, marker='x', label='Karşılanmayan Talep', color='black')
ax2.set_ylabel('Adet', color='gray')
ax2.tick_params(axis='y', labelcolor='gray')
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
plt.title('Dinamik Programlama Tabanlı Model Sonuçları')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
