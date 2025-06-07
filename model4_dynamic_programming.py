import numpy as np
import pandas as pd
from tabulate import tabulate
import yaml
import os

# parametreler.yaml dosyasını oku
with open(os.path.join(os.path.dirname(__file__), 'parametreler.yaml'), 'r', encoding='utf-8') as f:
    params = yaml.safe_load(f)

demand = np.array(params['demand']['normal'])  # veya 'high', 'seasonal' seçilebilir
working_days = np.array(params['workforce']['working_days'])
holding_cost = params['costs']['holding_cost']
stockout_cost = params['costs']['stockout_cost']
hiring_cost = params['costs']['hiring_cost']
firing_cost = params['costs']['firing_cost']
daily_hours = params['workforce']['daily_hours']
labor_per_unit = params['workforce']['labor_per_unit']
max_workers = params['workforce']['max_workers']
min_workers = params['workforce']['workers']
max_workforce_change = params['workforce']['max_workforce_change']
months = len(demand)
hourly_wage = params['costs']['hourly_wage']
production_cost = params['costs']['production_cost']

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

def maliyet_analizi(
    demand=demand,
    working_days=working_days,
    holding_cost=holding_cost,
    stockout_cost=stockout_cost,
    hiring_cost=hiring_cost,
    firing_cost=firing_cost,
    daily_hours=daily_hours,
    labor_per_unit=labor_per_unit,
    min_workers=min_workers,
    max_workers=max_workers,
    max_workforce_change=max_workforce_change,
    hourly_wage=hourly_wage,
    production_cost=production_cost
):
    months = len(demand)
    # DP tabloları
    cost_table = np.full((months+1, max_workers+1), np.inf)
    backtrack = np.full((months+1, max_workers+1), -1, dtype=int)
    cost_table[0, min_workers:max_workers+1] = 0
    for t in range(months):
        for prev_w in range(min_workers, max_workers+1):
            if cost_table[t, prev_w] < np.inf:
                for w in range(max(min_workers, prev_w-max_workforce_change), min(max_workers, prev_w+max_workforce_change)+1):
                    capacity = w * working_days[t] * daily_hours / labor_per_unit
                    unmet = max(0, demand[t] - capacity)
                    inventory = max(0, capacity - demand[t])
                    hire = max(0, w - prev_w) * hiring_cost
                    fire = max(0, prev_w - w) * firing_cost
                    holding = inventory * holding_cost
                    stockout = unmet * stockout_cost
                    labor = w * working_days[t] * daily_hours * hourly_wage
                    actual_prod = min(capacity, demand[t])
                    prod_cost = actual_prod * production_cost
                    total_cost = cost_table[t, prev_w] + hire + fire + holding + stockout + labor + prod_cost
                    if total_cost < cost_table[t+1, w]:
                        cost_table[t+1, w] = total_cost
                        backtrack[t+1, w] = prev_w
    # En düşük maliyetli yolun sonundaki işçi sayısı
    min_cost = np.min(cost_table[months, min_workers:max_workers+1])
    last_worker = np.argmin(cost_table[months, min_workers:max_workers+1]) + min_workers
    # Geriye doğru izleme
    workers = [0]*months
    w = last_worker
    for t in range(months, 0, -1):
        workers[t-1] = w
        w = backtrack[t, w]
    # Sonuçların hesaplanması
    total_labor = 0
    total_production = 0
    total_holding = 0
    total_stockout = 0
    total_hiring = 0
    total_firing = 0
    total_demand = sum(demand)
    total_produced = 0
    total_unfilled = 0
    for t in range(months):
        capacity = workers[t] * working_days[t] * daily_hours / labor_per_unit
        actual_prod = min(capacity, demand[t])
        unmet = max(0, demand[t] - capacity)
        inventory = max(0, capacity - demand[t])
        hire = max(0, workers[t] - (workers[t-1] if t > 0 else min_workers)) * hiring_cost
        fire = max(0, (workers[t-1] if t > 0 else min_workers) - workers[t]) * firing_cost
        holding = inventory * holding_cost
        stockout = unmet * stockout_cost
        labor = workers[t] * working_days[t] * daily_hours * hourly_wage
        prod_cost = actual_prod * production_cost
        total_labor += labor
        total_production += prod_cost
        total_holding += holding
        total_stockout += stockout
        total_hiring += hire
        total_firing += fire
        total_produced += actual_prod
        total_unfilled += unmet
    toplam_maliyet = total_labor + total_production + total_holding + total_stockout + total_hiring + total_firing
    if total_produced > 0:
        avg_unit_cost = toplam_maliyet / total_produced
        avg_labor_unit = total_labor / total_produced
        avg_prod_unit = total_production / total_produced
        avg_other_unit = (total_holding + total_stockout + total_hiring + total_firing) / total_produced
    else:
        avg_unit_cost = avg_labor_unit = avg_prod_unit = avg_other_unit = 0
    return {
        "Toplam Maliyet": toplam_maliyet,
        "İşçilik Maliyeti": total_labor,
        "Üretim Maliyeti": total_production,
        "Stok Maliyeti": total_holding,
        "Stoksuzluk Maliyeti": total_stockout,
        "İşe Alım Maliyeti": total_hiring,
        "İşten Çıkarma Maliyeti": total_firing,
        "Toplam Talep": total_demand,
        "Toplam Üretim": total_produced,
        "Karşılanmayan Talep": total_unfilled,
        "Ortalama Birim Maliyet": avg_unit_cost,
        "İşçilik Birim Maliyeti": avg_labor_unit,
        "Üretim Birim Maliyeti": avg_prod_unit,
        "Diğer Birim Maliyetler": avg_other_unit
    }

def ayrintili_toplam_maliyetler(total_labor, total_production, total_holding, total_stockout, total_hiring, total_firing):
    return {
        'total_labor': total_labor,
        'total_production': total_production,
        'total_holding': total_holding,
        'total_stockout': total_stockout,
        'total_hiring': total_hiring,
        'total_firing': total_firing
    }

def birim_maliyet_analizi(total_demand, total_produced, total_unfilled, total_cost, total_labor, total_production, total_holding, total_hiring, total_firing):
    if total_produced > 0:
        avg_unit_cost = total_cost / total_produced
        avg_labor_unit = total_labor / total_produced
        avg_prod_unit = total_production / total_produced
        avg_other_unit = (total_holding + total_hiring + total_firing) / total_produced
    else:
        avg_unit_cost = avg_labor_unit = avg_prod_unit = avg_other_unit = 0
    return {
        'total_demand': total_demand,
        'total_produced': total_produced,
        'total_unfilled': total_unfilled,
        'avg_unit_cost': avg_unit_cost,
        'avg_labor_unit': avg_labor_unit,
        'avg_prod_unit': avg_prod_unit,
        'avg_other_unit': avg_other_unit
    }

detay = ayrintili_toplam_maliyetler(total_labor, total_production, total_holding, total_stockout, total_hiring, total_firing)
print(f'\nToplam Maliyet: {min_cost:,.2f} TL')
print(f"\nAyrıntılı Toplam Maliyetler:")
print(f"- İşçilik Maliyeti Toplamı: {detay['total_labor']:,.2f} TL")
print(f"- Üretim Maliyeti Toplamı: {detay['total_production']:,.2f} TL")
print(f"- Stok Maliyeti Toplamı: {detay['total_holding']:,.2f} TL")
print(f"- Karşılanmayan Talep Maliyeti Toplamı: {detay['total_stockout']:,.2f} TL")
print(f"- İşe Alım Maliyeti Toplamı: {detay['total_hiring']:,.2f} TL")
print(f"- İşten Çıkarma Maliyeti Toplamı: {detay['total_firing']:,.2f} TL")

birim = birim_maliyet_analizi(
    sum(demand),
    sum(production_seq),
    sum(stockout_seq),
    min_cost,
    total_labor,
    total_production,
    total_holding,
    total_hiring,
    total_firing
)
print(f"\nBirim Maliyet Analizi:")
print(f"- Toplam Talep: {birim['total_demand']:,} birim")
print(f"- Toplam Üretim: {birim['total_produced']:,} birim ({birim['total_produced']/birim['total_demand']*100:.2f}%)")
print(f"- Karşılanmayan Talep: {birim['total_unfilled']:,} birim ({birim['total_unfilled']/birim['total_demand']*100:.2f}%)")
if birim['total_produced'] > 0:
    print(f"- Ortalama Birim Maliyet (Toplam): {birim['avg_unit_cost']:.2f} TL/birim")
    print(f"- Ortalama İşçilik Birim Maliyeti: {birim['avg_labor_unit']:.2f} TL/birim")
    print(f"- Ortalama Üretim Birim Maliyeti: {birim['avg_prod_unit']:.2f} TL/birim")
    print(f"- Diğer Maliyetler (Stok, İşe Alım/Çıkarma): {birim['avg_other_unit']:.2f} TL/birim")
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
