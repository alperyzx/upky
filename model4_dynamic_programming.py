import numpy as np
import pandas as pd

# Parametreler
# Çok yüksek talep için örnek:
demand = np.array([1500, 1200, 3600, 4500, 7200, 9000, 7200, 6400, 7200, 9000, 9600, 12000])
working_days = np.array([22, 20, 23, 19, 21, 19, 22, 22, 22, 21, 21, 21])
holding_cost = 5
stockout_cost = 20
hiring_cost = 1000
firing_cost = 800
daily_hours = 8
labor_per_unit = 0.5
max_workers = 40  # Daha yüksek üst sınır
min_workers = 8
max_workforce_change = 8  # Daha hızlı işçi artışı
months = len(demand)

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
                # Üretim kapasitesi işçiyle sınırlı, ancak üretim talebi kadar olmalı
                if capacity < demand[t]:
                    continue
                inventory = capacity - demand[t]
                hire = max(0, w - prev_w) * hiring_cost
                fire = max(0, prev_w - w) * firing_cost
                holding = inventory * holding_cost
                total_cost = cost_table[t, prev_w] + hire + fire + holding
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
for t, w in enumerate(workers_seq):
    cap = prod_capacity(w, t)
    production_seq.append(min(cap, demand[t]))
    inventory_seq.append(cap - demand[t])

# Sonuç tablosu
results = []
for t in range(months):
    results.append([
        t+1, workers_seq[t], production_seq[t], inventory_seq[t]
    ])
df = pd.DataFrame(results, columns=['Ay', 'İşçi', 'Üretim', 'Stok'])
print(df.to_string(index=False))
print(f'\nToplam Maliyet: {min_cost:,.2f} TL')

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
ax2.set_ylabel('Adet', color='gray')
ax2.tick_params(axis='y', labelcolor='gray')
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
plt.title('Dinamik Programlama Tabanlı Model Sonuçları')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

