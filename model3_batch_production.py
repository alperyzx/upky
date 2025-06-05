import numpy as np
import pandas as pd
from tabulate import tabulate

# Parametreler
demand = np.array([650, 650, 750, 650, 700, 600, 700, 650,650, 670, 750, 650])
working_days = np.array([22, 20, 23, 19, 21, 19, 22, 22, 22, 21, 21, 21])
holding_cost = 5
stockout_cost = 20
fixed_workers = 20
worker_monthly_cost = 1680  # İşçi başı aylık maliyet (ör: asgari ücret)
production_rate = 0.25 # Bir işçi günde 0.25 birim üretim yapabiliyor
daily_hours = 8
months = len(demand)
production_cost = 20  # birim üretim maliyeti (TL)

# Check that demand and working_days have the same length
if len(demand) != len(working_days):
    raise ValueError(f"Length of demand ({len(demand)}) and working_days ({len(working_days)}) must be equal.")

# Sabit üretim kapasitesi (işçi * gün * saat * hız)
monthly_capacity = fixed_workers * daily_hours * working_days * production_rate

results = []
production = np.zeros(months)
inventory = np.zeros(months)
cost = 0
prev_inventory = 0

for t in range(months):
    # Sabit üretim
    production[t] = monthly_capacity[t]
    inventory[t] = prev_inventory + production[t] - demand[t]
    holding = max(inventory[t], 0) * holding_cost
    stockout = abs(min(inventory[t], 0)) * stockout_cost
    labor = fixed_workers * worker_monthly_cost
    prod_cost = production[t] * production_cost
    cost += holding + stockout + labor + prod_cost
    results.append([
        t+1, production[t], inventory[t], holding, stockout, labor, prod_cost
    ])
    prev_inventory = inventory[t]

df = pd.DataFrame(results, columns=[
    'Ay', 'Üretim', 'Stok', 'Stok Maliyeti (₺)', 'Stoksuzluk Maliyeti (₺)', 'İşçilik Maliyeti (₺)', 'Üretim Maliyeti (₺)'
])

# Hücrelerden TL birimini kaldır, sadece sayısal kalsın (virgülsüz, int)
df['Stok Maliyeti (₺)'] = df['Stok Maliyeti (₺)'].astype(int)
df['Stoksuzluk Maliyeti (₺)'] = df['Stoksuzluk Maliyeti (₺)'].astype(int)
df.rename(columns={df.columns[5]: 'İşçilik Maliyeti (₺)'}, inplace=True)  # Fix encoding issue
df['İşçilik Maliyeti (₺)'] = df['İşçilik Maliyeti (₺)'].astype(int)
df['Üretim Maliyeti (₺)'] = df['Üretim Maliyeti (₺)'].astype(int)
print(tabulate(df, headers='keys', tablefmt='fancy_grid', showindex=False, numalign='right', stralign='center'))
print(f'\nToplam Maliyet: {cost:,.2f} TL')
print(f'Stok Maliyeti Toplamı: {df["Stok Maliyeti (₺)"].sum():,} TL')
print(f'Stoksuzluk Maliyeti Toplamı: {df["Stoksuzluk Maliyeti (₺)"].sum():,} TL')
print(f'İşçilik Maliyeti Toplamı: {df["İşçilik Maliyeti (₺)"].sum():,} TL')
print(f'Üretim Maliyeti Toplamı: {df["Üretim Maliyeti (₺)"].sum():,} TL')

# Birim maliyet hesaplaması
total_demand = demand.sum()
total_produced = production.sum()
total_unfilled = sum([abs(min(inventory[t], 0)) for t in range(months)])
total_holding = df["Stok Maliyeti (₺)"].sum()
total_stockout = df["Stoksuzluk Maliyeti (₺)"].sum()
total_labor = df["İşçilik Maliyeti (₺)"].sum()
total_production_cost = df["Üretim Maliyeti (₺)"].sum()

print(f"\nBirim Maliyet Analizi:")
print(f"- Toplam Talep: {total_demand:,} birim")
print(f"- Toplam Üretim: {total_produced:,} birim ({total_produced/total_demand*100:.2f}%)")
if total_unfilled > 0:
    print(f"- Karşılanmayan Talep: {total_unfilled:,} birim ({total_unfilled/total_demand*100:.2f}%)")

if total_produced > 0:
    print(f"- Ortalama Birim Maliyet (Toplam): {cost/total_produced:.2f} TL/birim")
    print(f"- İşçilik Birim Maliyeti: {total_labor/total_produced:.2f} TL/birim")
    print(f"- Üretim Birim Maliyeti: {total_production_cost/total_produced:.2f} TL/birim")
    print(f"- Diğer Maliyetler (Stok, Stoksuzluk): {(total_holding+total_stockout)/total_produced:.2f} TL/birim")
    print(f"- Sabit İşçi Sayısı: {fixed_workers} kişi")
    print(f"- İşçi Başına Aylık Ortalama Üretim: {total_produced/(fixed_workers*months):.2f} birim/ay")
else:
    print("- Ortalama Birim Maliyet: Hesaplanamadı (0 birim üretildi)")

# Grafiksel çıktı
try:
    import matplotlib.pyplot as plt
except ImportError:
    print('matplotlib kütüphanesi eksik. Kurmak için: pip install matplotlib')
    exit(1)
months_list = list(range(1, months+1))
plt.figure(figsize=(10,6))
plt.bar(months_list, production, color='skyblue', label='Üretim', alpha=0.7)
plt.plot(months_list, inventory, marker='d', label='Stok', color='red')
plt.plot(months_list, df['Stoksuzluk Maliyeti (₺)'], marker='x', label='Stoksuzluk Maliyeti', color='black')
plt.xlabel('Ay')
plt.ylabel('Adet / TL')
plt.title('Toplu Üretim ve Stoklama Modeli Sonuçları')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
