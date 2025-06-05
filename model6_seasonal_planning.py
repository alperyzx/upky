import pulp
import numpy as np
import pandas as pd
from tabulate import tabulate

# Örnek mevsimsel talep (12 ay)
seasonal_demand = np.array([120, 450, 550, 280, 210, 150, 450, 600, 300, 250, 150, 100])
months = len(seasonal_demand)
holding_cost = 5
stockout_cost = 80
production_cost = 30
max_production = int(seasonal_demand.mean() * 1.3)  # 12 aylık talep ortalaması * 1,1
labor_per_unit = 4  # 1 ürün için gereken işçilik süresi (saat)

# max_production'a göre ihtiyaç duyulan işçi sayısı
# max_production = işçi_sayısı * günlük_saat * aylık_gün / labor_per_unit
# işçi_sayısı = max_production * labor_per_unit / (günlük_saat * ortalama_aylık_gün)
avg_working_days = np.mean([22, 20, 23, 19, 21, 19, 22, 22, 22, 21, 21, 21])
needed_workers = int(np.ceil(max_production * labor_per_unit / (8 * avg_working_days)))
print(f"Max üretim kapasitesi için gereken işçi sayısı: {needed_workers}")

# Her ay işçilik maliyeti
hourly_wage = 10
monthly_labor_cost = needed_workers * avg_working_days * 8 * hourly_wage

# Doğrusal programlama modeli
model = pulp.LpProblem('Mevsimsel_Stok_Optimizasyonu', pulp.LpMinimize)

# Karar değişkenleri
y_production = [pulp.LpVariable(f'production_{t}', lowBound=0, cat='Integer') for t in range(months)]
y_inventory = [pulp.LpVariable(f'inventory_{t}', lowBound=0, cat='Integer') for t in range(months)]
y_stockout = [pulp.LpVariable(f'stockout_{t}', lowBound=0, cat='Integer') for t in range(months)]

# Amaç fonksiyonu
model += pulp.lpSum([
    production_cost * y_production[t] +
    holding_cost * y_inventory[t] +
    stockout_cost * y_stockout[t] +
    monthly_labor_cost  # Her ay işçilik maliyeti
    for t in range(months)
])

# Kısıtlar
for t in range(months):
    # Üretim kapasitesi
    model += y_production[t] <= max_production
    # Stok ve stoksuzluk denklemi
    if t == 0:
        prev_inventory = 0
    else:
        prev_inventory = y_inventory[t-1]
    model += prev_inventory + y_production[t] + y_stockout[t] == seasonal_demand[t] + y_inventory[t]
    model += y_inventory[t] >= 0
    model += y_stockout[t] >= 0

# Modeli çöz
solver = pulp.PULP_CBC_CMD(msg=0)
model.solve(solver)

results = []
total_holding = 0
total_stockout = 0
total_production_cost = 0
total_labor_cost = 0

for t in range(months):
    labor_cost = monthly_labor_cost
    holding = int(y_inventory[t].varValue) * holding_cost
    stockout = int(y_stockout[t].varValue) * stockout_cost
    prod_cost = int(y_production[t].varValue) * production_cost

    # Toplam maliyetleri hesapla
    total_holding += holding
    total_stockout += stockout
    total_production_cost += prod_cost
    total_labor_cost += labor_cost

    results.append([
        t+1,
        seasonal_demand[t],
        int(y_production[t].varValue),
        int(y_inventory[t].varValue),
        int(y_stockout[t].varValue),
        holding,
        stockout,
        prod_cost,
        labor_cost
    ])

df = pd.DataFrame(results, columns=[
    'Ay', 'Talep', 'Üretim', 'Stok', 'Stoksuzluk', 'Stok Maliyeti', 'Stoksuzluk Maliyeti', 'Üretim Maliyeti', 'İşçilik Maliyeti'
])
# Format cost columns
for col in ['Stok Maliyeti', 'Stoksuzluk Maliyeti', 'Üretim Maliyeti', 'İşçilik Maliyeti']:
    df[col] = df[col].apply(lambda x: f'{int(x):,} TL')
print(tabulate(df, headers='keys', tablefmt='fancy_grid', showindex=False, numalign='right', stralign='center'))

# Toplam maliyeti hesapla
total_cost = pulp.value(model.objective)
print(f'\nToplam Maliyet: {total_cost:,.2f} TL')

# Ayrıntılı maliyetleri göster
print(f"\nAyrıntılı Toplam Maliyetler:")
print(f"- Stok Maliyeti Toplamı: {total_holding:,.2f} TL")
print(f"- Stoksuzluk Maliyeti Toplamı: {total_stockout:,.2f} TL")
print(f"- Üretim Maliyeti Toplamı: {total_production_cost:,.2f} TL")
print(f"- İşçilik Maliyeti Toplamı: {total_labor_cost:,.2f} TL")

# Birim maliyet hesaplaması
total_demand = seasonal_demand.sum()
total_produced = sum([int(y_production[t].varValue) for t in range(months)])
total_unfilled = sum([int(y_stockout[t].varValue) for t in range(months)])

print(f"\nBirim Maliyet Analizi:")
print(f"- Toplam Talep: {total_demand:,} birim")
print(f"- Toplam Üretim: {total_produced:,} birim ({total_produced/total_demand*100:.2f}%)")
print(f"- Karşılanmayan Talep: {total_unfilled:,} birim ({total_unfilled/total_demand*100:.2f}%)")

if total_produced > 0:
    print(f"- Ortalama Birim Maliyet (Toplam): {total_cost/total_produced:.2f} TL/birim")
    print(f"- Ortalama İşçilik Birim Maliyeti: {total_labor_cost/total_produced:.2f} TL/birim")
    print(f"- Ortalama Üretim Birim Maliyeti: {total_production_cost/total_produced:.2f} TL/birim")
    print(f"- Diğer Maliyetler (Stok, Stoksuzluk): {(total_holding+total_stockout)/total_produced:.2f} TL/birim")
else:
    print("- Ortalama Birim Maliyet: Hesaplanamadı (0 birim üretildi)")

# Grafiksel çıktı
try:
    import matplotlib.pyplot as plt
except ImportError:
    print('matplotlib kütüphanesi eksik. Kurmak için: pip install matplotlib')
    exit(1)
months_list = list(range(1, months+1))
plt.figure(figsize=(12,6))
plt.plot(months_list, seasonal_demand, marker='o', label='Talep', color='orange')
plt.bar(months_list, [int(y_production[t].varValue) for t in range(months)], color='skyblue', label='Üretim', alpha=0.7)
plt.plot(months_list, [int(y_inventory[t].varValue) for t in range(months)], marker='d', label='Stok', color='red')
plt.plot(months_list, [int(y_stockout[t].varValue) for t in range(months)], marker='x', label='Stoksuzluk', color='black')
plt.xlabel('Ay')
plt.ylabel('Adet')
plt.title('Mevsimsellik ve Stok Optimizasyonu Sonuçları')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
