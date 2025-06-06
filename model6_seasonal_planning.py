import pulp
import numpy as np
import pandas as pd
from tabulate import tabulate

# 12 aylık talep verisi
demand = np.array([300, 310, 350, 300, 350, 450, 460, 450, 320, 350, 350, 300])
working_days= [22, 20, 23, 19, 21, 19, 22, 22, 22, 21, 21, 21]  # Aylık çalışma günleri
months = len(demand)
holding_cost = 5
stockout_cost = 80
production_cost = 30
labor_per_unit = 4  # 1 ürün için gereken işçilik süresi (saat)
hiring_cost = 1800
daily_hours = 8  # Günlük çalışma saati
max_production = int(demand.mean() + demand.std())  # 12 a ylık talep ortalaması * 1,1

# max_production'a göre ihtiyaç duyulan işçi sayısı
# max_production = işçi_sayısı * günlük_saat * aylık_gün / labor_per_unit
# işçi_sayısı = max_production * labor_per_unit / (günlük_saat * ortalama_aylık_gün)
needed_workers = int(np.ceil(max_production * labor_per_unit / (daily_hours * np.mean(working_days))))
print(f"Optimum üretim kapasitesi için gereken işçi sayısı: {needed_workers}")

# Her ay işçilik maliyeti
hourly_wage = 10
monthly_labor_cost = needed_workers * np.mean(working_days) * daily_hours * hourly_wage

# Doğrusal programlama modeli
model = pulp.LpProblem('Mevsimsel_Stok_Optimizasyonu', pulp.LpMinimize)

# Karar değişkenleri
y_production = [pulp.LpVariable(f'production_{t}', lowBound=0, cat='Integer') for t in range(months)]
y_inventory = [pulp.LpVariable(f'inventory_{t}', lowBound=0, cat='Integer') for t in range(months)]
y_stockout = [pulp.LpVariable(f'stockout_{t}', lowBound=0, cat='Integer') for t in range(months)]

# Amaç fonksiyonu: Üretim, stok, stoksuzluk, işe alım ve toplam işçilik maliyetlerinin toplamı
model += (
    pulp.lpSum([
        y_production[t] * production_cost +
        y_inventory[t] * holding_cost +
        y_stockout[t] * stockout_cost
        for t in range(months)
    ])
    + hiring_cost * needed_workers
    + monthly_labor_cost * months
)

# Kısıtlar
for t in range(months):
    # Üretim kapasitesi
    model += y_production[t] <= max_production
    # Stok ve stoksuzluk denklemi
    if t == 0:
        prev_inventory = 0
    else:
        prev_inventory = y_inventory[t-1]
    model += prev_inventory + y_production[t] + y_stockout[t] == demand[t] + y_inventory[t]
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
        demand[t],
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

def ayrintili_toplam_maliyetler(total_holding, total_stockout, total_production_cost, total_labor_cost, total_hiring_cost):
    return {
        'total_holding': total_holding,
        'total_stockout': total_stockout,
        'total_production_cost': total_production_cost,
        'total_labor_cost': total_labor_cost,
        'total_hiring_cost': total_hiring_cost
    }

def birim_maliyet_analizi(total_demand, total_produced, total_unfilled, total_cost, total_labor_cost, total_production_cost, total_holding, total_stockout):
    if total_produced > 0:
        avg_unit_cost = total_cost / total_produced
        avg_labor_unit = total_labor_cost / total_produced
        avg_prod_unit = total_production_cost / total_produced
        # Calculate other costs (total cost minus labor and production costs)
        other_costs = total_cost - total_labor_cost - total_production_cost
        avg_other_unit = other_costs / total_produced
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

# Ayrıntılı maliyetleri hesapla ve yazdır

total_hiring_cost = hiring_cost * needed_workers
detay = ayrintili_toplam_maliyetler(total_holding, total_stockout, total_production_cost, total_labor_cost, total_hiring_cost)
print(f"\nAyrıntılı Toplam Maliyetler:")
print(f"- Stok Maliyeti Toplamı: {detay['total_holding']:,.2f} TL")
print(f"- Stoksuzluk Maliyeti Toplamı: {detay['total_stockout']:,.2f} TL")
print(f"- Üretim Maliyeti Toplamı: {detay['total_production_cost']:,.2f} TL")
print(f"- İşçilik Maliyeti Toplamı: {detay['total_labor_cost']:,.2f} TL")
print(f"- İşe Alım Maliyeti Toplamı: {detay['total_hiring_cost']:,.2f} TL")

total_produced = sum([int(y_production[t].varValue) for t in range(months)])
total_unfilled = sum([int(y_stockout[t].varValue) for t in range(months)])
birim = birim_maliyet_analizi(
    demand.sum(),
    total_produced,
    total_unfilled,
    total_cost,
    total_labor_cost,
    total_production_cost,
    total_holding,
    total_stockout
)
print(f"\nBirim Maliyet Analizi:")
print(f"- Toplam Talep: {birim['total_demand']:,} birim")
print(f"- Toplam Üretim: {birim['total_produced']:,} birim ({birim['total_produced']/birim['total_demand']*100:.2f}%)")
print(f"- Karşılanmayan Talep: {birim['total_unfilled']:,} birim ({birim['total_unfilled']/birim['total_demand']*100:.2f}%)")
if birim['total_produced'] > 0:
    print(f"- Ortalama Birim Maliyet (Toplam): {birim['avg_unit_cost']:.2f} TL/birim")
    print(f"- Ortalama İşçilik Birim Maliyeti: {birim['avg_labor_unit']:.2f} TL/birim")
    print(f"- Ortalama Üretim Birim Maliyeti: {birim['avg_prod_unit']:.2f} TL/birim")
    print(f"- Diğer Maliyetler (Stok, Stoksuzluk): {birim['avg_other_unit']:.2f} TL/birim")
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
plt.plot(months_list, demand, marker='o', label='Talep', color='orange')
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

def maliyet_analizi(
    demand=demand,
    holding_cost=holding_cost,
    stockout_cost=stockout_cost,
    production_cost=production_cost,
    max_production=max_production,
    labor_per_unit=labor_per_unit,
    hourly_wage=hourly_wage
):
    months = len(demand)
    needed_workers = int(np.ceil(max_production * labor_per_unit / (daily_hours * np.mean(working_days))))
    monthly_labor_cost = needed_workers * np.mean(working_days) * daily_hours * hourly_wage
    model = pulp.LpProblem('Mevsimsel_Stok_Optimizasyonu', pulp.LpMinimize)
    y_production = [pulp.LpVariable(f'production_{t}', lowBound=0, cat='Integer') for t in range(months)]
    y_inventory = [pulp.LpVariable(f'inventory_{t}', lowBound=0, cat='Integer') for t in range(months)]
    y_stockout = [pulp.LpVariable(f'stockout_{t}', lowBound=0, cat='Integer') for t in range(months)]
    # Fix the objective function to match the main model
    model += (
        pulp.lpSum([
            production_cost * y_production[t] +
            holding_cost * y_inventory[t] +
            stockout_cost * y_stockout[t] +
            monthly_labor_cost
            for t in range(months)
        ])
        + hiring_cost * needed_workers
    )
    for t in range(months):
        model += y_production[t] <= max_production
        if t == 0:
            prev_inventory = 0
        else:
            prev_inventory = y_inventory[t-1]
        model += prev_inventory + y_production[t] + y_stockout[t] == demand[t] + y_inventory[t]
        model += y_inventory[t] >= 0
        model += y_stockout[t] >= 0
    solver = pulp.PULP_CBC_CMD(msg=0)
    model.solve(solver)
    total_production = 0
    total_holding = 0
    total_stockout = 0
    total_labor = 0
    total_demand = sum(demand)
    total_unfilled = 0
    for t in range(months):
        prod = int(y_production[t].varValue)
        inv = int(y_inventory[t].varValue)
        so = int(y_stockout[t].varValue)
        total_production += prod
        total_holding += inv * holding_cost
        total_stockout += so * stockout_cost
        total_labor += monthly_labor_cost
        total_unfilled += so
    total_hiring_cost = hiring_cost * needed_workers
    # Get the total cost directly from the model's objective value
    toplam_maliyet = pulp.value(model.objective)
    if total_production > 0:
        avg_unit_cost = toplam_maliyet / total_production
        avg_labor_unit = total_labor / total_production
        avg_prod_unit = (total_production * production_cost) / total_production
        avg_other_unit = (total_holding + total_stockout) / total_production
    else:
        avg_unit_cost = avg_labor_unit = avg_prod_unit = avg_other_unit = 0
    return {
        "Toplam Maliyet": toplam_maliyet,
        "İşçilik Maliyeti": total_labor,
        "Üretim Maliyeti": total_production * production_cost,
        "Stok Maliyeti": total_holding,
        "Stoksuzluk Maliyeti": total_stockout,
        "İşe Alım Maliyeti": total_hiring_cost,
        "İşten Çıkarma Maliyeti": 0,
        "Toplam Talep": total_demand,
        "Toplam Üretim": total_production,
        "Karşılanmayan Talep": total_unfilled,
        "Ortalama Birim Maliyet": avg_unit_cost,
        "İşçilik Birim Maliyeti": avg_labor_unit,
        "Üretim Birim Maliyeti": avg_prod_unit,
        "Diğer Birim Maliyetler": avg_other_unit
    }
