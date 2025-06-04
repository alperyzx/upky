import pulp
import numpy as np
import pandas as pd

# Örnek mevsimsel talep (12 ay)
seasonal_demand = np.array([1200, 4500, 5500, 2800, 2100, 1500, 4500, 6000, 3000, 2500, 1500, 1000])
months = len(seasonal_demand)
holding_cost = 5
stockout_cost = 20
production_cost = 10
max_production = 4000  # Maksimum aylık üretim kapasitesi

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
    stockout_cost * y_stockout[t]
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
for t in range(months):
    results.append([
        t+1,
        seasonal_demand[t],
        int(y_production[t].varValue),
        int(y_inventory[t].varValue),
        int(y_stockout[t].varValue),
        int(y_inventory[t].varValue) * holding_cost,
        int(y_stockout[t].varValue) * stockout_cost,
        int(y_production[t].varValue) * production_cost
    ])

df = pd.DataFrame(results, columns=[
    'Ay', 'Talep', 'Üretim', 'Stok', 'Stoksuzluk', 'Stok Maliyeti', 'Stoksuzluk Maliyeti', 'Üretim Maliyeti'
])
print(df.to_string(index=False))
print(f'\nToplam Maliyet: {pulp.value(model.objective):,.2f} TL')

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

