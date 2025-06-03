import numpy as np
import pandas as pd

# Örnek mevsimsel talep (12 ay)
seasonal_demand = np.array([900, 1200, 1500, 1650, 1300, 1200, 1000, 900, 900, 850, 850, 1000])
months = len(seasonal_demand)
holding_cost = 5
stockout_cost = 20
production_cost = 12
max_production = 1500  # Maksimum aylık üretim kapasitesi

results = []
production = np.zeros(months)
inventory = np.zeros(months)
cost = 0
prev_inventory = 0

for t in range(months):
    # Talep ve kapasiteye göre üretim kararı
    if seasonal_demand[t] > max_production:
        production[t] = max_production
    else:
        production[t] = seasonal_demand[t]
    inventory[t] = prev_inventory + production[t] - seasonal_demand[t]
    holding = max(inventory[t], 0) * holding_cost
    stockout = abs(min(inventory[t], 0)) * stockout_cost
    prod_cost = production[t] * production_cost
    cost += holding + stockout + prod_cost
    results.append([
        t+1, seasonal_demand[t], production[t], inventory[t], holding, stockout, prod_cost
    ])
    prev_inventory = inventory[t]

df = pd.DataFrame(results, columns=[
    'Ay', 'Talep', 'Üretim', 'Stok', 'Stok Maliyeti', 'Stoksuzluk Maliyeti', 'Üretim Maliyeti'
])
print(df.to_string(index=False))
print(f'\nToplam Maliyet: {cost:,.2f} TL')

# Grafiksel çıktı
try:
    import matplotlib.pyplot as plt
except ImportError:
    print('matplotlib kütüphanesi eksik. Kurmak için: pip install matplotlib')
    exit(1)
months_list = list(range(1, months+1))
plt.figure(figsize=(12,6))
plt.plot(months_list, seasonal_demand, marker='o', label='Talep', color='orange')
plt.bar(months_list, production, color='skyblue', label='Üretim', alpha=0.7)
plt.plot(months_list, inventory, marker='d', label='Stok', color='red')
plt.xlabel('Ay')
plt.ylabel('Adet')
plt.title('Mevsimsellik ve Talep Dalgaları Modeli Sonuçları')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

