import numpy as np
import pandas as pd

# Parametreler
demand = np.array([1000, 1200, 1500, 1300, 1100, 1400])
working_days = np.array([22, 20, 23, 21, 22, 20])
holding_cost = 5
stockout_cost = 20
fixed_workers = 10
production_rate = 1  # adet/saat
daily_hours = 8
months = len(demand)

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
    cost += holding + stockout
    results.append([
        t+1, production[t], inventory[t], holding, stockout
    ])
    prev_inventory = inventory[t]

df = pd.DataFrame(results, columns=[
    'Ay', 'Üretim', 'Stok', 'Stok Maliyeti', 'Stoksuzluk Maliyeti'
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
plt.figure(figsize=(10,6))
plt.bar(months_list, production, color='skyblue', label='Üretim', alpha=0.7)
plt.plot(months_list, inventory, marker='d', label='Stok', color='red')
plt.xlabel('Ay')
plt.ylabel('Adet')
plt.title('Toplu Üretim ve Stoklama Modeli Sonuçları')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

