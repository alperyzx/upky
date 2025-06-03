import pulp
import numpy as np
import pandas as pd

# Parametreler
demand = [2000, 4200, 13500, 1300, 1100, 1400]
months = len(demand)
holding_cost = 5
stockout_cost = 20
labor_per_unit = 0.5
internal_production_cost = 10
cost_supplier_A = 15
cost_supplier_B = 18
capacity_supplier_A = 500
working_days = [22, 20, 23, 21, 22, 20]
daily_hours = 8
max_internal_workers = 50
max_internal_production = 1500  # İç üretim kapasitesi (adet/ay)

# Model
decision_model = pulp.LpProblem('Dis_Kaynak_Karsilastirma', pulp.LpMinimize)

# Karar değişkenleri
internal_production = [pulp.LpVariable(f'internal_prod_{t}', lowBound=0, cat='Integer') for t in range(months)]
out_A = [pulp.LpVariable(f'out_A_{t}', lowBound=0, cat='Integer') for t in range(months)]
out_B = [pulp.LpVariable(f'out_B_{t}', lowBound=0, cat='Integer') for t in range(months)]
inventory = [pulp.LpVariable(f'inventory_{t}', lowBound=0, cat='Integer') for t in range(months)]

# Amaç fonksiyonu
decision_model += pulp.lpSum([
    internal_production_cost * internal_production[t] +
    cost_supplier_A * out_A[t] +
    cost_supplier_B * out_B[t] +
    holding_cost * inventory[t]
    for t in range(months)
])

# Kısıtlar
for t in range(months):
    # Talep karşılanmalı
    if t == 0:
        prev_inventory = 0
    else:
        prev_inventory = inventory[t-1]
    decision_model += (internal_production[t] + out_A[t] + out_B[t] + prev_inventory - demand[t] == inventory[t])
    # Tedarikçi A kapasite
    decision_model += (out_A[t] <= capacity_supplier_A)
    # İç üretim kapasitesi (sabit)
    decision_model += (internal_production[t] <= max_internal_production)

# Modeli çöz
solver = pulp.PULP_CBC_CMD(msg=0)
decision_model.solve(solver)

# Sonuçlar
def print_results():
    from tabulate import tabulate
    table = []
    for t in range(months):
        table.append([
            t+1,
            int(internal_production[t].varValue),
            int(out_A[t].varValue),
            int(out_B[t].varValue),
            int(inventory[t].varValue)
        ])
    headers = ['Ay', 'İç Üretim', 'Tedarikçi A', 'Tedarikçi B', 'Stok']
    print(tabulate(table, headers, tablefmt='github', numalign='right', stralign='center'))
    print(f'\nToplam Maliyet: {pulp.value(decision_model.objective):,.2f} TL')

    # Grafiksel çıktı
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib kütüphanesi eksik. Kurmak için: pip install matplotlib')
        return
    months_list = [t+1 for t in range(months)]
    plt.figure(figsize=(10,6))
    plt.bar(months_list, [int(internal_production[t].varValue) for t in range(months)], color='skyblue', label='İç Üretim', alpha=0.7)
    plt.bar(months_list, [int(out_A[t].varValue) for t in range(months)], bottom=[int(internal_production[t].varValue) for t in range(months)], color='orange', label='Tedarikçi A', alpha=0.7)
    bottom_sum = [int(internal_production[t].varValue) + int(out_A[t].varValue) for t in range(months)]
    plt.bar(months_list, [int(out_B[t].varValue) for t in range(months)], bottom=bottom_sum, color='green', label='Tedarikçi B', alpha=0.7)
    plt.plot(months_list, [int(inventory[t].varValue) for t in range(months)], marker='d', label='Stok', color='red', linewidth=2)
    plt.xlabel('Ay')
    plt.ylabel('Adet')
    plt.title('Dış Kaynak Kullanımı Karşılaştırması')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    try:
        import tabulate
    except ImportError:
        print('tabulate kütüphanesi eksik. Kurmak için: pip install tabulate')
        exit(1)
    print_results()

