import pulp
import numpy as np
import pandas as pd

# Parametreler
demand = [1650, 2500, 1300, 1300, 1100, 1400]
months = len(demand)
holding_cost = 5
stockout_cost = 20
labor_per_unit = 0.5
internal_production_cost = 10
cost_supplier_A = 12
cost_supplier_B = 15
capacity_supplier_A = 100
capacity_supplier_B = 1000  # Tedarikçi B kapasitesi (örnek varsayılan)
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

# Determine which supplier is cheaper
if cost_supplier_A <= cost_supplier_B:
    cheaper_supplier = 'A'
    cheaper_cost = cost_supplier_A
    cheaper_capacity = capacity_supplier_A
    expensive_supplier = 'B'
    expensive_cost = cost_supplier_B
    expensive_capacity = capacity_supplier_B
else:
    cheaper_supplier = 'B'
    cheaper_cost = cost_supplier_B
    cheaper_capacity = capacity_supplier_B
    expensive_supplier = 'A'
    expensive_cost = cost_supplier_A
    expensive_capacity = capacity_supplier_A

# Kısıtlar
for t in range(months):
    # Talep karşılanmalı
    if t == 0:
        prev_inventory = 0
    else:
        prev_inventory = inventory[t-1]
    decision_model += (internal_production[t] + out_A[t] + out_B[t] + prev_inventory - demand[t] == inventory[t])
    decision_model += (internal_production[t] <= max_internal_production)
    # Always use the cheaper supplier first, up to its capacity and remaining demand
    if cost_supplier_A <= cost_supplier_B:
        # A is cheaper
        decision_model += (out_A[t] <= capacity_supplier_A)
        decision_model += (out_A[t] <= demand[t] - internal_production[t])
        decision_model += (out_A[t] >= 0)
        decision_model += (out_B[t] == demand[t] - internal_production[t] - out_A[t])
        decision_model += (out_B[t] >= 0)
    else:
        # B is cheaper
        decision_model += (out_B[t] <= capacity_supplier_B)
        decision_model += (out_B[t] <= demand[t] - internal_production[t])
        decision_model += (out_B[t] >= 0)
        decision_model += (out_A[t] == demand[t] - internal_production[t] - out_B[t])
        decision_model += (out_A[t] >= 0)

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

