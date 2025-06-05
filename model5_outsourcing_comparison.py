import pulp
import numpy as np
import pandas as pd

# Parametreler
demand = [350, 330, 350, 300, 250, 900, 650, 250, 750, 250, 900, 550]
working_days = [22, 20, 23, 19, 21, 19, 22, 22, 22, 21, 21, 21]
months = len(demand)
holding_cost = 5
stockout_cost = 80  # Karşılanmayan talep maliyetini yükseltiyoruz ki tedarikçiler kullanılsın

# Tedarikçi özellikleri
cost_supplier_A = 50  # Düşük maliyetli tedarikçi (TL/birim)
cost_supplier_B = 80  # Yüksek maliyetli tedarikçi (TL/birim)
capacity_supplier_A = 300  # Tedarikçi A'nın sınırlı kapasitesi
# Tedarikçi B'nin sınırsız kapasitesi (pratikte çok büyük bir değer)
capacity_supplier_B = 99999

# Check that demand and working_days have the same length
if len(demand) != len(working_days):
    raise ValueError(f"Length of demand ({len(demand)}) and working_days ({len(working_days)}) must be equal.")

# Model
decision_model = pulp.LpProblem('Dis_Kaynak_Karsilastirma', pulp.LpMinimize)

# Karar değişkenleri
out_A = [pulp.LpVariable(f'out_A_{t}', lowBound=0, cat='Integer') for t in range(months)]
out_B = [pulp.LpVariable(f'out_B_{t}', lowBound=0, cat='Integer') for t in range(months)]
inventory = [pulp.LpVariable(f'inventory_{t}', lowBound=0, cat='Integer') for t in range(months)]
stockout = [pulp.LpVariable(f'stockout_{t}', lowBound=0, cat='Integer') for t in range(months)]

# Amaç fonksiyonu
decision_model += pulp.lpSum([
    cost_supplier_A * out_A[t] +
    cost_supplier_B * out_B[t] +
    holding_cost * inventory[t] +
    stockout_cost * stockout[t]
    for t in range(months)
])

# Kısıtlar
for t in range(months):
    # Talep karşılanmalı (tedarik + önceki stok + karşılanmayan talep = talep + yeni stok)
    if t == 0:
        prev_inventory = 0
    else:
        prev_inventory = inventory[t-1]
    decision_model += (out_A[t] + out_B[t] + prev_inventory + stockout[t] == demand[t] + inventory[t])

    # Tedarikçi kapasiteleri
    decision_model += (out_A[t] <= capacity_supplier_A)  # A tedarikçisi sınırlı kapasite
    # B tedarikçisi için kapasite sınırı yok (pratikte çok büyük bir değer)

# Modeli çöz
solver = pulp.PULP_CBC_CMD(msg=0)
decision_model.solve(solver)

# Sonuçlar
def print_results():
    from tabulate import tabulate
    table = []
    total_cost_A = 0
    total_cost_B = 0
    total_holding = 0
    total_stockout = 0

    for t in range(months):
        cost_A = int(out_A[t].varValue) * cost_supplier_A
        cost_B = int(out_B[t].varValue) * cost_supplier_B
        holding = int(inventory[t].varValue) * holding_cost
        stockout_val = int(stockout[t].varValue) * stockout_cost

        total_cost_A += cost_A
        total_cost_B += cost_B
        total_holding += holding
        total_stockout += stockout_val

        table.append([
            t+1,
            int(out_A[t].varValue),
            int(out_B[t].varValue),
            int(inventory[t].varValue),
            int(stockout[t].varValue),
            f"{cost_A:,} TL",
            f"{cost_B:,} TL",
            f"{holding:,} TL",
            f"{stockout_val:,} TL"
        ])

    headers = ['Ay', 'Tedarikçi A', 'Tedarikçi B', 'Stok', 'Karşılanmayan Talep',
              'Tedarikçi A Maliyeti (₺)', 'Tedarikçi B Maliyeti (₺)',
              'Stok Maliyeti (₺)', 'Stoksuzluk Maliyeti (₺)']

    print(f"Tedarikçi A: {cost_supplier_A} TL/birim, Kapasite: {capacity_supplier_A} birim/ay")
    print(f"Tedarikçi B: {cost_supplier_B} TL/birim, Kapasite: Sınırsız")
    print(tabulate(table, headers, tablefmt='fancy_grid', numalign='right', stralign='center'))
    print(f'\nToplam Maliyet: {pulp.value(decision_model.objective):,.2f} TL')
    print(f"\nAyrıntılı Maliyet Dağılımı:")
    print(f"- Tedarikçi A Toplam Maliyet: {total_cost_A:,} TL")
    print(f"- Tedarikçi B Toplam Maliyet: {total_cost_B:,} TL")
    print(f"- Stok Tutma Toplam Maliyet: {total_holding:,} TL")
    print(f"- Karşılanmayan Talep Toplam Maliyet: {total_stockout:,} TL")

    # Birim maliyet hesaplaması
    total_demand = sum(demand)
    total_fulfilled = sum([int(out_A[t].varValue) + int(out_B[t].varValue) for t in range(months)])
    total_cost = pulp.value(decision_model.objective)

    print(f"\nBirim Maliyet Analizi:")
    print(f"- Toplam Talep: {total_demand:,} birim")
    print(f"- Karşılanan Talep: {total_fulfilled:,} birim ({total_fulfilled/total_demand*100:.2f}%)")
    print(f"- Ortalama Birim Maliyet: {total_cost/total_fulfilled:.2f} TL/birim") if total_fulfilled > 0 else print("- Ortalama Birim Maliyet: Hesaplanamadı (0 birim karşılandı)")
    print(f"- Tedarikçi A Birim Maliyet: {cost_supplier_A:.2f} TL/birim")
    print(f"- Tedarikçi B Birim Maliyet: {cost_supplier_B:.2f} TL/birim")

    # Grafiksel çıktı
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib kütüphanesi eksik. Kurmak için: pip install matplotlib')
        return

    months_list = [t+1 for t in range(months)]
    plt.figure(figsize=(12,6))

    # Tedarikçi kullanımını gösteren yığılmış sütun grafiği
    plt.bar(months_list, [int(out_A[t].varValue) for t in range(months)],
            color='#3498db', label='Tedarikçi A (Düşük Maliyet)', alpha=0.7)
    plt.bar(months_list, [int(out_B[t].varValue) for t in range(months)],
            bottom=[int(out_A[t].varValue) for t in range(months)],
            color='#e74c3c', label='Tedarikçi B (Yüksek Maliyet)', alpha=0.7)

    # Talep çizgisi
    plt.plot(months_list, demand, marker='o', label='Talep', color='#2c3e50',
             linewidth=2, linestyle='--')

    # Stok ve karşılanmayan talep
    plt.plot(months_list, [int(inventory[t].varValue) for t in range(months)],
             marker='s', label='Stok', color='#27ae60', linewidth=2)
    plt.plot(months_list, [int(stockout[t].varValue) for t in range(months)],
             marker='x', label='Karşılanmayan Talep', color='#8e44ad', linewidth=2)

    # Tedarikçi A kapasitesi
    plt.axhline(y=capacity_supplier_A, color='#f39c12', linestyle='-.',
                label=f'Tedarikçi A Kapasite Limiti ({capacity_supplier_A})')

    plt.xlabel('Ay')
    plt.ylabel('Adet')
    plt.title('Dış Kaynak Kullanımı Karşılaştırması')
    plt.legend(loc='upper left')
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
