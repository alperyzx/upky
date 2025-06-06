import pulp
import numpy as np
import pandas as pd

# Parametreler
demand = [350, 330, 350, 380, 250, 900, 650, 250, 750, 250, 900, 550]
working_days = [22, 20, 23, 19, 21, 19, 22, 22, 22, 21, 21, 21]
months = len(demand)
holding_cost = 5
stockout_cost = 80  # Karşılanmayan talep maliyetini yükseltiyoruz ki tedarikçiler kullanılsın

# Tedarikçi özellikleri
cost_supplier_A = 50  # Düşük maliyetli tedarikçi (TL/birim)
cost_supplier_B = 75  # Yüksek maliyetli tedarikçi (TL/birim)
capacity_supplier_A = 300  # Tedarikçi A'nın sınırlı kapasitesi
# Tedarikçi B'nin sınırsız kapasitesi (pratikte çok büyük bir değer)
capacity_supplier_B = 9999

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
def ayrintili_toplam_maliyetler(total_cost_A, total_cost_B, total_holding, total_stockout):
    return {
        'total_cost_A': total_cost_A,
        'total_cost_B': total_cost_B,
        'total_holding': total_holding,
        'total_stockout': total_stockout
    }

def birim_maliyet_analizi(total_demand, total_fulfilled, total_cost, cost_supplier_A, cost_supplier_B):
    return {
        'total_demand': total_demand,
        'total_fulfilled': total_fulfilled,
        'avg_unit_cost': total_cost/total_fulfilled if total_fulfilled > 0 else 0,
        'cost_supplier_A': cost_supplier_A,
        'cost_supplier_B': cost_supplier_B
    }

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
    # Ayrıntılı Toplam Maliyetler fonksiyonundan alınır
    detay = ayrintili_toplam_maliyetler(total_cost_A, total_cost_B, total_holding, total_stockout)
    print(f"\nAyrıntılı Maliyet Dağılımı:")
    print(f"- Tedarikçi A Toplam Maliyet: {detay['total_cost_A']:,} TL")
    print(f"- Tedarikçi B Toplam Maliyet: {detay['total_cost_B']:,} TL")
    print(f"- Stok Tutma Toplam Maliyet: {detay['total_holding']:,} TL")
    print(f"- Karşılanmayan Talep Toplam Maliyet: {detay['total_stockout']:,} TL")
    # Birim Maliyet Analizi fonksiyonundan alınır
    total_demand = sum(demand)
    total_fulfilled = sum([int(out_A[t].varValue) + int(out_B[t].varValue) for t in range(months)])
    total_cost = pulp.value(decision_model.objective)
    birim = birim_maliyet_analizi(total_demand, total_fulfilled, total_cost, cost_supplier_A, cost_supplier_B)
    print(f"\nBirim Maliyet Analizi:")
    print(f"- Toplam Talep: {birim['total_demand']:,} birim")
    print(f"- Karşılanan Talep: {birim['total_fulfilled']:,} birim ({birim['total_fulfilled']/birim['total_demand']*100:.2f}%)")
    if birim['total_fulfilled'] > 0:
        print(f"- Ortalama Birim Maliyet: {birim['avg_unit_cost']:.2f} TL/birim")
    else:
        print("- Ortalama Birim Maliyet: Hesaplanamadı (0 birim karşılandı)")
    print(f"- Tedarikçi A Birim Maliyeti: {birim['cost_supplier_A']:.2f} TL/birim")
    print(f"- Tedarikçi B Birim Maliyeti: {birim['cost_supplier_B']:.2f} TL/birim")

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

def maliyet_analizi(
    demand=demand,
    working_days=working_days,
    holding_cost=holding_cost,
    stockout_cost=stockout_cost,
    cost_supplier_A=cost_supplier_A,
    cost_supplier_B=cost_supplier_B,
    capacity_supplier_A=capacity_supplier_A,
    capacity_supplier_B=capacity_supplier_B
):
    months = len(demand)
    # Model ve değişkenler yukarıdaki gibi tanımlanıyor
    decision_model = pulp.LpProblem('Dis_Kaynak_Karsilastirma', pulp.LpMinimize)
    out_A = [pulp.LpVariable(f'out_A_{t}', lowBound=0, cat='Integer') for t in range(months)]
    out_B = [pulp.LpVariable(f'out_B_{t}', lowBound=0, cat='Integer') for t in range(months)]
    inventory = [pulp.LpVariable(f'inventory_{t}', lowBound=0, cat='Integer') for t in range(months)]
    stockout = [pulp.LpVariable(f'stockout_{t}', lowBound=0, cat='Integer') for t in range(months)]
    decision_model += pulp.lpSum([
        cost_supplier_A * out_A[t] +
        cost_supplier_B * out_B[t] +
        holding_cost * inventory[t] +
        stockout_cost * stockout[t]
        for t in range(months)
    ])
    for t in range(months):
        if t == 0:
            prev_inventory = 0
        else:
            prev_inventory = inventory[t-1]
        decision_model += (out_A[t] + out_B[t] + prev_inventory + stockout[t] == demand[t] + inventory[t])
        decision_model += (out_A[t] <= capacity_supplier_A)
    solver = pulp.PULP_CBC_CMD(msg=0)
    decision_model.solve(solver)
    total_A = 0
    total_B = 0
    total_holding = 0
    total_stockout = 0
    total_produced = 0
    total_demand = sum(demand)
    total_unfilled = 0
    for t in range(months):
        a = int(out_A[t].varValue)
        b = int(out_B[t].varValue)
        inv = int(inventory[t].varValue)
        so = int(stockout[t].varValue)
        total_A += a
        total_B += b
        total_holding += inv * holding_cost
        total_stockout += so * stockout_cost
        total_produced += a + b
        total_unfilled += so
    toplam_maliyet = total_A * cost_supplier_A + total_B * cost_supplier_B + total_holding + total_stockout
    if total_produced > 0:
        avg_unit_cost = toplam_maliyet / total_produced
        avg_prod_unit = (total_A * cost_supplier_A + total_B * cost_supplier_B) / total_produced
        avg_other_unit = (total_holding + total_stockout) / total_produced
    else:
        avg_unit_cost = avg_prod_unit = avg_other_unit = 0
    return {
        "Toplam Maliyet": toplam_maliyet,
        "İşçilik Maliyeti": 0,
        "Üretim Maliyeti": total_A * cost_supplier_A + total_B * cost_supplier_B,
        "Stok Maliyeti": total_holding,
        "Stoksuzluk Maliyeti": total_stockout,
        "İşe Alım Maliyeti": 0,
        "İşten Çıkarma Maliyeti": 0,
        "Toplam Talep": total_demand,
        "Toplam Üretim": total_produced,
        "Karşılanmayan Talep": total_unfilled,
        "Ortalama Birim Maliyet": avg_unit_cost,
        "İşçilik Birim Maliyeti": 0,
        "Üretim Birim Maliyeti": avg_prod_unit,
        "Diğer Birim Maliyetler": avg_other_unit
    }

if __name__ == '__main__':
    try:
        import tabulate
    except ImportError:
        print('tabulate kütüphanesi eksik. Kurmak için: pip install tabulate')
        exit(1)
    print_results()
