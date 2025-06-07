import pulp
import numpy as np
import pandas as pd
import yaml
import os
from tabulate import tabulate  # Fix: Import tabulate function directly

# parametreler.yaml dosyasını oku
with open(os.path.join(os.path.dirname(__file__), 'parametreler.yaml'), 'r', encoding='utf-8') as f:
    params = yaml.safe_load(f)

demand = params['demand']['normal']  # veya 'normal', 'seasonal' seçilebilir
working_days = params['workforce']['working_days']
months = len(demand)
holding_cost = params['costs']['holding_cost']
stockout_cost = params['costs']['stockout_cost']
cost_supplier_A = params['costs']['cost_supplier_A']
cost_supplier_B = params['costs']['cost_supplier_B']
capacity_supplier_A = params['capacity']['capacity_supplier_A']
capacity_supplier_B = params['capacity']['capacity_supplier_B']

# Check that demand and working_days have the same length
if len(demand) != len(working_days):
    raise ValueError(f"Length of demand ({len(demand)}) and working_days ({len(working_days)}) must be equal.")

def solve_model(
    demand,
    working_days,
    holding_cost,
    stockout_cost,
    cost_supplier_A,
    cost_supplier_B,
    capacity_supplier_A,
    capacity_supplier_B
):
    """
    Core model logic for the outsourcing comparison model (Model 5)
    Returns calculated values as a dictionary
    """
    months = len(demand)

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

    # Results as variables, total calculations, etc.
    total_A = 0
    total_B = 0
    total_holding = 0
    total_stockout = 0
    total_produced = 0
    total_demand = sum(demand)
    total_unfilled = 0

    results = []
    for t in range(months):
        a = int(out_A[t].varValue)
        b = int(out_B[t].varValue)
        inv = int(inventory[t].varValue)
        so = int(stockout[t].varValue)

        cost_A = a * cost_supplier_A
        cost_B = b * cost_supplier_B
        holding = inv * holding_cost
        stockout_val = so * stockout_cost

        total_A += a
        total_B += b
        total_holding += holding
        total_stockout += stockout_val
        total_produced += a + b
        total_unfilled += so

        results.append([
            t+1, a, b, inv, so, cost_A, cost_B, holding, stockout_val
        ])

    toplam_maliyet = total_A * cost_supplier_A + total_B * cost_supplier_B + total_holding + total_stockout

    df = pd.DataFrame(results, columns=[
        'Ay', 'Tedarikçi A', 'Tedarikçi B', 'Stok', 'Karşılanmayan Talep',
        'Tedarikçi A Maliyeti', 'Tedarikçi B Maliyeti', 'Stok Maliyeti', 'Stoksuzluk Maliyeti'
    ])

    return {
        'out_A': out_A,
        'out_B': out_B,
        'inventory': inventory,
        'stockout': stockout,
        'objective_value': pulp.value(decision_model.objective),
        'df': df,
        'total_cost_A': total_A * cost_supplier_A,
        'total_cost_B': total_B * cost_supplier_B,
        'total_holding': total_holding,
        'total_stockout': total_stockout,
        'total_produced': total_produced,
        'total_unfilled': total_unfilled,
        'total_demand': total_demand,
        'toplam_maliyet': toplam_maliyet
    }


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
    # Get model results by running the model
    model_results = solve_model(
        demand, working_days, holding_cost, stockout_cost, cost_supplier_A,
        cost_supplier_B, capacity_supplier_A, capacity_supplier_B
    )

    # Use the model results from the shared solver
    df = model_results['df']
    out_A = model_results['out_A']
    out_B = model_results['out_B']
    inventory = model_results['inventory']
    stockout = model_results['stockout']
    total_cost_A = model_results['total_cost_A']
    total_cost_B = model_results['total_cost_B']
    total_holding = model_results['total_holding']
    total_stockout = model_results['total_stockout']
    toplam_maliyet = model_results['toplam_maliyet']
    total_demand = model_results['total_demand']
    total_fulfilled = model_results['total_produced']

    # Format the DataFrame for display
    table = []
    for _, row in df.iterrows():
        table.append([
            int(row['Ay']),
            int(row['Tedarikçi A']),
            int(row['Tedarikçi B']),
            int(row['Stok']),
            int(row['Karşılanmayan Talep']),
            f"{row['Tedarikçi A Maliyeti']:,} TL",
            f"{row['Tedarikçi B Maliyeti']:,} TL",
            f"{row['Stok Maliyeti']:,} TL",
            f"{row['Stoksuzluk Maliyeti']:,} TL"
        ])

    headers = ['Ay', 'Tedarikçi A', 'Tedarikçi B', 'Stok', 'Karşılanmayan Talep',
              'Tedarikçi A Maliyeti (₺)', 'Tedarikçi B Maliyeti (₺)',
              'Stok Maliyeti (₺)', 'Stoksuzluk Maliyeti (₺)']

    print(f"Tedarikçi A: {cost_supplier_A} TL/birim, Kapasite: {capacity_supplier_A} birim/ay")
    print(f"Tedarikçi B: {cost_supplier_B} TL/birim, Kapasite: Sınırsız")
    print(tabulate(table, headers, tablefmt='fancy_grid', numalign='right', stralign='center'))
    print(f'\nToplam Maliyet: {toplam_maliyet:,.2f} TL')

    # Ayrıntılı Toplam Maliyetler fonksiyonundan alınır
    detay = ayrintili_toplam_maliyetler(total_cost_A, total_cost_B, total_holding, total_stockout)
    print(f"\nAyrıntılı Maliyet Dağılımı:")
    print(f"- Tedarikçi A Toplam Maliyet: {detay['total_cost_A']:,} TL")
    print(f"- Tedarikçi B Toplam Maliyet: {detay['total_cost_B']:,} TL")
    print(f"- Stok Tutma Toplam Maliyet: {detay['total_holding']:,} TL")
    print(f"- Karşılanmayan Talep Toplam Maliyet: {detay['total_stockout']:,} TL")

    # Birim Maliyet Analizi fonksiyonundan alınır
    birim = birim_maliyet_analizi(total_demand, total_fulfilled, toplam_maliyet, cost_supplier_A, cost_supplier_B)
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
    # Use the shared model solver function
    model_results = solve_model(
        demand, working_days, holding_cost, stockout_cost,
        cost_supplier_A, cost_supplier_B,
        capacity_supplier_A, capacity_supplier_B
    )

    toplam_maliyet = model_results['toplam_maliyet']
    total_cost_A = model_results['total_cost_A']
    total_cost_B = model_results['total_cost_B']
    total_holding = model_results['total_holding']
    total_stockout = model_results['total_stockout']
    total_produced = model_results['total_produced']
    total_unfilled = model_results['total_unfilled']
    total_demand = model_results['total_demand']

    # Calculate unit costs
    if total_produced > 0:
        avg_unit_cost = toplam_maliyet / total_produced
        avg_prod_unit = (total_cost_A + total_cost_B) / total_produced
        avg_other_unit = (total_holding + total_stockout) / total_produced
    else:
        avg_unit_cost = avg_prod_unit = avg_other_unit = 0

    return {
        "Toplam Maliyet": toplam_maliyet,
        "İşçilik Maliyeti": 0,
        "Üretim Maliyeti": total_cost_A + total_cost_B,
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
        from tabulate import tabulate
    except ImportError:
        print('tabulate kütüphanesi eksik. Kurmak için: pip install tabulate')
        exit(1)
    print_results()