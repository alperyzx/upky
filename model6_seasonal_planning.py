import pulp
import numpy as np
import pandas as pd
from tabulate import tabulate
import yaml
import os

# parametreler.yaml dosyasını oku
with open(os.path.join(os.path.dirname(__file__), 'parametreler.yaml'), 'r', encoding='utf-8') as f:
    params = yaml.safe_load(f)

demand = np.array(params['demand']['seasonal'])
working_days = np.array(params['workforce']['working_days'])
months = len(demand)
holding_cost = params['costs']['holding_cost']
stockout_cost = params['costs']['stockout_cost']
production_cost = params['costs']['production_cost']
labor_per_unit = params['workforce']['labor_per_unit']
hiring_cost = params['costs']['hiring_cost']
daily_hours = params['workforce']['daily_hours']
hourly_wage = params['costs']['hourly_wage']

def solve_model(
    demand,
    holding_cost,
    stockout_cost,
    production_cost,
    labor_per_unit,
    hourly_wage,
    daily_hours
):
    """
    Core model logic for seasonal planning model (Model 6)
    Returns calculated values as a dictionary
    """
    months = len(demand)

    # Convert demand to numpy array if it's not already
    demand = np.array(demand)

    # Calculate required capacity and workers
    max_production = int(demand.mean() + demand.std())
    needed_workers = int(np.ceil(max_production * labor_per_unit / (daily_hours * np.mean(working_days))))
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
            y_stockout[t] * stockout_cost +
            monthly_labor_cost
            for t in range(months)
        ])
        + hiring_cost * needed_workers
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

    # Results collection
    results = []
    total_holding = 0
    total_stockout = 0
    total_production_cost = 0
    total_labor_cost = 0
    total_produced = 0
    total_unfilled = 0

    for t in range(months):
        production = int(y_production[t].varValue)
        inventory = int(y_inventory[t].varValue)
        stockout = int(y_stockout[t].varValue)
        labor_cost = monthly_labor_cost
        holding = inventory * holding_cost
        stockout_cost_val = stockout * stockout_cost
        prod_cost = production * production_cost

        # Toplam maliyetleri hesapla
        total_holding += holding
        total_stockout += stockout_cost_val
        total_production_cost += prod_cost
        total_labor_cost += labor_cost
        total_produced += production
        total_unfilled += stockout

        results.append([
            t+1,
            demand[t],
            production,
            inventory,
            stockout,
            holding,
            stockout_cost_val,
            prod_cost,
            labor_cost
        ])

    df = pd.DataFrame(results, columns=[
        'Ay', 'Talep', 'Üretim', 'Stok', 'Stoksuzluk', 'Stok Maliyeti', 'Stoksuzluk Maliyeti', 'Üretim Maliyeti', 'İşçilik Maliyeti'
    ])

    # Calculate the total cost from the model objective value
    total_cost = pulp.value(model.objective)
    total_hiring_cost = hiring_cost * needed_workers
    total_demand = sum(demand)

    return {
        'df': df,
        'total_cost': total_cost,
        'total_holding': total_holding,
        'total_stockout': total_stockout,
        'total_production_cost': total_production_cost,
        'total_labor_cost': total_labor_cost,
        'total_hiring_cost': total_hiring_cost,
        'total_produced': total_produced,
        'total_unfilled': total_unfilled,
        'total_demand': total_demand,
        'needed_workers': needed_workers,
        'max_production': max_production,
        'y_production': y_production,
        'y_inventory': y_inventory,
        'y_stockout': y_stockout,
    }

# Use the shared solver function
model_results = solve_model(
    demand, holding_cost, stockout_cost, production_cost,
    labor_per_unit, hourly_wage, daily_hours
)

max_production = model_results['max_production']
needed_workers = model_results['needed_workers']
df = model_results['df']
total_cost = model_results['total_cost']
y_production = model_results['y_production']
y_inventory = model_results['y_inventory']
y_stockout = model_results['y_stockout']

print(f"Optimum üretim kapasitesi için gereken işçi sayısı: {needed_workers}")

# Format cost columns
for col in ['Stok Maliyeti', 'Stoksuzluk Maliyeti', 'Üretim Maliyeti', 'İşçilik Maliyeti']:
    df[col] = df[col].apply(lambda x: f'{int(x):,} TL')
print(tabulate(df, headers='keys', tablefmt='fancy_grid', showindex=False, numalign='right', stralign='center'))

# Print the total cost
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
total_holding = model_results['total_holding']
total_stockout = model_results['total_stockout']
total_production_cost = model_results['total_production_cost']
total_labor_cost = model_results['total_labor_cost']
total_hiring_cost = model_results['total_hiring_cost']
total_produced = model_results['total_produced']
total_unfilled = model_results['total_unfilled']
total_demand = model_results['total_demand']

detay = ayrintili_toplam_maliyetler(total_holding, total_stockout, total_production_cost, total_labor_cost, total_hiring_cost)
print(f"\nAyrıntılı Toplam Maliyetler:")
print(f"- Stok Maliyeti Toplamı: {detay['total_holding']:,.2f} TL")
print(f"- Stoksuzluk Maliyeti Toplamı: {detay['total_stockout']:,.2f} TL")
print(f"- Üretim Maliyeti Toplamı: {detay['total_production_cost']:,.2f} TL")
print(f"- İşçilik Maliyeti Toplamı: {detay['total_labor_cost']:,.2f} TL")
print(f"- İşe Alım Maliyeti Toplamı: {detay['total_hiring_cost']:,.2f} TL")

birim = birim_maliyet_analizi(
    total_demand,
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
plt.bar(months_list, df['Üretim'], color='skyblue', label='Üretim', alpha=0.7)
plt.plot(months_list, df['Stok'], marker='d', label='Stok', color='red')
plt.plot(months_list, df['Stoksuzluk'], marker='x', label='Stoksuzluk', color='black')
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
    labor_per_unit=labor_per_unit,
    hourly_wage=hourly_wage,
    daily_hours=daily_hours
):
    # Use the shared model solver function
    model_results = solve_model(
        demand, holding_cost, stockout_cost, production_cost,
        labor_per_unit, hourly_wage, daily_hours
    )

    # Extract results
    total_cost = model_results['total_cost']
    total_labor_cost = model_results['total_labor_cost']
    total_production_cost = model_results['total_production_cost']
    total_holding = model_results['total_holding']
    total_stockout = model_results['total_stockout']
    total_hiring_cost = model_results['total_hiring_cost']
    total_produced = model_results['total_produced']
    total_unfilled = model_results['total_unfilled']
    total_demand = model_results['total_demand']

    # Calculate unit costs
    if total_produced > 0:
        avg_unit_cost = total_cost / total_produced
        avg_labor_unit = total_labor_cost / total_produced
        avg_prod_unit = total_production_cost / total_produced
        avg_other_unit = (total_holding + total_stockout) / total_produced
    else:
        avg_unit_cost = avg_labor_unit = avg_prod_unit = avg_other_unit = 0

    return {
        "Toplam Maliyet": total_cost,
        "İşçilik Maliyeti": total_labor_cost,
        "Üretim Maliyeti": total_production_cost,
        "Stok Maliyeti": total_holding,
        "Stoksuzluk Maliyeti": total_stockout,
        "İşe Alım Maliyeti": total_hiring_cost,
        "İşten Çıkarma Maliyeti": 0,
        "Toplam Talep": total_demand,
        "Toplam Üretim": total_produced,
        "Karşılanmayan Talep": total_unfilled,
        "Ortalama Birim Maliyet": avg_unit_cost,
        "İşçilik Birim Maliyeti": avg_labor_unit,
        "Üretim Birim Maliyeti": avg_prod_unit,
        "Diğer Birim Maliyetler": avg_other_unit
    }
