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
hiring_cost = params['costs']['hiring_cost']
firing_cost = params['costs']['firing_cost']
hourly_wage = params['costs']['hourly_wage']
daily_hours = params['workforce']['daily_hours']
labor_per_unit = params['workforce']['labor_per_unit']
min_workers = params['workforce']['workers']
max_workers = params['workforce']['max_workers']
max_workforce_change = params['workforce']['max_workforce_change']


def solve_model(
    demand,
    holding_cost,
    stockout_cost,
    production_cost,
    labor_per_unit,
    hourly_wage,
    daily_hours,
    working_days,
    hiring_cost,
    firing_cost,
    min_workers,
    max_workers,
    max_workforce_change
):
    months = len(demand)
    model = pulp.LpProblem('Mevsimsel_Stok_Optimizasyonu_Dinamik', pulp.LpMinimize)

    # Karar değişkenleri
    y_production = [pulp.LpVariable(f'production_{t}', lowBound=0, cat='Integer') for t in range(months)]
    y_inventory = [pulp.LpVariable(f'inventory_{t}', lowBound=0, cat='Integer') for t in range(months)]
    y_stockout = [pulp.LpVariable(f'stockout_{t}', lowBound=0, cat='Integer') for t in range(months)]
    y_workers = [pulp.LpVariable(f'workers_{t}', lowBound=min_workers, upBound=max_workers, cat='Integer') for t in range(months)]
    y_hire = [pulp.LpVariable(f'hire_{t}', lowBound=0, cat='Integer') for t in range(months)]
    y_fire = [pulp.LpVariable(f'fire_{t}', lowBound=0, cat='Integer') for t in range(months)]

    # Amaç fonksiyonu
    model += (
        pulp.lpSum([
            y_production[t] * production_cost +
            y_inventory[t] * holding_cost +
            y_stockout[t] * stockout_cost +
            y_workers[t] * working_days[t] * daily_hours * hourly_wage +
            y_hire[t] * hiring_cost +
            y_fire[t] * firing_cost
            for t in range(months)
        ])
    )

    for t in range(months):
        # Üretim kapasitesi: işçi sayısı * toplam çalışma saati / birim başına işçilik
        model += y_production[t] <= y_workers[t] * working_days[t] * daily_hours / labor_per_unit
        # Stok ve stoksuzluk denklemi
        if t == 0:
            prev_inventory = 0
        else:
            prev_inventory = y_inventory[t-1]
        model += prev_inventory + y_production[t] + y_stockout[t] == demand[t] + y_inventory[t]
        model += y_inventory[t] >= 0
        model += y_stockout[t] >= 0
        # İşgücü değişim denklemi
        if t == 0:
            model += y_workers[t] == min_workers + y_hire[t] - y_fire[t]
        else:
            model += y_workers[t] == y_workers[t-1] + y_hire[t] - y_fire[t]
        # Maksimum işgücü değişimi kısıtı
        if t > 0:
            model += y_hire[t] + y_fire[t] <= max_workforce_change

    # Modeli çöz
    solver = pulp.PULP_CBC_CMD(msg=0)
    model.solve(solver)

    # Sonuçları topla
    results = []
    total_holding = 0
    total_stockout = 0
    total_production_cost = 0
    total_labor_cost = 0
    total_hiring_cost = 0
    total_firing_cost = 0
    total_produced = 0
    total_unfilled = 0
    total_workers = 0

    for t in range(months):
        production = int(y_production[t].varValue)
        inventory = int(y_inventory[t].varValue)
        stockout = int(y_stockout[t].varValue)
        workers = int(y_workers[t].varValue)
        hire = int(y_hire[t].varValue)
        fire = int(y_fire[t].varValue)
        labor_cost = workers * working_days[t] * daily_hours * hourly_wage
        holding = inventory * holding_cost
        stockout_cost_val = stockout * stockout_cost
        prod_cost = production * production_cost
        hiring = hire * hiring_cost
        firing = fire * firing_cost

        total_holding += holding
        total_stockout += stockout_cost_val
        total_production_cost += prod_cost
        total_labor_cost += labor_cost
        total_hiring_cost += hiring
        total_firing_cost += firing
        total_produced += production
        total_unfilled += stockout
        total_workers += workers

        results.append([
            t+1,
            demand[t],
            production,
            inventory,
            stockout,
            workers,
            hire,
            fire,
            holding,
            stockout_cost_val,
            prod_cost,
            labor_cost,
            hiring,
            firing
        ])

    df = pd.DataFrame(results, columns=[
        'Ay', 'Talep', 'Üretim', 'Stok', 'Stoksuzluk', 'İşçi', 'İşe Alım', 'İşten Çıkarma',
        'Stok Maliyeti', 'Stoksuzluk Maliyeti', 'Üretim Maliyeti', 'İşçilik Maliyeti', 'İşe Alım Maliyeti', 'İşten Çıkarma Maliyeti'
    ])

    total_cost = pulp.value(model.objective)
    total_demand = sum(demand)

    return {
        'df': df,
        'total_cost': total_cost,
        'total_holding': total_holding,
        'total_stockout': total_stockout,
        'total_production_cost': total_production_cost,
        'total_labor_cost': total_labor_cost,
        'total_hiring_cost': total_hiring_cost,
        'total_firing_cost': total_firing_cost,
        'total_produced': total_produced,
        'total_unfilled': total_unfilled,
        'total_demand': total_demand,
        'y_production': y_production,
        'y_inventory': y_inventory,
        'y_stockout': y_stockout,
        'y_workers': y_workers,
        'y_hire': y_hire,
        'y_fire': y_fire
    }

# Modeli çalıştır
model_results = solve_model(
    demand, holding_cost, stockout_cost, production_cost,
    labor_per_unit, hourly_wage, daily_hours, working_days,
    hiring_cost, firing_cost, min_workers, max_workers, max_workforce_change
)

df = model_results['df']
total_cost = model_results['total_cost']

y_production = model_results['y_production']
y_inventory = model_results['y_inventory']
y_stockout = model_results['y_stockout']
y_workers = model_results['y_workers']
y_hire = model_results['y_hire']
y_fire = model_results['y_fire']

print(f"Mevsimsel talebe göre optimize edilen işçi sayıları ve maliyetler:")

# Format cost columns
for col in ['Stok Maliyeti', 'Stoksuzluk Maliyeti', 'Üretim Maliyeti', 'İşçilik Maliyeti', 'İşe Alım Maliyeti', 'İşten Çıkarma Maliyeti']:
    df[col] = df[col].apply(lambda x: f'{int(x):,} TL')
print(tabulate(df, headers='keys', tablefmt='fancy_grid', showindex=False, numalign='right', stralign='center'))

# Print the total cost
print(f'\nToplam Maliyet: {total_cost:,.2f} TL')

def ayrintili_toplam_maliyetler(total_holding, total_stockout, total_production_cost, total_labor_cost, total_hiring_cost, total_firing_cost=0):
    return {
        'total_holding': total_holding,
        'total_stockout': total_stockout,
        'total_production_cost': total_production_cost,
        'total_labor_cost': total_labor_cost,
        'total_hiring_cost': total_hiring_cost,
        'total_firing_cost': total_firing_cost
    }

def birim_maliyet_analizi(total_demand, total_produced, total_unfilled, total_cost, total_labor_cost, total_production_cost, total_holding, total_stockout, total_hiring_cost=0, total_firing_cost=0):
    if total_produced > 0:
        avg_unit_cost = total_cost / total_produced
        avg_labor_unit = total_labor_cost / total_produced
        avg_prod_unit = total_production_cost / total_produced
        avg_hiring_unit = total_hiring_cost / total_produced if total_hiring_cost else 0
        avg_firing_unit = total_firing_cost / total_produced if total_firing_cost else 0
        avg_holding_unit = total_holding / total_produced
        avg_stockout_unit = total_stockout / total_produced
        other_costs = total_holding + total_stockout + total_hiring_cost + total_firing_cost
        avg_other_unit = other_costs / total_produced
    else:
        avg_unit_cost = avg_labor_unit = avg_prod_unit = avg_hiring_unit = avg_firing_unit = avg_holding_unit = avg_stockout_unit = avg_other_unit = 0
    return {
        'total_demand': total_demand,
        'total_produced': total_produced,
        'total_unfilled': total_unfilled,
        'avg_unit_cost': avg_unit_cost,
        'avg_labor_unit': avg_labor_unit,
        'avg_prod_unit': avg_prod_unit,
        'avg_hiring_unit': avg_hiring_unit,
        'avg_firing_unit': avg_firing_unit,
        'avg_holding_unit': avg_holding_unit,
        'avg_stockout_unit': avg_stockout_unit,
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

detay = ayrintili_toplam_maliyetler(total_holding, total_stockout, total_production_cost, total_labor_cost, total_hiring_cost, model_results['total_firing_cost'])
print(f"\nAyrıntılı Toplam Maliyetler:")
print(f"- Stok Maliyeti Toplamı: {detay['total_holding']:,.2f} TL")
print(f"- Stoksuzluk Maliyeti Toplamı: {detay['total_stockout']:,.2f} TL")
print(f"- Üretim Maliyeti Toplamı: {detay['total_production_cost']:,.2f} TL")
print(f"- İşçilik Maliyeti Toplamı: {detay['total_labor_cost']:,.2f} TL")
print(f"- İşe Alım Maliyeti Toplamı: {detay['total_hiring_cost']:,.2f} TL")
print(f"- İşten Çıkarma Maliyeti Toplamı: {detay['total_firing_cost']:,.2f} TL")

birim = birim_maliyet_analizi(
    total_demand,
    total_produced,
    total_unfilled,
    total_cost,
    total_labor_cost,
    total_production_cost,
    total_holding,
    total_stockout,
    total_hiring_cost,
    model_results['total_firing_cost']
)
print(f"\nBirim Maliyet Analizi:")
print(f"- Toplam Talep: {birim['total_demand']:,} birim")
print(f"- Toplam Üretim: {birim['total_produced']:,} birim ({birim['total_produced']/birim['total_demand']*100:.2f}%)")
print(f"- Karşılanmayan Talep: {birim['total_unfilled']:,} birim ({birim['total_unfilled']/birim['total_demand']*100:.2f}%)")
if birim['total_produced'] > 0:
    print(f"- Ortalama Birim Maliyet (Toplam): {birim['avg_unit_cost']:.2f} TL/birim")
    print(f"- Ortalama İşçilik Birim Maliyeti: {birim['avg_labor_unit']:.2f} TL/birim")
    print(f"- Ortalama Üretim Birim Maliyeti: {birim['avg_prod_unit']:.2f} TL/birim")
    print(f"- Ortalama İşe Alım Birim Maliyeti: {birim['avg_hiring_unit']:.2f} TL/birim")
    print(f"- Ortalama İşten Çıkarma Birim Maliyeti: {birim['avg_firing_unit']:.2f} TL/birim")
    print(f"- Ortalama Stok Birim Maliyeti: {birim['avg_holding_unit']:.2f} TL/birim")
    print(f"- Ortalama Stoksuzluk Birim Maliyeti: {birim['avg_stockout_unit']:.2f} TL/birim")
    print(f"- Diğer Maliyetler: {birim['avg_other_unit']:.2f} TL/birim")
else:
    print("- Ortalama Birim Maliyet: Hesaplanamadı (0 birim üretildi)")

# Grafiksel çıktı
try:
    import matplotlib.pyplot as plt
except ImportError:
    print('matplotlib kütüphanesi eksik. Kurmak için: pip install matplotlib')
    exit(1)
months_list = list(range(1, months+1))
plt.figure(figsize=(14,7))
plt.plot(months_list, demand, marker='o', label='Talep', color='orange')
plt.bar(months_list, df['Üretim'], color='skyblue', label='Üretim', alpha=0.7)
plt.plot(months_list, df['Stok'], marker='d', label='Stok', color='red')
plt.plot(months_list, df['Stoksuzluk'], marker='x', label='Stoksuzluk', color='black')
plt.plot(months_list, df['İşçi'], marker='s', label='İşçi Sayısı', color='green')
plt.xlabel('Ay')
plt.ylabel('Adet / Kişi')
plt.title('Mevsimsellik ve Dinamik İşgücü ile Stok Optimizasyonu Sonuçları')
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
        labor_per_unit, hourly_wage, daily_hours, working_days,
        hiring_cost, firing_cost, min_workers, max_workers, max_workforce_change
    )

    # Extract results
    total_cost = model_results['total_cost']
    total_labor_cost = model_results['total_labor_cost']
    total_production_cost = model_results['total_production_cost']
    total_holding = model_results['total_holding']
    total_stockout = model_results['total_stockout']
    total_hiring_cost = model_results['total_hiring_cost']
    total_firing_cost = model_results['total_firing_cost']
    total_produced = model_results['total_produced']
    total_unfilled = model_results['total_unfilled']
    total_demand = model_results['total_demand']

    # Calculate unit costs
    if total_produced > 0:
        avg_unit_cost = total_cost / total_produced
        avg_labor_unit = total_labor_cost / total_produced
        avg_prod_unit = total_production_cost / total_produced
        avg_other_unit = (total_holding + total_stockout + total_hiring_cost + total_firing_cost) / total_produced
    else:
        avg_unit_cost = avg_labor_unit = avg_prod_unit = avg_other_unit = 0

    return {
        "Toplam Maliyet": total_cost,
        "İşçilik Maliyeti": total_labor_cost,
        "Üretim Maliyeti": total_production_cost,
        "Stok Maliyeti": total_holding,
        "Stoksuzluk Maliyeti": total_stockout,
        "İşe Alım Maliyeti": total_hiring_cost,
        "İşten Çıkarma Maliyeti": total_firing_cost,
        "Toplam Talep": total_demand,
        "Toplam Üretim": total_produced,
        "Karşılanmayan Talep": total_unfilled,
        "Ortalama Birim Maliyet": avg_unit_cost,
        "İşçilik Birim Maliyeti": avg_labor_unit,
        "Üretim Birim Maliyeti": avg_prod_unit,
        "Diğer Birim Maliyetler": avg_other_unit
    }