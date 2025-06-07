import numpy as np
import pandas as pd

# Parametreler
months = 12
demand = np.array([500, 520, 500, 490, 500, 490, 500, 510, 520, 510, 510, 500])
working_days = np.array([22, 20, 23, 19, 21, 19, 22, 22, 22, 21, 21, 21])
holding_cost = 5
stockout_cost = 80
labor_per_unit = 4
daily_hours = 8
fixed_workers = 11
overtime_wage_multiplier = 1.5
max_overtime_per_worker = 20  # saat/ay
normal_hourly_wage = 10  # TL/saat
overtime_cost_per_hour = normal_hourly_wage * overtime_wage_multiplier  # Fazla mesai saatlik ücret otomatik hesaplanır
production_cost = 30  # birim üretim maliyeti (TL)
hiring_cost = 1800  # İşe alım maliyeti (TL/işçi)

# Check that demand and working_days have the same length
if len(demand) != len(working_days):
    raise ValueError(f"Length of demand ({len(demand)}) and working_days ({len(working_days)}) must be equal.")

def birim_maliyet_analizi(
    demand, production, inventory, cost, total_normal_labor, total_overtime, total_production, total_holding, total_stockout, production_cost, total_hiring_cost
):
    total_demand = sum(demand)
    total_produced = sum(production)
    total_unfilled = sum([abs(min(inventory[t], 0)) for t in range(len(demand))])
    result = {
        'total_demand': total_demand,
        'total_produced': total_produced,
        'total_unfilled': total_unfilled,
        'cost': cost,
        'avg_unit_cost': cost/total_produced if total_produced > 0 else 0,
        'labor_unit_cost': (total_normal_labor+total_overtime)/total_produced if total_produced > 0 else 0,
        'normal_labor_unit_cost': total_normal_labor/total_produced if total_produced > 0 else 0,
        'overtime_unit_cost': total_overtime/total_produced if total_produced > 0 else 0,
        'prod_unit_cost': total_production/total_produced if total_produced > 0 else 0,
        'other_unit_cost': (total_holding+total_stockout+total_hiring_cost)/total_produced if total_produced > 0 else 0
    }
    return result

def overtime_model():
    production = np.zeros(months)
    overtime_hours = np.zeros(months)
    inventory = np.zeros(months)
    cost = 0
    prev_inventory = 0
    results = []
    total_holding = 0
    total_stockout = 0
    total_overtime = 0
    total_normal_labor = 0
    total_production = 0
    total_hiring_cost = hiring_cost * fixed_workers
    for t in range(months):
        # Normal kapasiteyle üretilebilecek miktar
        normal_prod = fixed_workers * working_days[t] * daily_hours / labor_per_unit
        # Fazla mesaiyle üretilebilecek maksimum miktar
        max_overtime_total_hours = fixed_workers * max_overtime_per_worker
        max_overtime_units = max_overtime_total_hours / labor_per_unit
        remaining_demand = demand[t] - prev_inventory
        if remaining_demand <= 0:
            prod = 0
            ot_hours = 0
        elif remaining_demand <= normal_prod:
            prod = remaining_demand
            ot_hours = 0
        else:
            prod = normal_prod
            extra_needed = remaining_demand - normal_prod
            overtime_units = min(extra_needed, max_overtime_units)
            ot_hours = overtime_units * labor_per_unit
            prod += overtime_units
        production[t] = prod
        overtime_hours[t] = ot_hours
        inventory[t] = prev_inventory + prod - demand[t]
        holding = max(inventory[t], 0) * holding_cost
        stockout = abs(min(inventory[t], 0)) * stockout_cost
        overtime = max(ot_hours, 0) * overtime_cost_per_hour
        normal_labor_cost = fixed_workers * working_days[t] * daily_hours * normal_hourly_wage
        production_cost_val = prod * production_cost
        cost += holding + stockout + overtime + normal_labor_cost + production_cost_val
        total_holding += holding
        total_stockout += stockout
        total_overtime += overtime
        total_normal_labor += normal_labor_cost
        total_production += production_cost_val
        results.append([
            t+1, fixed_workers, prod, ot_hours, inventory[t], holding, stockout, overtime, normal_labor_cost, production_cost_val
        ])
        prev_inventory = inventory[t]
    headers = [
        'Ay', 'İşçi', 'Üretim', 'Fazla Mesai', 'Stok', 'Stok Maliyeti', 'Stoksuzluk Maliyeti', 'Fazla Mesai Maliyeti', 'Normal İşçilik Maliyeti', 'Üretim Maliyeti'
    ]
    df = pd.DataFrame(results, columns=headers)
    from tabulate import tabulate
    # Format cost columns
    for col in ['Stok Maliyeti', 'Stoksuzluk Maliyeti', 'Fazla Mesai Maliyeti', 'Normal İşçilik Maliyeti', 'Üretim Maliyeti']:
        df[col] = df[col].apply(lambda x: f'{int(x):,} TL')
    print(tabulate(df, headers='keys', tablefmt='fancy_grid', showindex=False, numalign='right', stralign='center'))
    print(f'\nToplam Maliyet: {cost:,.2f} TL')
    # Ayrıntılı Toplam Maliyetler fonksiyonundan alınır
    detay = ayrintili_toplam_maliyetler(total_holding, total_stockout, total_overtime, total_normal_labor, total_production, total_hiring_cost)
    print(f"\nAyrıntılı Toplam Maliyetler:")
    print(f"- Toplam Stok Maliyeti: {detay['total_holding']:,.2f} TL")
    print(f"- Toplam Stoksuzluk Maliyeti: {detay['total_stockout']:,.2f} TL")
    print(f"- Toplam Fazla Mesai Maliyeti: {detay['total_overtime']:,.2f} TL")
    print(f"- Toplam Normal İşçilik Maliyeti: {detay['total_normal_labor']:,.2f} TL")
    print(f"- Toplam Üretim Maliyeti: {detay['total_production']:,.2f} TL")
    print(f"- Toplam İşe Alım Maliyeti: {detay['total_hiring_cost']:,.2f} TL")
    # Birim maliyet analizini fonksiyon ile yap
    birim = birim_maliyet_analizi(
        demand, production, inventory, cost, total_normal_labor, total_overtime, total_production, total_holding, total_stockout, production_cost, total_hiring_cost
    )
    print(f"\nBirim Maliyet Analizi:")
    print(f"- Toplam Talep: {birim['total_demand']:,} birim")
    print(f"- Toplam Üretim: {birim['total_produced']:,} birim ({birim['total_produced']/birim['total_demand']*100:.2f}%)")
    if birim['total_unfilled'] > 0:
        print(f"- Karşılanmayan Talep: {birim['total_unfilled']:,} birim ({birim['total_unfilled']/birim['total_demand']*100:.2f}%)")
    if birim['total_produced'] > 0:
        print(f"- Ortalama Birim Maliyet (Toplam): {birim['avg_unit_cost']:.2f} TL/birim")
        print(f"- İşçilik Birim Maliyeti: {birim['labor_unit_cost']:.2f} TL/birim")
        print(f"  * Normal İşçilik: {birim['normal_labor_unit_cost']:.2f} TL/birim")
        print(f"  * Fazla Mesai: {birim['overtime_unit_cost']:.2f} TL/birim")
        print(f"- Üretim Birim Maliyeti: {birim['prod_unit_cost']:.2f} TL/birim")
        print(f"- Diğer Maliyetler (Stok, Stoksuzluk, İşe Alım): {birim['other_unit_cost']:.2f} TL/birim")
    else:
        print("- Ortalama Birim Maliyet: Hesaplanamadı (0 birim üretildi)")

    # Grafiksel çıktı
    # Grafik: Fazla Mesai saatlerini bar olarak göster
    import matplotlib.pyplot as plt
    months_list = df['Ay'].tolist()
    fig, ax1 = plt.subplots(figsize=(10,6))
    ax1.bar(months_list, df['Fazla Mesai'], color='orange', label='Fazla Mesai (saat)', alpha=0.7)
    ax1.set_xlabel('Ay')
    ax1.set_ylabel('Fazla Mesai (saat)', color='orange')
    ax1.tick_params(axis='y', labelcolor='orange')
    ax2 = ax1.twinx()
    ax2.plot(months_list, df['Üretim'], marker='s', label='Üretim', color='green')
    ax2.plot(months_list, df['Stok'], marker='d', label='Stok', color='red')
    ax2.set_ylabel('Adet', color='gray')
    ax2.tick_params(axis='y', labelcolor='gray')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    plt.title('Fazla Mesaili Üretim Modeli Sonuçları')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()
    return df, cost

def maliyet_analizi(
    demand=demand,
    working_days=working_days,
    holding_cost=holding_cost,
    stockout_cost=stockout_cost,
    labor_per_unit=labor_per_unit,
    daily_hours=daily_hours,
    fixed_workers=fixed_workers,
    overtime_wage_multiplier=overtime_wage_multiplier,
    max_overtime_per_worker=max_overtime_per_worker,
    normal_hourly_wage=normal_hourly_wage,
    production_cost=production_cost
):
    months = len(demand)
    overtime_cost_per_hour = normal_hourly_wage * overtime_wage_multiplier
    production = np.zeros(months)
    overtime_hours = np.zeros(months)
    inventory = np.zeros(months)
    prev_inventory = 0
    total_holding = 0
    total_stockout = 0
    total_overtime = 0
    total_normal_labor = 0
    total_production = 0
    total_hiring_cost = hiring_cost * fixed_workers
    total_demand = sum(demand)
    total_unfilled = 0
    for t in range(months):
        normal_prod = fixed_workers * working_days[t] * daily_hours / labor_per_unit
        max_overtime_total_hours = fixed_workers * max_overtime_per_worker
        max_overtime_units = max_overtime_total_hours / labor_per_unit
        remaining_demand = demand[t] - prev_inventory
        if remaining_demand <= 0:
            prod = 0
            ot_hours = 0
        elif remaining_demand <= normal_prod:
            prod = remaining_demand
            ot_hours = 0
        else:
            prod = normal_prod
            extra_needed = remaining_demand - normal_prod
            overtime_units = min(extra_needed, max_overtime_units)
            ot_hours = overtime_units * labor_per_unit
            prod += overtime_units
        production[t] = prod
        overtime_hours[t] = ot_hours
        inventory[t] = prev_inventory + prod - demand[t]
        holding = max(inventory[t], 0) * holding_cost
        stockout = abs(min(inventory[t], 0)) * stockout_cost
        overtime = max(ot_hours, 0) * overtime_cost_per_hour
        normal_labor_cost = fixed_workers * working_days[t] * daily_hours * normal_hourly_wage
        production_cost_val = prod * production_cost
        total_holding += holding
        total_stockout += stockout
        total_overtime += overtime
        total_normal_labor += normal_labor_cost
        total_production += prod
        prev_inventory = inventory[t]
        if inventory[t] < 0:
            total_unfilled += abs(inventory[t])
    toplam_maliyet = total_holding + total_stockout + total_overtime + total_normal_labor + total_production * production_cost + total_hiring_cost
    if total_production > 0:
        avg_unit_cost = toplam_maliyet / total_production
        avg_labor_unit = (total_normal_labor + total_overtime) / total_production
        avg_prod_unit = (total_production * production_cost) / total_production
        avg_other_unit = (total_holding + total_stockout) / total_production
    else:
        avg_unit_cost = avg_labor_unit = avg_prod_unit = avg_other_unit = 0
    return {
        "Toplam Maliyet": toplam_maliyet,
        "İşçilik Maliyeti": total_normal_labor + total_overtime,
        "Üretim Maliyeti": total_production * production_cost,
        "Stok Maliyeti": total_holding,
        "Stoksuzluk Maliyeti": total_stockout,
        "İşe Alım Maliyeti": total_hiring_cost,
        "İşten Çıkarma Maliyeti": 0,
        "Toplam Talep": total_demand,
        "Toplam Üretim": total_production,
        "Karşılanmayan Talep": total_unfilled,
        "Ortalama Birim Maliyet": avg_unit_cost,
        "İşçilik Birim Maliyeti": avg_labor_unit,
        "Üretim Birim Maliyeti": avg_prod_unit,
        "Diğer Birim Maliyetler": avg_other_unit
    }

def ayrintili_toplam_maliyetler(total_holding, total_stockout, total_overtime, total_normal_labor, total_production, total_hiring_cost):
    return {
        'total_holding': total_holding,
        'total_stockout': total_stockout,
        'total_overtime': total_overtime,
        'total_normal_labor': total_normal_labor,
        'total_production': total_production,
        'total_hiring_cost': total_hiring_cost
    }

if __name__ == '__main__':
    overtime_model()
