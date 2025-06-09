import numpy as np
import pandas as pd
import yaml
import os

# parametreler.yaml dosyasını oku
with open(os.path.join(os.path.dirname(__file__), 'parametreler.yaml'), 'r', encoding='utf-8') as f:
    params = yaml.safe_load(f)

demand = np.array(params['demand']['normal'])  # veya 'high', 'seasonal' seçilebilir
working_days = np.array(params['workforce']['working_days'])
holding_cost = params['costs']['holding_cost']
stockout_cost = params['costs']['stockout_cost']
labor_per_unit = params['workforce']['labor_per_unit']
daily_hours = params['workforce']['daily_hours']
workers = params['workforce']['workers']
overtime_wage_multiplier = params['costs']['overtime_wage_multiplier']
max_overtime_per_worker = params['costs']['max_overtime_per_worker']
normal_hourly_wage = params['costs']['hourly_wage']
overtime_cost_per_hour = normal_hourly_wage * overtime_wage_multiplier
production_cost = params['costs']['production_cost']
hiring_cost = params['costs']['hiring_cost']
months = len(demand)

# Check that demand and working_days have the same length
if len(demand) != len(working_days):
    raise ValueError(f"Length of demand ({len(demand)}) and working_days ({len(working_days)}) must be equal.")

def calculate_optimal_workers(demand, working_days, daily_hours, labor_per_unit, max_overtime_per_worker):
    """
    Talebe ve fazla mesai parametrelerine göre optimal işçi sayısını hesaplar

    Returns:
        int: Hesaplanan optimal işçi sayısı
    """
    total_demand = sum(demand)
    avg_working_days = sum(working_days) / len(working_days)

    # Toplam talep için gereken işçilik saati
    total_labor_hours_needed = total_demand * labor_per_unit

    # Her ay için bir işçinin çalışabileceği normal + fazla mesai saati
    monthly_hours_per_worker = (avg_working_days * daily_hours) + max_overtime_per_worker

    # Tüm planlama dönemi için işçi başına toplam çalışma saati
    total_hours_per_worker = monthly_hours_per_worker * len(demand)

    # Optimal işçi sayısı hesaplaması (yukarı yuvarlama)
    optimal_workers = int(np.ceil(total_labor_hours_needed / total_hours_per_worker))

    # En az 1 işçi olmalı
    return max(1, optimal_workers)

def solve_model(
    demand,
    working_days,
    holding_cost,
    labor_per_unit,
    workers,
    daily_hours,
    overtime_wage_multiplier,
    max_overtime_per_worker,
    stockout_cost,
    normal_hourly_wage,
    production_cost
):
    """
    Core model logic for Model 2 (Fazla Mesaili Üretim)
    Returns calculated values as a dictionary
    """
    months = len(demand)

    # İşçi sayısını optimize et - talebe göre makul bir değer hesapla
    optimal_workers = calculate_optimal_workers(
        demand, working_days, daily_hours, labor_per_unit, max_overtime_per_worker
    )

    # Kullanıcı girdisi çok fazlaysa, optimal değeri kullan
    if workers > optimal_workers * 2:
        print(f"UYARI: İşçi sayısı ({workers}) optimal seviyenin ({optimal_workers}) çok üzerinde.")
        print(f"Model verimliliği için işçi sayısı {optimal_workers} olarak güncellendi.")
        workers = optimal_workers

    overtime_cost_per_hour = normal_hourly_wage * overtime_wage_multiplier
    production = np.zeros(months)
    overtime_hours = np.zeros(months)
    inventory = np.zeros(months)
    prev_inventory = 0
    results = []
    total_cost = 0
    total_holding = 0
    total_stockout = 0
    total_overtime = 0
    total_normal_labor = 0
    total_production_cost = 0
    total_hiring_cost = hiring_cost * workers

    for t in range(months):
        # Normal kapasiteyle üretilebilecek miktar
        normal_prod = workers * working_days[t] * daily_hours / labor_per_unit
        # Fazla mesaiyle üretilebilecek maksimum miktar
        max_overtime_total_hours = workers * max_overtime_per_worker
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
        normal_labor_cost = workers * working_days[t] * daily_hours * normal_hourly_wage
        prod_cost = prod * production_cost

        total_cost += holding + stockout + overtime + normal_labor_cost + prod_cost
        total_holding += holding
        total_stockout += stockout
        total_overtime += overtime
        total_normal_labor += normal_labor_cost
        total_production_cost += prod_cost

        results.append([
            t+1, workers, prod, ot_hours, inventory[t], holding, stockout, overtime, normal_labor_cost, prod_cost
        ])
        prev_inventory = inventory[t]

    headers = [
        'Ay', 'İşçi', 'Üretim', 'Fazla Mesai', 'Stok', 'Stok Maliyeti', 'Stoksuzluk Maliyeti',
        'Fazla Mesai Maliyeti', 'Normal İşçilik Maliyeti', 'Üretim Maliyeti'
    ]
    df = pd.DataFrame(results, columns=headers)

    # Calculate total produced and unfilled demand
    total_produced = np.sum(production)
    total_unfilled = np.sum([abs(min(inv, 0)) for inv in inventory])

    return {
        'df': df,
        'production': production,
        'overtime_hours': overtime_hours,
        'inventory': inventory,
        'total_cost': total_cost,
        'total_holding': total_holding,
        'total_stockout': total_stockout,
        'total_overtime': total_overtime,
        'total_normal_labor': total_normal_labor,
        'total_production_cost': total_production_cost,
        'total_hiring_cost': total_hiring_cost,
        'total_produced': total_produced,
        'total_unfilled': total_unfilled,
        'optimal_workers': optimal_workers  # Optimal işçi sayısını sonuçlara ekle
    }

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

def print_results():
    from tabulate import tabulate

    # Use solve_model() to get the model variables
    model_results = solve_model(
        demand, working_days, holding_cost, labor_per_unit, workers,
        daily_hours, overtime_wage_multiplier, max_overtime_per_worker,
        stockout_cost, normal_hourly_wage, production_cost
    )

    # Extract variables from model_results
    df = model_results['df']
    production = model_results['production']
    overtime_hours = model_results['overtime_hours']
    inventory = model_results['inventory']
    total_cost = model_results['total_cost']

    # Format cost columns
    for col in ['Stok Maliyeti', 'Stoksuzluk Maliyeti', 'Fazla Mesai Maliyeti', 'Normal İşçilik Maliyeti', 'Üretim Maliyeti']:
        df[col] = df[col].apply(lambda x: f'{int(x):,} TL')

    print(tabulate(df, headers='keys', tablefmt='fancy_grid', showindex=False, numalign='right', stralign='center'))
    print(f'\nToplam Maliyet: {total_cost:,.2f} TL')

    # Ayrıntılı Toplam Maliyetler
    detay = ayrintili_toplam_maliyetler(
        model_results['total_holding'],
        model_results['total_stockout'],
        model_results['total_overtime'],
        model_results['total_normal_labor'],
        model_results['total_production_cost'],
        model_results['total_hiring_cost']
    )

    print(f"\nAyrıntılı Toplam Maliyetler:")
    print(f"- Toplam Stok Maliyeti: {detay['total_holding']:,.2f} TL")
    print(f"- Toplam Stoksuzluk Maliyeti: {detay['total_stockout']:,.2f} TL")
    print(f"- Toplam Fazla Mesai Maliyeti: {detay['total_overtime']:,.2f} TL")
    print(f"- Toplam Normal İşçilik Maliyeti: {detay['total_normal_labor']:,.2f} TL")
    print(f"- Toplam Üretim Maliyeti: {detay['total_production']:,.2f} TL")
    print(f"- Toplam İşe Alım Maliyeti: {detay['total_hiring_cost']:,.2f} TL")

    # Birim maliyet analizi
    birim = birim_maliyet_analizi(
        demand,
        production,
        inventory,
        total_cost,
        model_results['total_normal_labor'],
        model_results['total_overtime'],
        model_results['total_production_cost'],
        model_results['total_holding'],
        model_results['total_stockout'],
        production_cost,
        model_results['total_hiring_cost']
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
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib kütüphanesi eksik. Kurmak için: pip install matplotlib')
        return

    # Grafik: Fazla Mesai saatlerini bar olarak göster
    months_list = df['Ay'].tolist()
    fig, ax1 = plt.subplots(figsize=(10,6))
    ax1.bar(months_list, overtime_hours, color='orange', label='Fazla Mesai (saat)', alpha=0.7)
    ax1.set_xlabel('Ay')
    ax1.set_ylabel('Fazla Mesai (saat)', color='orange')
    ax1.tick_params(axis='y', labelcolor='orange')
    ax2 = ax1.twinx()
    ax2.plot(months_list, production, marker='s', label='Üretim', color='green')
    ax2.plot(months_list, inventory, marker='d', label='Stok', color='red')
    ax2.set_ylabel('Adet', color='gray')
    ax2.tick_params(axis='y', labelcolor='gray')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    plt.title('Fazla Mesaili Üretim Modeli Sonuçları')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

def overtime_model():
    # Use the print_results function instead of duplicating code
    print_results()

    # For compatibility with existing code that expects a return value from overtime_model()
    model_results = solve_model(
        demand, working_days, holding_cost, labor_per_unit, workers,
        daily_hours, overtime_wage_multiplier, max_overtime_per_worker,
        stockout_cost, normal_hourly_wage, production_cost
    )
    return model_results['df'], model_results['total_cost']

def maliyet_analizi(
    demand=demand,
    working_days=working_days,
    holding_cost=holding_cost,
    stockout_cost=stockout_cost,
    labor_per_unit=labor_per_unit,
    daily_hours=daily_hours,
    workers=workers,
    overtime_wage_multiplier=overtime_wage_multiplier,
    max_overtime_per_worker=max_overtime_per_worker,
    normal_hourly_wage=normal_hourly_wage,
    production_cost=production_cost
):
    # Use the shared model solver function
    model_results = solve_model(
        demand, working_days, holding_cost, labor_per_unit, workers,
        daily_hours, overtime_wage_multiplier, max_overtime_per_worker,
        stockout_cost, normal_hourly_wage, production_cost
    )

    total_production = model_results['total_produced']
    toplam_maliyet = model_results['total_cost']
    total_normal_labor = model_results['total_normal_labor']
    total_overtime = model_results['total_overtime']
    total_holding = model_results['total_holding']
    total_stockout = model_results['total_stockout']
    total_unfilled = model_results['total_unfilled']
    total_demand = sum(demand)
    total_hiring_cost = model_results['total_hiring_cost']

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
        "Fazla Mesai Maliyeti": total_overtime,  # Added this line to include overtime cost separately
        "Üretim Maliyeti": model_results['total_production_cost'],
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
    try:
        import tabulate
    except ImportError:
        print('tabulate kütüphanesi eksik. Kurmak için: pip install tabulate')
        exit(1)
    print_results()