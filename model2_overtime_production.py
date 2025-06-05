import numpy as np
import pandas as pd

# Parametreler
months = 12
demand = np.array([3000, 900, 3000, 1200, 3200, 2000, 3000, 1800, 3200, 1000, 2900, 1700])
working_days = np.array([22, 20, 23, 19, 21, 19, 22, 22, 22, 21, 21, 21])
holding_cost = 5
stockout_cost = 20
labor_per_unit = 0.5
daily_hours = 8
fixed_workers = 8
overtime_wage_multiplier = 1.5
max_overtime_per_worker = 20  # saat/ay
normal_hourly_wage = 10  # TL/saat
overtime_cost_per_hour = normal_hourly_wage * overtime_wage_multiplier  # Fazla mesai saatlik ücret otomatik hesaplanır

# Check that demand and working_days have the same length
if len(demand) != len(working_days):
    raise ValueError(f"Length of demand ({len(demand)}) and working_days ({len(working_days)}) must be equal.")

def overtime_model():
    production = np.zeros(months)
    overtime_hours = np.zeros(months)
    inventory = np.zeros(months)
    cost = 0
    prev_inventory = 0
    results = []
    for t in range(months):
        # Normal kapasiteyle üretilebilecek miktar
        normal_prod = fixed_workers * working_days[t] * daily_hours / labor_per_unit
        # Fazla mesaiyle üretilebilecek maksimum miktar
        max_overtime_total_hours = fixed_workers * max_overtime_per_worker
        max_overtime_units = max_overtime_total_hours / labor_per_unit
        # Kümülatif stok devriyle talep karşılanmalı
        required = demand[t] - prev_inventory
        if required <= 0:
            prod = 0
            ot_hours = 0
        elif required <= normal_prod:
            prod = required
            ot_hours = 0
        else:
            prod = normal_prod
            extra_needed = required - normal_prod
            overtime_units = min(extra_needed, max_overtime_units)
            ot_hours = overtime_units * labor_per_unit
            prod += overtime_units
        production[t] = prod
        overtime_hours[t] = ot_hours
        inventory[t] = prev_inventory + prod - demand[t]
        holding = max(inventory[t], 0) * holding_cost
        stockout = abs(min(inventory[t], 0)) * stockout_cost
        normal_labor_cost = fixed_workers * working_days[t] * daily_hours * normal_hourly_wage
        overtime_cost = ot_hours * overtime_cost_per_hour
        cost += holding + stockout + overtime_cost + normal_labor_cost
        results.append([
            t+1, prod, ot_hours, inventory[t], holding, stockout, overtime_cost, normal_labor_cost
        ])
        prev_inventory = inventory[t]
    df = pd.DataFrame(results, columns=[
        'Ay', 'Üretim', 'Fazla Mesai (saat)', 'Stok', 'Stok Maliyeti', 'Stoksuzluk Maliyeti', 'Fazla Mesai Maliyeti', 'Normal İşçilik Maliyeti'
    ])
    # Tabulate ile tabloyu ve toplam maliyeti göster
    from tabulate import tabulate
    table = []
    for t in range(months):
        table.append([
            t+1,
            int(fixed_workers),
            int(df['Üretim'][t]),
            int(df['Fazla Mesai (saat)'][t]),
            int(df['Stok'][t]),
            int(df['Stok Maliyeti'][t]),
            int(df['Stoksuzluk Maliyeti'][t]),
            int(df['Fazla Mesai Maliyeti'][t]),
            int(df['Normal İşçilik Maliyeti'][t])
        ])
    headers = [
        'Ay', 'İşçi', 'Üretim', 'Fazla Mesai', 'Stok', 'Stok Maliyeti', 'Stoksuzluk Maliyeti', 'Fazla Mesai Maliyeti', 'Normal İşçilik Maliyeti'
    ]
    print(tabulate(table, headers, tablefmt='github', numalign='right', stralign='center'))
    print(f'\nToplam Maliyet: {cost:,.2f} TL')
    # Grafiksel tablo ve değişken görselleştirme
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib kütüphanesi eksik. Kurmak için: pip install matplotlib')
        return df
    months_list = list(range(1, months+1))
    fig, ax1 = plt.subplots(figsize=(12,7))
    # İşçi sayısı bar (kolon) olarak, ikincil eksende
    worker_count = [fixed_workers] * months
    bar_width = 0.6
    ax1.bar(months_list, worker_count, color='skyblue', label='İşçi', alpha=0.8, width=bar_width, zorder=2)
    ax1.set_xlabel('Ay')
    ax1.set_ylabel('İşçi Sayısı', color='skyblue')
    ax1.tick_params(axis='y', labelcolor='skyblue')
    # Diğer değişkenler ikinci y ekseninde
    ax2 = ax1.twinx()
    ax2.plot(months_list, df['Üretim'], marker='o', label='Üretim', color='green', markersize=7, zorder=3)
    ax2.plot(months_list, df['Fazla Mesai (saat)'], marker='^', label='Fazla Mesai (saat)', color='orange', markersize=7, zorder=3)
    ax2.plot(months_list, df['Stok'], marker='s', label='Stok', color='red', markersize=7, zorder=3)
    ax2.set_ylabel('Adet / Saat', color='gray')
    ax2.tick_params(axis='y', labelcolor='gray')
    # Legend birleştir
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    plt.title('Fazla Mesaili Üretim Modeli Sonuçları')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()
    return df

if __name__ == '__main__':
    overtime_model()
