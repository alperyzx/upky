import numpy as np
import pandas as pd

# Parametreler
months = 12
demand = np.array([3000, 6000, 3000, 4200, 3200, 4000, 3000, 1800, 3700, 4500, 2900, 4200])
working_days = np.array([22, 20, 23, 19, 21, 19, 22, 22, 22, 21, 21, 21])
holding_cost = 5
stockout_cost = 20
labor_per_unit = 0.5
daily_hours = 8
fixed_workers = 12
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
        cost += holding + stockout + overtime + normal_labor_cost
        results.append([
            t+1, fixed_workers, prod, ot_hours, inventory[t], holding, stockout, overtime, normal_labor_cost
        ])
        prev_inventory = inventory[t]
    headers = [
        'Ay', 'İşçi', 'Üretim', 'Fazla Mesai', 'Stok', 'Stok Maliyeti', 'Stoksuzluk Maliyeti', 'Fazla Mesai Maliyeti', 'Normal İşçilik Maliyeti'
    ]
    df = pd.DataFrame(results, columns=headers)
    # Tabloyu daha okunaklı şekilde göster
    from tabulate import tabulate
    print(tabulate(df, headers='keys', tablefmt='psql', showindex=False))
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

if __name__ == '__main__':
    overtime_model()
