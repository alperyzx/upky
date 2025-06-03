import numpy as np
import pandas as pd

# Parametreler
months = 6
demand = np.array([1000, 1200, 1500, 1300, 1100, 1400])
working_days = np.array([22, 20, 23, 21, 22, 20])
holding_cost = 5
stockout_cost = 20
labor_per_unit = 0.5
hiring_cost = 1000  # Kullanılmayacak
firing_cost = 800   # Kullanılmayacak
daily_hours = 8
fixed_workers = 50
overtime_wage_multiplier = 1.5
max_overtime_per_worker = 20  # saat/ay
overtime_cost_per_hour = 100  # örnek: 100 TL/saat
overtime_limit = fixed_workers * max_overtime_per_worker

# Kapasiteler
total_normal_hours = fixed_workers * working_days * daily_hours
total_overtime_hours = fixed_workers * max_overtime_per_worker
normal_capacity = total_normal_hours / labor_per_unit

def overtime_model():
    production = np.zeros(months)
    overtime_hours = np.zeros(months)
    inventory = np.zeros(months)
    cost = 0
    prev_inventory = 0
    results = []
    for t in range(months):
        # Normal kapasiteyle üretilebilecek miktar
        normal_prod = normal_capacity[t]
        remaining_demand = demand[t] - prev_inventory
        if remaining_demand <= normal_prod:
            production[t] = remaining_demand
            overtime_hours[t] = 0
            inventory[t] = prev_inventory + production[t] - demand[t]
        else:
            production[t] = normal_prod
            extra_needed = remaining_demand - normal_prod
            # Fazla mesaiyle karşılanacak miktar
            overtime_units = min(extra_needed, total_overtime_hours[t] / labor_per_unit)
            overtime_hours[t] = overtime_units * labor_per_unit
            production[t] += overtime_units
            inventory[t] = prev_inventory + production[t] - demand[t]
        # Maliyetler
        holding = max(inventory[t], 0) * holding_cost
        stockout = abs(min(inventory[t], 0)) * stockout_cost
        overtime = overtime_hours[t] * overtime_wage_multiplier * overtime_cost_per_hour
        cost += holding + stockout + overtime
        results.append([
            t+1, production[t], overtime_hours[t], inventory[t], holding, stockout, overtime
        ])
        prev_inventory = inventory[t]
    df = pd.DataFrame(results, columns=[
        'Ay', 'Üretim', 'Fazla Mesai (saat)', 'Stok', 'Stok Maliyeti', 'Stoksuzluk Maliyeti', 'Fazla Mesai Maliyeti'
    ])
    print(df.to_string(index=False))
    print(f'\nToplam Maliyet: {cost:,.2f} TL')

    # Grafiksel çıktı
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib kütüphanesi eksik. Kurmak için: pip install matplotlib')
        return df
    months_list = list(range(1, months+1))
    fig, ax1 = plt.subplots(figsize=(10,6))
    # İşçi sayısı sabit, bar olarak göster
    ax1.bar(months_list, [fixed_workers]*months, color='skyblue', label='İşçi (Sabit)', alpha=0.7)
    ax1.set_xlabel('Ay')
    ax1.set_ylabel('İşçi Sayısı', color='skyblue')
    ax1.tick_params(axis='y', labelcolor='skyblue')
    # Diğer değişkenler çizgi olarak, ikinci y ekseninde
    ax2 = ax1.twinx()
    ax2.plot(months_list, production, marker='s', label='Üretim', color='green')
    ax2.plot(months_list, overtime_hours, marker='^', label='Fazla Mesai (saat)', color='orange')
    ax2.plot(months_list, inventory, marker='d', label='Stok', color='red')
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

