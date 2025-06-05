import pulp
import numpy as np

# Örnek parametreler
demand = [5000, 9000, 27000, 12000, 15000, 25000, 40000, 18000, 12000, 10000, 12000, 15000]
working_days = [22, 20, 23, 19, 21, 19, 22, 22, 22, 21, 21, 21]
holding_cost = 5
stockout_cost = 20
outsourcing_cost = 15
labor_per_unit = 0.5
hiring_cost = 1000
firing_cost = 800  # işçi çıkarma maliyeti
daily_hours = 8
min_internal_ratio = 0.70
max_workforce_change = 8  # Daha hızlı işçi azaltımı
max_outsourcing_ratio = 0.30
outsourcing_capacity = 6000

T = len(demand)

# Model
decision_model = pulp.LpProblem('Karma_Planlama_Modeli', pulp.LpMinimize)

# Karar değişkenleri
workers = [pulp.LpVariable(f'workers_{t}', lowBound=0, cat='Integer') for t in range(T)]
internal_production = [pulp.LpVariable(f'internal_prod_{t}', lowBound=0, cat='Integer') for t in range(T)]
outsourced_production = [pulp.LpVariable(f'outsourced_prod_{t}', lowBound=0, cat='Integer') for t in range(T)]
inventory = [pulp.LpVariable(f'inventory_{t}', lowBound=0, cat='Integer') for t in range(T)]
hired = [pulp.LpVariable(f'hired_{t}', lowBound=0, cat='Integer') for t in range(T)]
fired = [pulp.LpVariable(f'fired_{t}', lowBound=0, cat='Integer') for t in range(T)]
stockout = [pulp.LpVariable(f'stockout_{t}', lowBound=0, cat='Integer') for t in range(T)]  # Karşılanmayan talep

# Amaç fonksiyonu: Toplam maliyet
cost = (
    pulp.lpSum([
        holding_cost * inventory[t] +
        outsourcing_cost * outsourced_production[t] +
        hiring_cost * hired[t] +
        firing_cost * fired[t] +
        stockout_cost * stockout[t]  # Stokta bulundurmama maliyeti eklendi
        for t in range(T)
    ])
)
decision_model += cost

# Kısıtlar
for t in range(T):
    # Talep karşılanmalı (stok + üretim + fason = talep + stok + karşılanmayan talep)
    if t == 0:
        prev_inventory = 0
    else:
        prev_inventory = inventory[t-1]
    decision_model += (internal_production[t] + outsourced_production[t] + prev_inventory == demand[t] + inventory[t] + stockout[t])

    # Toplam üretimin en az %70'i iç üretim olmalı
    decision_model += (internal_production[t] >= min_internal_ratio * (internal_production[t] + outsourced_production[t]))
    # Fason üretim toplam üretimin %Z'sini geçemez
    decision_model += (outsourced_production[t] <= max_outsourcing_ratio * (internal_production[t] + outsourced_production[t]))
    # Fason kapasite
    decision_model += (outsourced_production[t] <= outsourcing_capacity)
    # İç üretim kapasitesi (işçi * gün * saat / birim işgücü)
    decision_model += (internal_production[t] <= workers[t] * working_days[t] * daily_hours / labor_per_unit)
    # İşgücü değişim sınırı
    if t > 0:
        decision_model += (workers[t] - workers[t-1] <= max_workforce_change)
        decision_model += (workers[t-1] - workers[t] <= max_workforce_change)
        decision_model += (workers[t] - workers[t-1] == hired[t] - fired[t])
    else:
        decision_model += (hired[t] - fired[t] == workers[t])
    # İşçi sayısı, iç üretim ihtiyacından fazla olamaz (yuvarlama payı ile)
    decision_model += (workers[t] <= internal_production[t] * labor_per_unit / (working_days[t] * daily_hours) + 1)
    decision_model += (workers[t] >= 0)

# Modeli çöz
solver = pulp.PULP_CBC_CMD(msg=0)
decision_model.solve(solver)

# Sonuçlar
def print_results():
    from tabulate import tabulate
    table = []
    for t in range(T):
        table.append([
            t+1,
            int(workers[t].varValue),
            int(internal_production[t].varValue),
            int(outsourced_production[t].varValue),
            int(inventory[t].varValue),
            int(hired[t].varValue),
            int(fired[t].varValue),
            int(stockout[t].varValue)  # Karşılanmayan talep
        ])
    headers = [
        'Ay', 'İşçi', 'İç Üretim', 'Fason', 'Stok', 'Alım', 'Çıkış', 'Karşılanmayan Talep'
    ]
    print(tabulate(table, headers, tablefmt='github', numalign='right', stralign='center'))
    print(f'\nToplam Maliyet: {pulp.value(decision_model.objective):,.2f} TL')

    # Grafiksel çıktı
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib kütüphanesi eksik. Kurmak için: pip install matplotlib')
        return
    months = [t+1 for t in range(T)]
    fig, ax1 = plt.subplots(figsize=(10,6))
    # İşçi sayısı bar (column) olarak
    ax1.bar(months, [int(workers[t].varValue) for t in range(T)], color='skyblue', label='İşçi', alpha=0.7)
    ax1.set_xlabel('Ay')
    ax1.set_ylabel('İşçi Sayısı', color='skyblue')
    ax1.tick_params(axis='y', labelcolor='skyblue')
    # Diğer değişkenler çizgi olarak, ikinci y ekseninde
    ax2 = ax1.twinx()
    ax2.plot(months, [int(internal_production[t].varValue) for t in range(T)], marker='s', label='İç Üretim', color='green')
    ax2.plot(months, [int(outsourced_production[t].varValue) for t in range(T)], marker='^', label='Fason', color='orange')
    ax2.plot(months, [int(inventory[t].varValue) for t in range(T)], marker='d', label='Stok', color='red')
    ax2.plot(months, [int(stockout[t].varValue) for t in range(T)], marker='x', label='Karşılanmayan Talep', color='black')
    ax2.set_ylabel('Adet', color='gray')
    ax2.tick_params(axis='y', labelcolor='gray')
    # Legend birleştir
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    plt.title('Karma Planlama Modeli Sonuçları')
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