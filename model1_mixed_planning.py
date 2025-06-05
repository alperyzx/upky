import pulp
import numpy as np

# Örnek parametreler
demand = [500, 290, 5000, 250, 700, 500, 400, 800, 1200, 1000, 1200, 500]
working_days = [22, 20, 23, 19, 21, 19, 22, 22, 22, 21, 21, 21]
holding_cost = 5
stockout_cost = 20
outsourcing_cost = 15
labor_per_unit = 4
hiring_cost = 1800
firing_cost = 1500  # işçi çıkarma maliyeti
daily_hours = 8
min_internal_ratio = 0.70
max_workforce_change = 12  # Daha hızlı işçi azaltımı
max_outsourcing_ratio = 0.30
outsourcing_capacity = 500
hourly_wage = 10  # İşçi saatlik ücreti (TL)
production_cost = 30  # birim üretim maliyeti (TL)

T = len(demand)

# Check that demand and working_days have the same length
if len(demand) != len(working_days):
    raise ValueError(f"Length of demand ({len(demand)}) and working_days ({len(working_days)}) must be equal.")

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

# Amaç fonksiyonu: işçi ücretleri eklendi
cost = (
    pulp.lpSum([
        holding_cost * inventory[t] +
        stockout_cost * stockout[t] +
        outsourcing_cost * outsourced_production[t] +
        hiring_cost * hired[t] +
        firing_cost * fired[t] +
        workers[t] * working_days[t] * daily_hours * hourly_wage +  # işçi ücretleri
        production_cost * internal_production[t]  # iç üretim birim üretim maliyeti
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
    total_internal_labor = 0
    total_internal_prod = 0
    total_outsource = 0
    total_holding = 0
    total_stockout = 0
    total_hiring = 0
    total_firing = 0
    for t in range(T):
        internal_cost = int(internal_production[t].varValue) * working_days[t] * daily_hours * hourly_wage / (working_days[t] * daily_hours / labor_per_unit)
        internal_prod_cost = int(internal_production[t].varValue) * production_cost
        outsourcing_cost_val = int(outsourced_production[t].varValue) * outsourcing_cost
        holding = int(inventory[t].varValue) * holding_cost
        stockout_ = int(stockout[t].varValue) * stockout_cost
        hiring = int(hired[t].varValue) * hiring_cost
        firing = int(fired[t].varValue) * firing_cost
        total_internal_labor += internal_cost
        total_internal_prod += internal_prod_cost
        total_outsource += outsourcing_cost_val
        total_holding += holding
        total_stockout += stockout_
        total_hiring += hiring
        total_firing += firing
        table.append([
            t+1,
            int(workers[t].varValue),
            int(internal_production[t].varValue),
            int(outsourced_production[t].varValue),
            int(inventory[t].varValue),
            int(hired[t].varValue),
            int(fired[t].varValue),
            int(stockout[t].varValue),
            internal_cost,
            internal_prod_cost,
            outsourcing_cost_val
        ])
    headers = [
        'Ay', 'İşçi', 'İç Üretim', 'Fason', 'Stok', 'Alım', 'Çıkış', 'Karşılanmayan Talep',
        'İç Üretim İşçilik Maliyeti', 'İç Üretim Birim Maliyeti', 'Fason Üretim Maliyeti'
    ]
    # Format cost columns
    for row in table:
        row[8] = f'{int(row[8]):,} TL'
        row[9] = f'{int(row[9]):,} TL'
        row[10] = f'{int(row[10]):,} TL'
    print(tabulate(table, headers, tablefmt='fancy_grid', numalign='right', stralign='center'))
    print(f'\nToplam Maliyet: {pulp.value(decision_model.objective):,.2f} TL')
    print(f"\nAyrıntılı Toplam Maliyetler:")
    print(f"- İç Üretim İşçilik Maliyeti Toplamı: {total_internal_labor:,.2f} TL")
    print(f"- İç Üretim Birim Maliyeti Toplamı: {total_internal_prod:,.2f} TL")
    print(f"- Fason Üretim Maliyeti Toplamı: {total_outsource:,.2f} TL")
    print(f"- Stok Maliyeti Toplamı: {total_holding:,.2f} TL")
    print(f"- Karşılanmayan Talep Maliyeti Toplamı: {total_stockout:,.2f} TL")
    print(f"- İşe Alım Maliyeti Toplamı: {total_hiring:,.2f} TL")
    print(f"- İşten Çıkarma Maliyeti Toplamı: {total_firing:,.2f} TL")

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