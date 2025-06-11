import pulp
import numpy as np
import yaml
import os

# parametreler.yaml dosyasını oku
with open(os.path.join(os.path.dirname(__file__), 'parametreler.yaml'), 'r', encoding='utf-8') as f:
    params = yaml.safe_load(f)

demand = params['demand']['normal']  # veya 'high', 'seasonal' seçilebilir
working_days = params['workforce']['working_days']
holding_cost = params['costs']['holding_cost']
stockout_cost = params['costs']['stockout_cost']
outsourcing_cost = params['costs']['cost_supplier_A']  # Fason üretim maliyeti (Tedarikçi A)
labor_per_unit = params['workforce']['labor_per_unit']
hiring_cost = params['costs']['hiring_cost']
firing_cost = params['costs']['firing_cost']
daily_hours = params['workforce']['daily_hours']
min_internal_ratio = params['capacity']['min_internal_ratio']
max_workforce_change = params['workforce']['max_workforce_change']
max_outsourcing_ratio = params['capacity']['max_outsourcing_ratio']
outsourcing_capacity = params['capacity']['capacity_supplier_A']
hourly_wage = params['costs']['hourly_wage']
production_cost = params['costs']['production_cost']
overtime_wage_multiplier = params['costs']['overtime_wage_multiplier']
max_overtime_per_worker = params['costs']['max_overtime_per_worker']
initial_inventory = params['capacity']['initial_inventory']
safety_stock_ratio = params['capacity']['safety_stock_ratio']

T = len(demand)

# Check that demand and working_days have the same length
if len(demand) != len(working_days):
    raise ValueError(f"Length of demand ({len(demand)}) and working_days ({len(working_days)}) must be equal.")

def solve_model(
    demand,
    working_days,
    holding_cost,
    outsourcing_cost,
    labor_per_unit,
    hiring_cost,
    firing_cost,
    daily_hours,
    outsourcing_capacity,
    min_internal_ratio,
    max_workforce_change,
    max_outsourcing_ratio,
    stockout_cost,
    hourly_wage,
    production_cost,
    overtime_wage_multiplier,
    max_overtime_per_worker,
    initial_inventory,
    safety_stock_ratio
):
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
    overtime_hours = [pulp.LpVariable(f'overtime_{t}', lowBound=0, cat='Integer') for t in range(T)]  # Fazla mesai saatleri

    # Amaç fonksiyonu: işçi ücretleri ve fazla mesai ücretleri eklendi
    cost = (
        pulp.lpSum([
            holding_cost * inventory[t] +
            stockout_cost * stockout[t] +
            outsourcing_cost * outsourced_production[t] +
            hiring_cost * hired[t] +
            firing_cost * fired[t] +
            workers[t] * working_days[t] * daily_hours * hourly_wage +  # işçi ücretleri
            overtime_hours[t] * hourly_wage * overtime_wage_multiplier +  # fazla mesai ücretleri
            production_cost * internal_production[t]  # iç üretim birim üretim maliyeti
            for t in range(T)
        ])
    )
    decision_model += cost

    # Kısıtlar
    for t in range(T):
        # Talep karşılanmalı (stok + üretim + fason = talep + stok + karşılanmayan talep)
        if t == 0:
            prev_inventory = initial_inventory
        else:
            prev_inventory = inventory[t-1]
        decision_model += (internal_production[t] + outsourced_production[t] + prev_inventory == demand[t] + inventory[t] + stockout[t])
        # Safety stock constraint
        decision_model += (inventory[t] >= safety_stock_ratio * demand[t])

        # Toplam üretimin en az %70'i iç üretim olmalı
        decision_model += (internal_production[t] >= min_internal_ratio * (internal_production[t] + outsourced_production[t]))
        # Fason üretim toplam üretimin %30'unu geçemez
        decision_model += (outsourced_production[t] <= max_outsourcing_ratio * (internal_production[t] + outsourced_production[t]))
        # Fason kapasite
        decision_model += (outsourced_production[t] <= outsourcing_capacity)
        # İç üretim kapasitesi (işçi * gün * saat / birim işgücü) + fazla mesai kapasitesi
        decision_model += (internal_production[t] <= (workers[t] * working_days[t] * daily_hours + overtime_hours[t]) / labor_per_unit)
        # Fazla mesai işçi sayısı ile ilişkili olmalı
        decision_model += (overtime_hours[t] <= workers[t] * max_overtime_per_worker)
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
        # Fazla mesai kısıtı
        decision_model += (overtime_hours[t] <= max_overtime_per_worker)
        decision_model += (overtime_hours[t] >= 0)

    # Modeli çöz
    solver = pulp.PULP_CBC_CMD(msg=0)
    decision_model.solve(solver)

    # Return model variables and objective value
    model_vars = {
        'workers': workers,
        'internal_production': internal_production,
        'outsourced_production': outsourced_production,
        'inventory': inventory,
        'hired': hired,
        'fired': fired,
        'stockout': stockout,
        'overtime_hours': overtime_hours,
        'objective_value': pulp.value(decision_model.objective)
    }

    return model_vars

# Sonuçlar
def ayrintili_toplam_maliyetler(T, internal_production, outsourced_production, inventory, hired, fired, stockout, overtime_hours, working_days, daily_hours, hourly_wage, production_cost, outsourcing_cost, holding_cost, stockout_cost, hiring_cost, firing_cost, overtime_wage_multiplier):
    total_internal_labor = 0
    total_internal_prod = 0
    total_outsource = 0
    total_holding = 0
    total_stockout = 0
    total_hiring = 0
    total_firing = 0
    total_overtime = 0
    for t in range(T):
        internal_cost = int(internal_production[t].varValue) * working_days[t] * daily_hours * hourly_wage / (working_days[t] * daily_hours / labor_per_unit)
        internal_prod_cost = int(internal_production[t].varValue) * production_cost
        outsourcing_cost_val = int(outsourced_production[t].varValue) * outsourcing_cost
        holding = int(inventory[t].varValue) * holding_cost
        stockout_ = int(stockout[t].varValue) * stockout_cost
        hiring = int(hired[t].varValue) * hiring_cost
        firing = int(fired[t].varValue) * firing_cost
        overtime_cost = int(overtime_hours[t].varValue) * hourly_wage * overtime_wage_multiplier
        total_internal_labor += internal_cost
        total_internal_prod += internal_prod_cost
        total_outsource += outsourcing_cost_val
        total_holding += holding
        total_stockout += stockout_
        total_hiring += hiring
        total_firing += firing
        total_overtime += overtime_cost
    return {
        'total_internal_labor': total_internal_labor,
        'total_internal_prod': total_internal_prod,
        'total_outsource': total_outsource,
        'total_holding': total_holding,
        'total_stockout': total_stockout,
        'total_hiring': total_hiring,
        'total_firing': total_firing,
        'total_overtime': total_overtime
    }

def birim_maliyet_analizi(T, demand, internal_production, outsourced_production, stockout, total_internal_labor, total_internal_prod, total_outsource, total_holding, total_hiring, total_firing, total_cost):
    total_demand = sum(demand)
    total_internal_produced = sum([int(internal_production[t].varValue) for t in range(T)])
    total_outsourced = sum([int(outsourced_production[t].varValue) for t in range(T)])
    total_produced = total_internal_produced + total_outsourced
    total_unfilled = sum([int(stockout[t].varValue) for t in range(T)])

    # Calculate weighted average production cost - Fix the variable name from total_outsourcing_cost to total_outsource
    weighted_avg_prod_unit = (total_internal_prod + total_outsource) / total_produced if total_produced > 0 else 0

    result = {
        'total_demand': total_demand,
        'total_internal_produced': total_internal_produced,
        'total_outsourced': total_outsourced,
        'total_produced': total_produced,
        'total_unfilled': total_unfilled,
        'total_cost': total_cost,
        'avg_unit_cost': total_cost/total_produced if total_produced > 0 else 0,
        'internal_labor_unit_cost': total_internal_labor/total_internal_produced if total_internal_produced > 0 else 0,
        'weighted_avg_prod_unit': weighted_avg_prod_unit,  # Add weighted average
        'other_unit_cost': (total_holding+total_hiring+total_firing)/total_produced if total_produced > 0 else 0
    }
    return result

def print_results():
    from tabulate import tabulate

    # Use solve_model() to get the model variables
    model_vars = solve_model(
        demand, working_days, holding_cost, outsourcing_cost, labor_per_unit,
        hiring_cost, firing_cost, daily_hours, outsourcing_capacity, min_internal_ratio,
        max_workforce_change, max_outsourcing_ratio, stockout_cost, hourly_wage,
        production_cost, overtime_wage_multiplier, max_overtime_per_worker, initial_inventory, safety_stock_ratio
    )

    # Extract variables from model_vars
    workers = model_vars['workers']
    internal_production = model_vars['internal_production']
    outsourced_production = model_vars['outsourced_production']
    inventory = model_vars['inventory']
    hired = model_vars['hired']
    fired = model_vars['fired']
    stockout = model_vars['stockout']
    overtime_hours = model_vars['overtime_hours']
    objective_value = model_vars['objective_value']

    table = []
    for t in range(T):
        internal_cost = int(internal_production[t].varValue) * working_days[t] * daily_hours * hourly_wage / (working_days[t] * daily_hours / labor_per_unit)
        internal_prod_cost = int(internal_production[t].varValue) * production_cost
        outsourcing_cost_val = int(outsourced_production[t].varValue) * outsourcing_cost
        holding = int(inventory[t].varValue) * holding_cost
        stockout_ = int(stockout[t].varValue) * stockout_cost
        hiring = int(hired[t].varValue) * hiring_cost
        firing = int(fired[t].varValue) * firing_cost
        overtime_cost = int(overtime_hours[t].varValue) * hourly_wage * overtime_wage_multiplier
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
            outsourcing_cost_val,
            overtime_cost
        ])
    headers = [
        'Ay', 'İşçi', 'İç Üretim', 'Fason', 'Stok', 'Alım', 'Çıkış', 'Karşılanmayan Talep',
        'İç Üretim İşçilik Maliyeti', 'İç Üretim Birim Maliyeti', 'Fason Üretim Maliyeti', 'Fazla Mesai Maliyeti'
    ]
    # Format cost columns
    for row in table:
        row[8] = f'{int(row[8]):,} TL'
        row[9] = f'{int(row[9]):,} TL'
        row[10] = f'{int(row[10]):,} TL'
        row[11] = f'{int(row[11]):,} TL'
    print(tabulate(table, headers, tablefmt='fancy_grid', numalign='right', stralign='center'))
    print(f'\nToplam Maliyet: {objective_value:,.2f} TL')
    detay = ayrintili_toplam_maliyetler(
        T, internal_production, outsourced_production, inventory, hired, fired, stockout, overtime_hours,
        working_days, daily_hours, hourly_wage, production_cost, outsourcing_cost, holding_cost, stockout_cost, hiring_cost, firing_cost, overtime_wage_multiplier
    )
    print(f"\nAyrıntılı Toplam Maliyetler:")
    print(f"- İç Üretim İşçilik Maliyeti Toplamı: {detay['total_internal_labor']:,.2f} TL")
    print(f"- İç Üretim Birim Maliyeti Toplamı: {detay['total_internal_prod']:,.2f} TL")
    print(f"- Fason Üretim Maliyeti Toplamı: {detay['total_outsource']:,.2f} TL")
    print(f"- Stok Maliyeti Toplamı: {detay['total_holding']:,.2f} TL")
    print(f"- Karşılanmayan Talep Maliyeti Toplamı: {detay['total_stockout']:,.2f} TL")
    print(f"- İşe Alım Maliyeti Toplamı: {detay['total_hiring']:,.2f} TL")
    print(f"- İşten Çıkarma Maliyeti Toplamı: {detay['total_firing']:,.2f} TL")
    print(f"- Fazla Mesai Maliyeti Toplamı: {detay['total_overtime']:,.2f} TL")
    birim = birim_maliyet_analizi(
        T, demand, internal_production, outsourced_production, stockout,
        detay['total_internal_labor'], detay['total_internal_prod'], detay['total_outsource'], detay['total_holding'], detay['total_hiring'], detay['total_firing'], objective_value
    )
    print(f"\nBirim Maliyet Analizi:")
    print(f"- Toplam Talep: {birim['total_demand']:,} birim")
    print(f"- Toplam İç Üretim: {birim['total_internal_produced']:,} birim ({birim['total_internal_produced']/birim['total_demand']*100:.2f}%)")
    print(f"- Toplam Fason Üretim: {birim['total_outsourced']:,} birim ({birim['total_outsourced']/birim['total_demand']*100:.2f}%)")
    print(f"- Toplam Üretim: {birim['total_produced']:,} birim ({birim['total_produced']/birim['total_demand']*100:.2f}%)")
    print(f"- Karşılanmayan Talep: {birim['total_unfilled']:,} birim ({birim['total_unfilled']/birim['total_demand']*100:.2f}%)")
    if birim['total_produced'] > 0:
        print(f"- Ortalama Birim Maliyet (Toplam): {birim['avg_unit_cost']:.2f} TL/birim")
        if birim['total_internal_produced'] > 0:
            print(f"- İşçilik Birim Maliyeti: {birim['internal_labor_unit_cost']:.2f} TL/birim")
            # Display weighted average production cost
            print(f"- Üretim Birim Maliyeti (Ağırlıklı Ortalama): {birim['weighted_avg_prod_unit']:.2f} TL/birim")
            # Add note about the weighted average
            print(f"  Not: Bu maliyet, iç üretim ({production_cost} TL/birim) ve fason üretimin ({outsourcing_cost} TL/birim) ağırlıklı ortalamasıdır.")
        print(f"- Diğer Maliyetler (Stok, İşe Alım/Çıkarma): {birim['other_unit_cost']:.2f} TL/birim")
    else:
        print("- Ortalama Birim Maliyet: Hesaplanamadı (0 birim üretildi)")

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

def maliyet_analizi(
    demand=demand,
    working_days=working_days,
    holding_cost=holding_cost,
    stockout_cost=stockout_cost,
    outsourcing_cost=outsourcing_cost,
    labor_per_unit=labor_per_unit,
    hiring_cost=hiring_cost,
    firing_cost=firing_cost,
    daily_hours=daily_hours,
    min_internal_ratio=min_internal_ratio,
    max_workforce_change=max_workforce_change,
    max_outsourcing_ratio=max_outsourcing_ratio,
    outsourcing_capacity=outsourcing_capacity,
    hourly_wage=hourly_wage,
    production_cost=production_cost,
    overtime_wage_multiplier=overtime_wage_multiplier,
    max_overtime_per_worker=max_overtime_per_worker,
    initial_inventory=initial_inventory,
    safety_stock_ratio=safety_stock_ratio
):
    # Solve model using the shared function
    model_vars = solve_model(
        demand, working_days, holding_cost, outsourcing_cost, labor_per_unit,
        hiring_cost, firing_cost, daily_hours, outsourcing_capacity, min_internal_ratio,
        max_workforce_change, max_outsourcing_ratio, stockout_cost, hourly_wage,
        production_cost, overtime_wage_multiplier, max_overtime_per_worker,
        initial_inventory, safety_stock_ratio
    )

    # Extract variables from model_vars
    workers = model_vars['workers']
    internal_production = model_vars['internal_production']
    outsourced_production = model_vars['outsourced_production']
    inventory = model_vars['inventory']
    hired = model_vars['hired']
    fired = model_vars['fired']
    stockout = model_vars['stockout']
    overtime_hours = model_vars['overtime_hours']
    toplam_maliyet = model_vars['objective_value']

    T = len(demand)

    # Calculate detailed costs
    total_internal_labor = 0
    total_internal_prod = 0
    total_outsource = 0
    total_holding = 0
    total_stockout = 0
    total_hiring = 0
    total_firing = 0
    total_overtime = 0
    total_produced = 0
    total_demand = sum(demand)
    total_unfilled = 0
    for t in range(T):
        internal_cost = int(internal_production[t].varValue) * working_days[t] * daily_hours * hourly_wage / (working_days[t] * daily_hours / labor_per_unit)
        internal_prod_cost = int(internal_production[t].varValue) * production_cost
        outsourcing_cost_val = int(outsourced_production[t].varValue) * outsourcing_cost
        holding = int(inventory[t].varValue) * holding_cost
        stockout_ = int(stockout[t].varValue) * stockout_cost
        hiring = int(hired[t].varValue) * hiring_cost
        firing = int(fired[t].varValue) * firing_cost
        overtime_cost = int(overtime_hours[t].varValue) * hourly_wage * overtime_wage_multiplier
        total_internal_labor += internal_cost
        total_internal_prod += internal_prod_cost
        total_outsource += outsourcing_cost_val
        total_holding += holding
        total_stockout += stockout_
        total_hiring += hiring
        total_firing += firing
        total_overtime += overtime_cost
        total_produced += int(internal_production[t].varValue) + int(outsourced_production[t].varValue)
        total_unfilled += int(stockout[t].varValue)

    if total_produced > 0:
        avg_unit_cost = toplam_maliyet / total_produced
        avg_labor_unit = total_internal_labor / total_produced
        # This is the correct weighted average calculation
        avg_prod_unit = (total_internal_prod + total_outsource) / total_produced
        avg_other_unit = (total_holding + total_stockout + total_hiring + total_firing) / total_produced
    else:
        avg_unit_cost = avg_labor_unit = avg_prod_unit = avg_other_unit = 0
    
    result = {
        "Toplam Maliyet": toplam_maliyet,
        "İşçilik Maliyeti": total_internal_labor,
        "Üretim Maliyeti": total_internal_prod + total_outsource,
        "Stok Maliyeti": total_holding,
        "Stoksuzluk Maliyeti": total_stockout,
        "İşe Alım Maliyeti": total_hiring,
        "İşten Çıkarma Maliyeti": total_firing,
        "Fazla Mesai Maliyeti": total_overtime,
        "Toplam Talep": total_demand,
        "Toplam Üretim": total_produced,
        "Karşılanmayan Talep": total_unfilled,
        "Ortalama Birim Maliyet": avg_unit_cost,
        "İşçilik Birim Maliyeti": avg_labor_unit,
        "Üretim Birim Maliyeti": avg_prod_unit,
        "Diğer Birim Maliyetler": avg_other_unit
    }

    # Bellek temizliği
    del model_vars
    import gc
    gc.collect()

    return result

if __name__ == '__main__':
    try:
        import tabulate
    except ImportError:
        print('tabulate kütüphanesi eksik. Kurmak için: pip install tabulate')
        exit(1)
    print_results()