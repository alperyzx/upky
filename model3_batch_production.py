import numpy as np
import pandas as pd
from tabulate import tabulate
import yaml
import os

# parametreler.yaml dosyasını oku
with open(os.path.join(os.path.dirname(__file__), 'parametreler.yaml'), 'r', encoding='utf-8') as f:
    params = yaml.safe_load(f)

demand = np.array(params['demand']['normal'])  # veya 'high', 'seasonal' seçilebilir
working_days = np.array(params['workforce']['working_days'])
holding_cost = params['costs']['holding_cost']
stockout_cost = params['costs']['stockout_cost']
fixed_workers = params['workforce']['workers']
worker_monthly_cost = params['costs']['monthly_wage']
production_rate = 1 / params['workforce']['labor_per_unit']  # Bir işçi günde kaç birim üretir
# (ör: labor_per_unit=4 ise, 1/4=0.25 birim/işçi/gün)
daily_hours = params['workforce']['daily_hours']
months = len(demand)
production_cost = params['costs']['production_cost']

# Check that demand and working_days have the same length
if len(demand) != len(working_days):
    raise ValueError(f"Length of demand ({len(demand)}) and working_days ({len(working_days)}) must be equal.")

def solve_model(
    demand,
    working_days,
    holding_cost,
    stockout_cost,
    fixed_workers,
    production_rate,
    daily_hours,
    production_cost,
    worker_monthly_cost=None
):
    """
    Core model logic for Model 3 (Toplu Üretim ve Stoklama)
    Returns calculated values as a dictionary
    """
    months = len(demand)

    # Convert inputs to numpy arrays to ensure proper broadcasting
    demand_array = np.array(demand)
    working_days_array = np.array(working_days)

    # Ensure scalar values are properly handled
    fixed_workers = float(fixed_workers)
    daily_hours = float(daily_hours)
    production_rate = float(production_rate)

    # Sabit üretim kapasitesi (işçi * gün * saat * hız)
    monthly_capacity = fixed_workers * daily_hours * working_days_array * production_rate

    production = np.zeros(months)
    inventory = np.zeros(months)
    prev_inventory = 0
    results = []
    total_cost = 0

    # Use default monthly cost if not provided
    if worker_monthly_cost is None:
        worker_monthly_cost = fixed_workers * np.mean(working_days_array) * daily_hours * 10

    for t in range(months):
        # Sabit üretim
        production[t] = monthly_capacity[t]
        inventory[t] = prev_inventory + production[t] - demand_array[t]
        holding = max(inventory[t], 0) * holding_cost
        stockout = abs(min(inventory[t], 0)) * stockout_cost
        unfilled = abs(min(inventory[t], 0))
        labor_cost = fixed_workers * worker_monthly_cost
        prod_cost = production[t] * production_cost

        total_cost += holding + stockout + labor_cost + prod_cost

        results.append([
            t+1, production[t], inventory[t], holding, stockout, labor_cost, prod_cost, unfilled
        ])
        prev_inventory = inventory[t]

    headers = [
        'Ay', 'Üretim', 'Stok', 'Stok Maliyeti', 'Stoksuzluk Maliyeti',
        'İşçilik Maliyeti', 'Üretim Maliyeti', 'Karşılanmayan Talep'
    ]

    df = pd.DataFrame(results, columns=headers)

    # Calculate summary statistics
    total_produced = np.sum(production)
    total_unfilled = np.sum([abs(min(inv, 0)) for inv in inventory])
    total_holding = df["Stok Maliyeti"].sum()
    total_stockout = df["Stoksuzluk Maliyeti"].sum()
    total_labor = df["İşçilik Maliyeti"].sum()
    total_production_cost = df["Üretim Maliyeti"].sum()

    return {
        'df': df,
        'production': production,
        'inventory': inventory,
        'total_cost': total_cost,
        'total_holding': total_holding,
        'total_stockout': total_stockout,
        'total_labor': total_labor,
        'total_production_cost': total_production_cost,
        'total_produced': total_produced,
        'total_unfilled': total_unfilled,
    }

def print_results():
    """
    Runs the model and displays formatted results, detailed analyses, and visualizations.
    """
    from tabulate import tabulate

    # Use solve_model to get the model variables
    model_results = solve_model(
        demand, working_days, holding_cost, stockout_cost, fixed_workers,
        production_rate, daily_hours, production_cost, worker_monthly_cost
    )

    df = model_results['df']
    cost = model_results['total_cost']

    # Hücrelerden TL birimini kaldır, sadece sayısal kalsın (virgülsüz, int)
    df['Stok Maliyeti'] = df['Stok Maliyeti'].astype(int)
    df['Stoksuzluk Maliyeti'] = df['Stoksuzluk Maliyeti'].astype(int)
    df.rename(columns={df.columns[5]: 'İşçilik Maliyeti'}, inplace=True)  # Fix encoding issue
    df['İşçilik Maliyeti'] = df['İşçilik Maliyeti'].astype(int)
    df['Üretim Maliyeti'] = df['Üretim Maliyeti'].astype(int)
    print(tabulate(df, headers='keys', tablefmt='fancy_grid', showindex=False, numalign='right', stralign='center'))

    detay = ayrintili_toplam_maliyetler(df)
    print(f'\nToplam Maliyet: {cost:,.2f} TL')
    print(f'Stok Maliyeti Toplamı: {detay["total_holding"]:,} TL')
    print(f'Stoksuzluk Maliyeti Toplamı: {detay["total_stockout"]:,} TL')
    print(f'İşçilik Maliyeti Toplamı: {detay["total_labor"]:,} TL')
    print(f'Üretim Maliyeti Toplamı: {detay["total_production_cost"]:,} TL')

    # Birim maliyet analizini fonksiyon ile yap
    birim = birim_maliyet_analizi(
        demand, model_results['production'], model_results['inventory'],
        cost, df, fixed_workers, months
    )
    print(f"\nBirim Maliyet Analizi:")
    print(f"- Toplam Talep: {birim['total_demand']:,} birim")
    print(f"- Toplam Üretim: {birim['total_produced']:,} birim ({birim['total_produced']/birim['total_demand']*100:.2f}%)")
    if birim['total_unfilled'] > 0:
        print(f"- Karşılanmayan Talep: {birim['total_unfilled']:,} birim ({birim['total_unfilled']/birim['total_demand']*100:.2f}%)")
    if birim['total_produced'] > 0:
        print(f"- Ortalama Birim Maliyet (Toplam): {birim['avg_unit_cost']:.2f} TL/birim")
        print(f"- İşçilik Birim Maliyeti: {birim['labor_unit_cost']:.2f} TL/birim")
        print(f"- Üretim Birim Maliyeti: {birim['prod_unit_cost']:.2f} TL/birim")
        print(f"- Diğer Maliyetler (Stok, Stoksuzluk): {birim['other_unit_cost']:.2f} TL/birim")
        print(f"- Sabit İşçi Sayısı: {birim['fixed_workers']} kişi")
        print(f"- İşçi Başına Aylık Ortalama Üretim: {birim['avg_prod_per_worker']:.2f} birim/ay")
    else:
        print("- Ortalama Birim Maliyet: Hesaplanamadı (0 birim üretildi)")

    # Grafiksel çıktı
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib kütüphanesi eksik. Kurmak için: pip install matplotlib')
        exit(1)
    months_list = list(range(1, months+1))
    plt.figure(figsize=(10,6))
    plt.bar(months_list, model_results['production'], color='skyblue', label='Üretim', alpha=0.7)
    plt.plot(months_list, model_results['inventory'], marker='d', label='Stok', color='red')
    plt.plot(months_list, df['Karşılanmayan Talep'], marker='x', label='Karşılanmayan Talep', color='black')
    plt.xlabel('Ay')
    plt.ylabel('Adet / TL')
    plt.title('Toplu Üretim ve Stoklama Modeli Sonuçları')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    return df, cost

def ayrintili_toplam_maliyetler(df):
    return {
        'total_holding': df["Stok Maliyeti"].sum(),
        'total_stockout': df["Stoksuzluk Maliyeti"].sum(),
        'total_labor': df["İşçilik Maliyeti"].sum(),
        'total_production_cost': df["Üretim Maliyeti"].sum(),
    }

def birim_maliyet_analizi(demand, production, inventory, cost, df, fixed_workers, months):
    total_demand = demand.sum()
    total_produced = production.sum()
    total_unfilled = sum([abs(min(inventory[t], 0)) for t in range(months)])
    total_holding = df["Stok Maliyeti"].sum()
    total_stockout = df["Stoksuzluk Maliyeti"].sum()
    total_labor = df["İşçilik Maliyeti"].sum()
    total_production_cost = df["Üretim Maliyeti"].sum()
    result = {
        'total_demand': total_demand,
        'total_produced': total_produced,
        'total_unfilled': total_unfilled,
        'avg_unit_cost': cost/total_produced if total_produced > 0 else 0,
        'labor_unit_cost': total_labor/total_produced if total_produced > 0 else 0,
        'prod_unit_cost': total_production_cost/total_produced if total_produced > 0 else 0,
        'other_unit_cost': (total_holding+total_stockout)/total_produced if total_produced > 0 else 0,
        'fixed_workers': fixed_workers,
        'avg_prod_per_worker': total_produced/(fixed_workers*months) if fixed_workers > 0 else 0
    }
    return result

def maliyet_analizi(
    demand=demand,
    working_days=working_days,
    holding_cost=holding_cost,
    stockout_cost=stockout_cost,
    fixed_workers=fixed_workers,
    worker_monthly_cost=worker_monthly_cost,
    production_rate=production_rate,
    daily_hours=daily_hours,
    production_cost=production_cost
):
    # Use the shared model solver function
    model_results = solve_model(
        demand, working_days, holding_cost, stockout_cost,
        fixed_workers, production_rate, daily_hours,
        production_cost, worker_monthly_cost
    )

    total_production = model_results['total_produced']
    toplam_maliyet = model_results['total_cost']
    total_holding = model_results['total_holding']
    total_stockout = model_results['total_stockout']
    total_labor = model_results['total_labor']
    total_prod_cost = model_results['total_production_cost']
    total_unfilled = model_results['total_unfilled']
    total_demand = sum(demand)

    if total_production > 0:
        avg_unit_cost = toplam_maliyet / total_production
        avg_labor_unit = total_labor / total_production
        avg_prod_unit = total_prod_cost / total_production
        avg_other_unit = (total_holding + total_stockout) / total_production
    else:
        avg_unit_cost = avg_labor_unit = avg_prod_unit = avg_other_unit = 0

    return {
        "Toplam Maliyet": toplam_maliyet,
        "İşçilik Maliyeti": total_labor,
        "Üretim Maliyeti": total_prod_cost,
        "Stok Maliyeti": total_holding,
        "Stoksuzluk Maliyeti": total_stockout,
        "İşe Alım Maliyeti": 0,
        "İşten Çıkarma Maliyeti": 0,
        "Toplam Talep": total_demand,
        "Toplam Üretim": total_production,
        "Karşılanmayan Talep": total_unfilled,
        "Ortalama Birim Maliyet": avg_unit_cost,
        "İşçilik Birim Maliyeti": avg_labor_unit,
        "Üretim Birim Maliyeti": avg_prod_unit,
        "Diğer Birim Maliyetler": avg_other_unit
    }

if __name__ == '__main__':
    try:
        from tabulate import tabulate
    except ImportError:
        print('tabulate kütüphanesi eksik. Kurmak için: pip install tabulate')
        exit(1)

    # Call the print_results function to run the model and display results
    print_results()