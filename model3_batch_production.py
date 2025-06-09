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
workers = params['workforce']['workers']
worker_monthly_cost = params['costs']['monthly_wage']
labor_per_unit = params['workforce']['labor_per_unit']  # Birim başına gerekli işçilik saati
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
    workers,
    labor_per_unit,  # Changed from production_rate to labor_per_unit
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
    workers = float(workers)
    daily_hours = float(daily_hours)
    labor_per_unit = float(labor_per_unit)

    # Sabit üretim kapasitesi (işçi * gün * saat / birim işgücü)
    # Dividing by labor_per_unit instead of multiplying by production_rate
    monthly_capacity = workers * daily_hours * working_days_array / labor_per_unit

    production = np.zeros(months)
    inventory = np.zeros(months)
    real_inventory = np.zeros(months)  # To track actual non-negative inventory levels
    prev_inventory = 0
    results = []
    total_cost = 0

    # Use default monthly cost if not provided
    if worker_monthly_cost is None:
        worker_monthly_cost = workers * np.mean(working_days_array) * daily_hours * 10

    for t in range(months):
        # Sabit üretim
        production[t] = monthly_capacity[t]
        inventory[t] = prev_inventory + production[t] - demand_array[t]

        # Calculate actual inventory (cannot be negative) and unfilled demand separately
        real_inventory[t] = max(0, inventory[t])
        unfilled = abs(min(inventory[t], 0))

        holding = real_inventory[t] * holding_cost
        stockout = unfilled * stockout_cost
        labor_cost = workers * worker_monthly_cost
        prod_cost = production[t] * production_cost

        total_cost += holding + stockout + labor_cost + prod_cost

        results.append([
            t+1, production[t], real_inventory[t], holding, stockout, labor_cost, prod_cost, unfilled
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
        'inventory': real_inventory,  # Now using real_inventory for visualization
        'internal_inventory': inventory,  # Keep the internal inventory calculation for reference
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
        demand, working_days, holding_cost, stockout_cost, workers,
        labor_per_unit, daily_hours, production_cost, worker_monthly_cost
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

    # Get hiring cost from params
    hiring_cost = params['costs']['hiring_cost']
    total_hiring_cost = hiring_cost * workers

    # Add hiring cost to total cost
    adjusted_cost = cost + total_hiring_cost

    # Display only the adjusted cost (includes hiring cost)
    print(f'\nToplam Maliyet: {adjusted_cost:,.2f} TL')

    print(f'Stok Maliyeti Toplamı: {detay["total_holding"]:,} TL')
    print(f'Stoksuzluk Maliyeti Toplamı: {detay["total_stockout"]:,} TL')
    print(f'İşçilik Maliyeti Toplamı: {detay["total_labor"]:,} TL')
    print(f'Üretim Maliyeti Toplamı: {detay["total_production_cost"]:,} TL')
    print(f'İşe Alım Maliyeti Toplamı: {total_hiring_cost:,} TL')

    # Birim maliyet analizini fonksiyon ile yap
    birim = birim_maliyet_analizi(
        demand, model_results['production'], model_results['internal_inventory'],
        adjusted_cost, df, workers, hiring_cost
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
        print(f"- Diğer Maliyetler (Stok, Stoksuzluk, İşe Alım): {birim['other_unit_cost']:.2f} TL/birim")
        print(f"- İşe Alım Maliyeti: {birim['hiring_cost']:,} TL (İşçi başına {hiring_cost:,} TL)")
        print(f"- Sabit İşçi Sayısı: {birim['workers']} kişi")
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

def birim_maliyet_analizi(demand, production, inventory, cost, df, workers, hiring_cost=params['costs']['hiring_cost']):
    # Convert demand to a sum if it's a list
    total_demand = sum(demand) if isinstance(demand, list) else demand.sum()
    total_produced = production.sum()
    total_unfilled = sum([abs(min(inventory[t], 0)) for t in range(months)])
    total_holding = df["Stok Maliyeti"].sum()
    total_stockout = df["Stoksuzluk Maliyeti"].sum()
    total_labor = df["İşçilik Maliyeti"].sum()
    total_production_cost = df["Üretim Maliyeti"].sum()

    # Calculate total hiring cost based on worker count
    total_hiring_cost = hiring_cost * workers

    # Add hiring cost to the total cost for unit cost calculation
    adjusted_cost = cost + total_hiring_cost

    result = {
        'total_demand': total_demand,
        'total_produced': total_produced,
        'total_unfilled': total_unfilled,
        'avg_unit_cost': adjusted_cost/total_produced if total_produced > 0 else 0,
        'labor_unit_cost': total_labor/total_produced if total_produced > 0 else 0,
        'prod_unit_cost': total_production_cost/total_produced if total_produced > 0 else 0,
        'other_unit_cost': (total_holding+total_stockout+total_hiring_cost)/total_produced if total_produced > 0 else 0,
        'workers': workers,
        'hiring_cost': total_hiring_cost
    }
    return result

def maliyet_analizi(
    demand=demand,
    working_days=working_days,
    holding_cost=holding_cost,
    stockout_cost=stockout_cost,
    workers=workers,
    labor_per_unit=labor_per_unit,  # Changed from worker_monthly_cost
    daily_hours=daily_hours,
    production_cost=production_cost,
    worker_monthly_cost=worker_monthly_cost,
    hiring_cost=params['costs']['hiring_cost']  # Add hiring_cost parameter with default value
):
    # Use the shared model solver function
    model_results = solve_model(
        demand, working_days, holding_cost, stockout_cost,
        workers, labor_per_unit, daily_hours,  # Changed production_rate to labor_per_unit
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

    # Calculate the hiring cost for the workers (one-time cost)
    total_hiring_cost = hiring_cost * workers

    # Update the total cost to include hiring cost
    toplam_maliyet += total_hiring_cost

    if total_production > 0:
        avg_unit_cost = toplam_maliyet / total_production
        avg_labor_unit = total_labor / total_production
        avg_prod_unit = total_prod_cost / total_production
        avg_other_unit = (total_holding + total_stockout + total_hiring_cost) / total_production
    else:
        avg_unit_cost = avg_labor_unit = avg_prod_unit = avg_other_unit = 0

    return {
        "Toplam Maliyet": toplam_maliyet,
        "İşçilik Maliyeti": total_labor,
        "Üretim Maliyeti": total_prod_cost,
        "Stok Maliyeti": total_holding,
        "Stoksuzluk Maliyeti": total_stockout,
        "İşe Alım Maliyeti": total_hiring_cost,  # Now returning actual hiring cost
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
