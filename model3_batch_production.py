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

# Verimlilik parametrelerini oku
base_efficiency = params['efficiency']['base_efficiency']
scale_threshold = params['efficiency']['scale_threshold']
max_efficiency = params['efficiency']['max_efficiency']
scale_factor = params['efficiency']['scale_factor']

# Kapasite parametrelerini oku
initial_inventory = params['capacity']['initial_inventory']
safety_stock_ratio = params['capacity']['safety_stock_ratio']

# Check that demand and working_days have the same length
if len(demand) != len(working_days):
    raise ValueError(f"Length of demand ({len(demand)}) and working_days ({len(working_days)}) must be equal.")

def calculate_efficiency_factor(total_demand):
    """
    Calculate efficiency factor based on total demand.
    As total demand increases above threshold, efficiency improves.

    Args:
        total_demand: Total production demand

    Returns:
        efficiency_factor: A multiplier that improves (increases) with higher demand
    """
    if total_demand <= scale_threshold:
        return base_efficiency
    else:
        # Efficiency improves with scale but is capped at max_efficiency
        return min(max_efficiency,
                  base_efficiency + (total_demand - scale_threshold) * scale_factor)

def calculate_optimal_workers(demand, working_days, daily_hours, labor_per_unit, efficiency_factor=None):
    """
    Calculate the minimum required workers to meet total demand, considering efficiency.
    Args:
        demand: list or np.array of monthly demand
        working_days: list or np.array of monthly working days
        daily_hours: hours per day
        labor_per_unit: labor hours per unit
        efficiency_factor: optional, if None will be calculated from demand
    Returns:
        optimal_workers: int, minimum required workers (rounded up)
    """
    demand = np.array(demand)
    working_days = np.array(working_days)
    total_demand = np.sum(demand)
    if efficiency_factor is None:
        efficiency_factor = calculate_efficiency_factor(total_demand)
    adjusted_labor_per_unit = labor_per_unit / efficiency_factor
    # Calculate total available labor per worker in the year
    total_available_labor_per_worker = np.sum(working_days) * daily_hours
    # Total labor needed
    total_labor_needed = total_demand * adjusted_labor_per_unit
    optimal_workers = int(np.ceil(total_labor_needed / total_available_labor_per_worker))
    return max(1, optimal_workers)

def solve_model(
    demand,
    working_days,
    holding_cost,
    stockout_cost,
    workers,
    labor_per_unit,
    daily_hours,
    production_cost,
    worker_monthly_cost,
    initial_inventory,
    safety_stock_ratio
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

    # Calculate total demand to determine efficiency factor
    total_demand = np.sum(demand_array)

    # Calculate efficiency factor based on total production volume
    efficiency_factor = calculate_efficiency_factor(total_demand)

    # Optimize workers: restrict to ±10% of optimal
    optimal_workers = calculate_optimal_workers(demand_array, working_days_array, daily_hours, labor_per_unit, efficiency_factor)
    min_workers = int(np.floor(optimal_workers * 0.9))
    max_workers = int(np.ceil(optimal_workers * 1.1))
    original_workers = workers  # Store user input for reporting
    if workers < min_workers:
        workers = min_workers
    elif workers > max_workers:
        workers = max_workers

    # Adjust labor_per_unit based on efficiency - higher efficiency means less labor needed per unit
    adjusted_labor_per_unit = labor_per_unit / efficiency_factor

    # Sabit üretim kapasitesi (işçi * gün * saat / birim işgücü)
    # Using adjusted labor_per_unit to account for efficiency improvements
    monthly_capacity = workers * daily_hours * working_days_array / adjusted_labor_per_unit

    production = np.zeros(months)
    inventory = np.zeros(months)
    real_inventory = np.zeros(months)  # To track actual non-negative inventory levels
    prev_inventory = initial_inventory
    results = []
    total_cost = 0

    # Use default monthly cost if not provided
    if worker_monthly_cost is None:
        worker_monthly_cost = workers * np.mean(working_days_array) * daily_hours * 10

    for t in range(months):
        # Sabit üretim
        production[t] = monthly_capacity[t]
        inventory[t] = prev_inventory + production[t] - demand_array[t]

        # Güvenlik stoğu ve karşılanmayan talep kontrolü, tam sayı olarak
        min_safety_stock = safety_stock_ratio * demand_array[t]
        if inventory[t] >= min_safety_stock:
            real_inventory[t] = int(round(inventory[t]))
            unfilled = 0
        else:
            real_inventory[t] = int(round(min_safety_stock))
            unfilled = int(round(min_safety_stock - inventory[t]))

        holding = max(real_inventory[t], 0) * holding_cost
        stockout = unfilled * stockout_cost
        labor_cost = workers * worker_monthly_cost
        prod_cost = production[t] * production_cost

        total_cost += holding + stockout + labor_cost + prod_cost

        results.append([
            t+1, production[t], real_inventory[t], holding, stockout, labor_cost, prod_cost, unfilled
        ])
        prev_inventory = real_inventory[t]

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
        'efficiency_factor': efficiency_factor,  # Add the efficiency factor to the results
        'adjusted_labor_per_unit': adjusted_labor_per_unit,  # Add the adjusted labor_per_unit
        'optimized_workers': int(workers),  # Return the actual worker count used
        'original_workers': int(original_workers),  # Return the user input for reference
        'optimal_workers': int(optimal_workers),  # Return the calculated optimal worker count
    }

def print_results():
    """
    Runs the model and displays formatted results, detailed analyses, and visualizations.
    """
    from tabulate import tabulate

    # Use solve_model to get the model variables
    model_results = solve_model(
        demand, working_days, holding_cost, stockout_cost, workers,
        labor_per_unit, daily_hours, production_cost, worker_monthly_cost,
        initial_inventory, safety_stock_ratio
    )

    df = model_results['df']
    cost = model_results['total_cost']

    # Get efficiency factor and adjusted labor from model results
    efficiency_factor = model_results['efficiency_factor']
    adjusted_labor_per_unit = model_results['adjusted_labor_per_unit']

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

    # Display efficiency information
    print(f'\nVerimlilik Bilgileri:')
    print(f'Toplam Talep: {model_results["total_produced"] + model_results["total_unfilled"]:,} birim')
    print(f'Verimlilik Faktörü: {efficiency_factor:.2f}')
    print(f'Orijinal Birim İşgücü: {labor_per_unit:.2f} saat/birim')
    print(f'Ayarlanmış Birim İşgücü: {adjusted_labor_per_unit:.2f} saat/birim')
    print(f'Verimlilik İyileşmesi: %{(efficiency_factor - 1) * 100:.2f}')

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
    labor_per_unit=labor_per_unit,
    daily_hours=daily_hours,
    production_cost=production_cost,
    worker_monthly_cost=worker_monthly_cost,
    hiring_cost=params['costs']['hiring_cost'],
    initial_inventory=initial_inventory,
    safety_stock_ratio=safety_stock_ratio
):
    # Use the shared model solver function
    model_results = solve_model(
        demand, working_days, holding_cost, stockout_cost,
        workers, labor_per_unit, daily_hours,
        production_cost, worker_monthly_cost,
        initial_inventory, safety_stock_ratio
    )

    total_production = model_results['total_produced']
    toplam_maliyet = model_results['total_cost']
    total_holding = model_results['total_holding']
    total_stockout = model_results['total_stockout']
    total_labor = model_results['total_labor']
    total_prod_cost = model_results['total_production_cost']
    total_unfilled = model_results['total_unfilled']
    total_demand = sum(demand)
    efficiency_factor = model_results['efficiency_factor']
    adjusted_labor_per_unit = model_results['adjusted_labor_per_unit']
    optimized_workers = model_results['optimized_workers']
    original_workers = model_results['original_workers']
    optimal_workers = model_results['optimal_workers']

    # Calculate the hiring cost for the workers (one-time cost)
    # Use optimized_workers instead of workers to ensure consistency
    total_hiring_cost = hiring_cost * optimized_workers

    # Update the total cost to include hiring cost
    toplam_maliyet += total_hiring_cost

    if total_production > 0:
        avg_unit_cost = toplam_maliyet / total_production
        avg_labor_unit = total_labor / total_production
        avg_prod_unit = total_prod_cost / total_production
        avg_other_unit = (total_holding + total_stockout + total_hiring_cost) / total_production
    else:
        avg_unit_cost = avg_labor_unit = avg_prod_unit = avg_other_unit = 0

    # Bellek temizliği
    del model_results
    import gc
    gc.collect()

    return {
        "Toplam Maliyet": toplam_maliyet,
        "İşçilik Maliyeti": total_labor,
        "Üretim Maliyeti": total_prod_cost,
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
        "Diğer Birim Maliyetler": avg_other_unit,
        "Verimlilik Faktörü": efficiency_factor
    }

if __name__ == '__main__':
    try:
        from tabulate import tabulate
    except ImportError:
        print('tabulate kütüphanesi eksik. Kurmak için: pip install tabulate')
        exit(1)

    # Call the print_results function to run the model and display results
    print_results()