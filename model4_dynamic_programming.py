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
hiring_cost = params['costs']['hiring_cost']
firing_cost = params['costs']['firing_cost']
daily_hours = params['workforce']['daily_hours']
labor_per_unit = params['workforce']['labor_per_unit']
max_workers = params['workforce']['max_workers']
workers = params['workforce']['workers']
max_workforce_change = params['workforce']['max_workforce_change']
months = len(demand)
hourly_wage = params['costs']['hourly_wage']
production_cost = params['costs']['production_cost']
initial_inventory = params['capacity']['initial_inventory']
safety_stock_ratio = params['capacity']['safety_stock_ratio']

# Check that demand and working_days have the same length
if len(demand) != len(working_days):
    raise ValueError(f"Length of demand ({len(demand)}) and working_days ({len(working_days)}) must be equal.")

# Üretim kapasitesi: işçi * gün * saat / birim işgücü
def prod_capacity(workers, t):
    return workers * working_days[t] * daily_hours / labor_per_unit

def solve_model(
    demand,
    working_days,
    holding_cost,
    stockout_cost,
    hiring_cost,
    firing_cost,
    daily_hours,
    labor_per_unit,
    workers,
    max_workers,
    max_workforce_change,
    hourly_wage,
    production_cost,
    initial_inventory,
    safety_stock_ratio
):
    """
    Core model logic for Dynamic Programming model (Model 4)
    Returns calculated values as a dictionary
    """
    months = len(demand)

    # Üretim kapasitesi: işçi * gün * saat / birim işgücü
    def prod_capacity(workers, t):
        return workers * working_days[t] * daily_hours / labor_per_unit

    # DP tabloları
    cost_table = np.full((months+1, max_workers+1), np.inf)
    backtrack = np.full((months+1, max_workers+1), -1, dtype=int)

    # Başlangıç: 0. ayda 0 işçiyle başla (işçi alımı model içinde değerlendirilecek)
    cost_table[0, 0] = 0

    # DP algoritması
    for t in range(months):
        for prev_w in range(0, max_workers+1):  # 0 işçiden başlayarak tüm olasılıkları değerlendir
            if cost_table[t, prev_w] < np.inf:
                # İlk dönemde işçi sayısı kısıtı yok (min 0), sonraki dönemlerde max_workforce_change kısıtı var
                if t == 0:
                    min_w = 0  # İlk dönemde işçi sayısı en az 0 olabilir (talebe göre optimize edilecek)
                    max_w = max_workers  # İlk dönemde istediğimiz kadar işe alabiliriz
                else:
                    min_w = max(0, prev_w - max_workforce_change)  # Minimum 0 işçi olabilir
                    max_w = min(max_workers, prev_w + max_workforce_change)

                for w in range(min_w, max_w + 1):
                    capacity = prod_capacity(w, t)
                    # Karşılanamayan talep varsa ceza maliyeti uygula
                    unmet = max(0, demand[t] - capacity)
                    inventory = max(0, capacity - demand[t])
                    hire = max(0, w - prev_w) * hiring_cost
                    fire = max(0, prev_w - w) * firing_cost
                    holding = inventory * holding_cost
                    stockout = unmet * stockout_cost
                    labor = w * working_days[t] * daily_hours * hourly_wage

                    # Kapasiteye değil, gerçek üretim miktarına göre maliyet hesaplanmalı
                    actual_prod = min(capacity, demand[t])
                    prod_cost = actual_prod * production_cost  # Kapasite değil, gerçek üretim miktarı

                    total_cost = cost_table[t, prev_w] + hire + fire + holding + stockout + labor + prod_cost
                    if total_cost < cost_table[t+1, w]:
                        cost_table[t+1, w] = total_cost
                        backtrack[t+1, w] = prev_w

    # En düşük maliyetli yolun sonundaki işçi sayısı
    final_workers = np.argmin(cost_table[months])
    min_cost = cost_table[months, final_workers]

    # Geriye doğru optimal yolun çıkarılması
    workers_seq = []
    w = final_workers
    for t in range(months, 0, -1):
        workers_seq.append(w)
        w = backtrack[t, w]
    workers_seq = workers_seq[::-1]

    # İlk dönemdeki işçi sayısı (başlangıç işçisi)
    initial_workers = w  # Bu, gerçekte sıfırdan başlamışsa, ilk dönemde işe aldığımız işçi sayısıdır

    # Üretim, stok, ve diğer hesaplamalar
    production_seq = []
    inventory_seq = []
    labor_cost_seq = []
    unmet_seq = []
    stockout_cost_seq = []
    prod_cost_seq = []
    hiring_seq = []
    firing_seq = []

    # Başlangıç envanteri
    prev_inventory = initial_inventory
    prev_w = 0  # Başlangıçta sıfır işçi var
    for t, w in enumerate(workers_seq):
        cap = prod_capacity(w, t)
        prod = min(cap, demand[t])
        # Stok ve güvenlik stoğu kontrolü
        ending_inventory = prev_inventory + prod - demand[t]
        min_safety_stock = safety_stock_ratio * demand[t]
        if ending_inventory >= min_safety_stock:
            inventory = int(round(ending_inventory))
            unmet = 0
        else:
            inventory = int(round(min_safety_stock))
            unmet = int(round(min_safety_stock - ending_inventory))
        labor_cost = w * working_days[t] * daily_hours * hourly_wage
        prod_cost = prod * production_cost
        hire = max(0, w - prev_w) * hiring_cost
        fire = max(0, prev_w - w) * firing_cost
        production_seq.append(prod)
        inventory_seq.append(inventory)
        labor_cost_seq.append(labor_cost)
        unmet_seq.append(unmet)
        stockout_cost_seq.append(unmet * stockout_cost)
        prod_cost_seq.append(prod_cost)
        hiring_seq.append(hire)
        firing_seq.append(fire)
        prev_inventory = inventory
        prev_w = w

    # Calculate totals
    total_labor = sum(labor_cost_seq)
    total_production = sum(prod_cost_seq)
    total_holding = sum(inventory * holding_cost for inventory in inventory_seq)
    total_stockout = sum(unmet * stockout_cost for unmet in unmet_seq)
    total_hiring = sum(hiring_seq)
    total_firing = sum(firing_seq)
    total_demand = sum(demand)
    total_produced = sum(production_seq)
    total_unfilled = sum(unmet_seq)

    # Create results dataframe
    results = []
    for t in range(months):
        holding = inventory_seq[t] * holding_cost
        stockout = unmet_seq[t] * stockout_cost

        results.append([
            t+1, workers_seq[t], production_seq[t], inventory_seq[t], unmet_seq[t],
            labor_cost_seq[t], prod_cost_seq[t], hiring_seq[t], firing_seq[t], holding, stockout
        ])

    df = pd.DataFrame(results, columns=[
        'Ay', 'İşçi', 'Üretim', 'Stok', 'Karşılanmayan Talep',
        'İşçilik Maliyeti', 'Üretim Maliyeti', 'İşe Alım Maliyeti',
        'İşten Çıkarma Maliyeti', 'Stok Maliyeti', 'Stoksuzluk Maliyeti'
    ])

    # Return comprehensive results
    return {
        'df': df,
        'workers_seq': workers_seq,
        'production_seq': production_seq,
        'inventory_seq': inventory_seq,
        'unmet_seq': unmet_seq,
        'min_cost': min_cost,
        'total_labor': total_labor,
        'total_production': total_production,
        'total_holding': total_holding,
        'total_stockout': total_stockout,
        'total_hiring': total_hiring,
        'total_firing': total_firing,
        'total_demand': total_demand,
        'total_produced': total_produced,
        'total_unfilled': total_unfilled,
        'initial_workers': initial_workers
    }

def maliyet_analizi(
    demand=demand,
    working_days=working_days,
    holding_cost=holding_cost,
    stockout_cost=stockout_cost,
    hiring_cost=hiring_cost,
    firing_cost=firing_cost,
    daily_hours=daily_hours,
    labor_per_unit=labor_per_unit,
    workers=workers,
    max_workers=max_workers,
    max_workforce_change=max_workforce_change,
    hourly_wage=hourly_wage,
    production_cost=production_cost,
    initial_inventory=initial_inventory,
    safety_stock_ratio=safety_stock_ratio
):
    # Use the shared model solver function
    model_results = solve_model(
        demand, working_days, holding_cost, stockout_cost, hiring_cost, firing_cost,
        daily_hours, labor_per_unit, workers, max_workers, max_workforce_change,
        hourly_wage, production_cost, initial_inventory, safety_stock_ratio
    )

    toplam_maliyet = model_results['min_cost']
    total_labor = model_results['total_labor']
    total_production = model_results['total_production']
    total_holding = model_results['total_holding']
    total_stockout = model_results['total_stockout']
    total_hiring = model_results['total_hiring']
    total_firing = model_results['total_firing']
    total_demand = model_results['total_demand']
    total_produced = model_results['total_produced']
    total_unfilled = model_results['total_unfilled']

    # Calculate unit costs
    if total_produced > 0:
        avg_unit_cost = toplam_maliyet / total_produced
        avg_labor_unit = total_labor / total_produced
        avg_prod_unit = total_production / total_produced
        avg_other_unit = (total_holding + total_stockout + total_hiring + total_firing) / total_produced
    else:
        avg_unit_cost = avg_labor_unit = avg_prod_unit = avg_other_unit = 0

    # Bellek temizliği
    del model_results
    import gc
    gc.collect()

    return {
        "Toplam Maliyet": toplam_maliyet,
        "İşçilik Maliyeti": total_labor,
        "Üretim Maliyeti": total_production,
        "Stok Maliyeti": total_holding,
        "Stoksuzluk Maliyeti": total_stockout,
        "İşe Alım Maliyeti": total_hiring,
        "İşten Çıkarma Maliyeti": total_firing,
        "Toplam Talep": total_demand,
        "Toplam Üretim": total_produced,
        "Karşılanmayan Talep": total_unfilled,
        "Ortalama Birim Maliyet": avg_unit_cost,
        "İşçilik Birim Maliyeti": avg_labor_unit,
        "Üretim Birim Maliyeti": avg_prod_unit,
        "Diğer Birim Maliyetler": avg_other_unit
    }

def ayrintili_toplam_maliyetler(total_labor, total_production, total_holding, total_stockout, total_hiring, total_firing):
    return {
        'total_labor': total_labor,
        'total_production': total_production,
        'total_holding': total_holding,
        'total_stockout': total_stockout,
        'total_hiring': total_hiring,
        'total_firing': total_firing
    }

def birim_maliyet_analizi(total_demand, total_produced, total_unfilled, total_cost, total_labor, total_production, total_holding, total_stockout, total_hiring, total_firing):
    if total_produced > 0:
        avg_unit_cost = total_cost / total_produced
        avg_labor_unit = total_labor / total_produced
        avg_prod_unit = total_production / total_produced
        # Include total_stockout in calculation to match maliyet_analizi function
        avg_other_unit = (total_holding + total_stockout + total_hiring + total_firing) / total_produced
    else:
        avg_unit_cost = avg_labor_unit = avg_prod_unit = avg_other_unit = 0
    return {
        'total_demand': total_demand,
        'total_produced': total_produced,
        'total_unfilled': total_unfilled,
        'avg_unit_cost': avg_unit_cost,
        'avg_labor_unit': avg_labor_unit,
        'avg_prod_unit': avg_prod_unit,
        'avg_other_unit': avg_other_unit
    }

def print_results():
    """
    Runs the model and displays formatted results, detailed analyses, and visualizations.
    """
    # Use solve_model to get the model variables
    model_results = solve_model(
        demand, working_days, holding_cost, stockout_cost, hiring_cost, firing_cost,
        daily_hours, labor_per_unit, workers, max_workers, max_workforce_change,
        hourly_wage, production_cost,initial_inventory, safety_stock_ratio
    )

    df = model_results['df']
    min_cost = model_results['min_cost']
    workers_seq = model_results['workers_seq']
    production_seq = model_results['production_seq']
    inventory_seq = model_results['inventory_seq']
    stockout_seq = model_results['unmet_seq']
    total_labor = model_results['total_labor']
    total_production = model_results['total_production']
    total_holding = model_results['total_holding']
    total_stockout = model_results['total_stockout']
    total_hiring = model_results['total_hiring']
    total_firing = model_results['total_firing']

    print(tabulate(df, headers='keys', tablefmt='fancy_grid', showindex=False, numalign='right'))

    detay = ayrintili_toplam_maliyetler(total_labor, total_production, total_holding, total_stockout, total_hiring, total_firing)
    print(f'\nToplam Maliyet: {min_cost:,.2f} TL')
    print(f"\nAyrıntılı Toplam Maliyetler:")
    print(f"- İşçilik Maliyeti Toplamı: {detay['total_labor']:,.2f} TL")
    print(f"- Üretim Maliyeti Toplamı: {detay['total_production']:,.2f} TL")
    print(f"- Stok Maliyeti Toplamı: {detay['total_holding']:,.2f} TL")
    print(f"- Karşılanmayan Talep Maliyeti Toplamı: {detay['total_stockout']:,.2f} TL")
    print(f"- İşe Alım Maliyeti Toplamı: {detay['total_hiring']:,.2f} TL")
    print(f"- İşten Çıkarma Maliyeti Toplamı: {detay['total_firing']:,.2f} TL")

    birim = birim_maliyet_analizi(
        model_results['total_demand'],
        model_results['total_produced'],
        model_results['total_unfilled'],
        min_cost,
        total_labor,
        total_production,
        total_holding,
        total_stockout,  # Add total_stockout parameter
        total_hiring,
        total_firing
    )
    print(f"\nBirim Maliyet Analizi:")
    print(f"- Toplam Talep: {birim['total_demand']:,} birim")
    print(f"- Toplam Üretim: {birim['total_produced']:,} birim ({birim['total_produced']/birim['total_demand']*100:.2f}%)")
    print(f"- Karşılanmayan Talep: {birim['total_unfilled']:,} birim ({birim['total_unfilled']/birim['total_demand']*100:.2f}%)")
    if birim['total_produced'] > 0:
        print(f"- Ortalama Birim Maliyet (Toplam): {birim['avg_unit_cost']:.2f} TL/birim")
        print(f"- Ortalama İşçilik Birim Maliyeti: {birim['avg_labor_unit']:.2f} TL/birim")
        print(f"- Ortalama Üretim Birim Maliyeti: {birim['avg_prod_unit']:.2f} TL/birim")
        print(f"- Diğer Maliyetler (Stok, İşe Alım/Çıkarma): {birim['avg_other_unit']:.2f} TL/birim")
    else:
        print("- Ortalama Birim Maliyet: Hesaplanamadı (0 birim üretildi)")

    # Grafiksel çıktı
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib kütüphanesi eksik. Kurmak için: pip install matplotlib')
        exit(1)
    months_list = list(range(1, months+1))
    fig, ax1 = plt.subplots(figsize=(10,6))
    ax1.bar(months_list, workers_seq, color='skyblue', label='İşçi', alpha=0.7)
    ax1.set_xlabel('Ay')
    ax1.set_ylabel('İşçi Sayısı', color='skyblue')
    ax1.tick_params(axis='y', labelcolor='skyblue')
    ax2 = ax1.twinx()
    ax2.plot(months_list, production_seq, marker='s', label='Üretim', color='green')
    ax2.plot(months_list, inventory_seq, marker='d', label='Stok', color='red')
    ax2.plot(months_list, stockout_seq, marker='x', label='Karşılanmayan Talep', color='black')
    ax2.set_ylabel('Adet', color='gray')
    ax2.tick_params(axis='y', labelcolor='gray')
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    plt.title('Dinamik Programlama Tabanlı Model Sonuçları')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    return model_results

if __name__ == '__main__':
    try:
        from tabulate import tabulate
    except ImportError:
        print('tabulate kütüphanesi eksik. Kurmak için: pip install tabulate')
        exit(1)

    # Call the print_results function to run the model and display results
    print_results()