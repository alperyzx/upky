import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pulp

# Import default parameters from model files
import model1_mixed_planning as model1
import model2_overtime_production as model2
import model3_batch_production as model3
import model4_dynamic_programming as model4
import model5_outsourcing_comparison as model5
import model6_seasonal_planning as model6

# Import ayrintili_toplam_maliyetler and birim_maliyet_analizi from each model
from model1_mixed_planning import ayrintili_toplam_maliyetler as m1_ayrintili, birim_maliyet_analizi as m1_birim
from model2_overtime_production import ayrintili_toplam_maliyetler as m2_ayrintili, birim_maliyet_analizi as m2_birim
from model3_batch_production import ayrintili_toplam_maliyetler as m3_ayrintili, birim_maliyet_analizi as m3_birim
from model4_dynamic_programming import ayrintili_toplam_maliyetler as m4_ayrintili, birim_maliyet_analizi as m4_birim
from model5_outsourcing_comparison import ayrintili_toplam_maliyetler as m5_ayrintili, birim_maliyet_analizi as m5_birim
from model6_seasonal_planning import ayrintili_toplam_maliyetler as m6_ayrintili, birim_maliyet_analizi as m6_birim

st.set_page_config(page_title="Üretim Planlama Modelleri", layout="wide", initial_sidebar_state="expanded")
st.title("Üretim Planlama Modelleri Karar Destek Arayüzü")

model = st.sidebar.selectbox("Model Seçiniz", [
    "Karşılaştırma Tablosu",
    "Karma Planlama (Model 1)",
    "Fazla Mesaili Üretim (Model 2)",
    "Toplu Üretim ve Stoklama (Model 3)",
    "Dinamik Programlama (Model 4)",
    "Dış Kaynak Karşılaştırma (Model 5)",
    "Mevsimsellik ve Dalga (Model 6)"
])

st.sidebar.markdown("---")

def model1_run(demand, working_days, holding_cost, outsourcing_cost, labor_per_unit, hiring_cost, firing_cost, daily_hours, outsourcing_capacity, min_internal_ratio, max_workforce_change, max_outsourcing_ratio, stockout_cost=20, min_workers=10, hourly_wage=10, production_cost=30, overtime_wage_multiplier=1.5, max_overtime_per_worker=20):
    T = len(demand)
    decision_model = pulp.LpProblem('Karma_Planlama_Modeli', pulp.LpMinimize)
    workers = [pulp.LpVariable(f'workers_{t}', lowBound=0, cat='Integer') for t in range(T)]
    internal_production = [pulp.LpVariable(f'internal_prod_{t}', lowBound=0, cat='Integer') for t in range(T)]
    outsourced_production = [pulp.LpVariable(f'outsourced_prod_{t}', lowBound=0, cat='Integer') for t in range(T)]
    inventory = [pulp.LpVariable(f'inventory_{t}', lowBound=0, cat='Integer') for t in range(T)]
    hired = [pulp.LpVariable(f'hired_{t}', lowBound=0, cat='Integer') for t in range(T)]
    fired = [pulp.LpVariable(f'fired_{t}', lowBound=0, cat='Integer') for t in range(T)]
    stockout = [pulp.LpVariable(f'stockout_{t}', lowBound=0, cat='Integer') for t in range(T)]
    cost = (
        pulp.lpSum([
            holding_cost * inventory[t] +
            stockout_cost * stockout[t] +
            outsourcing_cost * outsourced_production[t] +
            hiring_cost * hired[t] +
            firing_cost * fired[t] +
            workers[t] * working_days[t] * daily_hours * hourly_wage +
            production_cost * internal_production[t]
            for t in range(T)
        ])
    )
    decision_model += cost
    for t in range(T):
        if t == 0:
            prev_inventory = 0
        else:
            prev_inventory = inventory[t-1]
        decision_model += (internal_production[t] + outsourced_production[t] + prev_inventory == demand[t] + inventory[t] + stockout[t])
        decision_model += (internal_production[t] >= min_internal_ratio * (internal_production[t] + outsourced_production[t]))
        decision_model += (outsourced_production[t] <= max_outsourcing_ratio * (internal_production[t] + outsourced_production[t]))
        decision_model += (outsourced_production[t] <= outsourcing_capacity)
        decision_model += (internal_production[t] <= workers[t] * working_days[t] * daily_hours / labor_per_unit)
        if t > 0:
            decision_model += (workers[t] - workers[t-1] <= max_workforce_change)
            decision_model += (workers[t-1] - workers[t] <= max_workforce_change)
            decision_model += (workers[t] - workers[t-1] == hired[t] - fired[t])
        else:
            decision_model += (hired[t] - fired[t] == workers[t])
            decision_model += (workers[t] >= min_workers)
        decision_model += (workers[t] <= internal_production[t] * labor_per_unit / (working_days[t] * daily_hours) + 1)
        decision_model += (workers[t] >= 0)
    solver = pulp.PULP_CBC_CMD(msg=0)
    decision_model.solve(solver)
    results = []
    for t in range(T):
        internal_prod_cost = int(internal_production[t].varValue) * production_cost
        outsourcing_cost_val = int(outsourced_production[t].varValue) * outsourcing_cost
        labor_cost = int(workers[t].varValue) * working_days[t] * daily_hours * hourly_wage
        results.append([
            t+1,
            int(workers[t].varValue),
            int(internal_production[t].varValue),
            int(outsourced_production[t].varValue),
            int(inventory[t].varValue),
            int(hired[t].varValue),
            int(fired[t].varValue),
            int(stockout[t].varValue),
            internal_prod_cost,
            outsourcing_cost_val,
            labor_cost
        ])
    df = pd.DataFrame(results, columns=["Ay", "İşçi", "İç Üretim", "Fason", "Stok", "Alım", "Çıkış", "Karşılanmayan Talep", "İç Üretim Maliyeti", "Fason Üretim Maliyeti", "İşçilik Maliyeti"])
    toplam_maliyet = pulp.value(decision_model.objective)
    return df, toplam_maliyet

def model2_run(demand, working_days, holding_cost, labor_per_unit, fixed_workers, daily_hours, overtime_wage_multiplier, max_overtime_per_worker, stockout_cost, normal_hourly_wage, production_cost=12):
    months = len(demand)
    production = np.zeros(months)
    overtime_hours = np.zeros(months)
    inventory = np.zeros(months)
    prev_inventory = 0
    results = []
    total_cost = 0
    overtime_cost_per_hour = normal_hourly_wage * overtime_wage_multiplier
    for t in range(months):
        normal_prod = fixed_workers * working_days[t] * daily_hours / labor_per_unit
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
        production_cost_val = prod * production_cost
        total_cost += holding + stockout + overtime + normal_labor_cost + production_cost_val
        results.append([
            t+1, fixed_workers, prod, ot_hours, inventory[t], holding, stockout, overtime, normal_labor_cost, production_cost_val
        ])
        prev_inventory = inventory[t]
    headers = [
        'Ay', 'İşçi', 'Üretim', 'Fazla Mesai', 'Stok', 'Stok Maliyeti', 'Stoksuzluk Maliyeti', 'Fazla Mesai Maliyeti', 'Normal İşçilik Maliyeti', 'Üretim Maliyeti'
    ]
    df = pd.DataFrame(results, columns=headers)
    return df, total_cost

def model3_run(demand, working_days, holding_cost, stockout_cost, fixed_workers, production_rate, daily_hours, worker_monthly_cost=None, production_cost=20):
    months = len(demand)
    monthly_capacity = fixed_workers * daily_hours * working_days * production_rate
    production = monthly_capacity
    inventory = np.zeros(months)
    cost = 0
    prev_inventory = 0
    results = []
    if worker_monthly_cost is None:
        worker_monthly_cost = fixed_workers * np.mean(working_days) * daily_hours * 10
    for t in range(months):
        inventory[t] = prev_inventory + production[t] - demand[t]
        holding = max(inventory[t], 0) * holding_cost
        stockout = abs(min(inventory[t], 0)) * stockout_cost
        unfilled = abs(min(inventory[t], 0))
        labor_cost = fixed_workers * worker_monthly_cost if worker_monthly_cost else fixed_workers * working_days[t] * daily_hours * 10
        prod_cost = production[t] * production_cost
        cost += holding + stockout + labor_cost + prod_cost
        results.append([
            t+1, production[t], inventory[t], holding, stockout, labor_cost, prod_cost, unfilled
        ])
        prev_inventory = inventory[t]
    df = pd.DataFrame(results, columns=["Ay", "Üretim", "Stok", "Stok Maliyeti", "Stoksuzluk Maliyeti", "İşçilik Maliyeti", "Üretim Maliyeti", "Karşılanmayan Talep"])
    # Ayrıntılı toplam maliyetler
    total_holding = df["Stok Maliyeti"].sum()
    total_stockout = df["Stoksuzluk Maliyeti"].sum()
    total_labor = df["İşçilik Maliyeti"].sum()
    total_production_cost = df["Üretim Maliyeti"].sum()
    total_demand = demand.sum()
    total_produced = production.sum()
    total_unfilled = df["Karşılanmayan Talep"].sum()
    # Birim maliyetler
    if total_produced > 0:
        avg_unit_cost = cost / total_produced
        avg_labor_unit = total_labor / total_produced
        avg_prod_unit = total_production_cost / total_produced
        avg_other_unit = (total_holding + total_stockout) / total_produced
    else:
        avg_unit_cost = avg_labor_unit = avg_prod_unit = avg_other_unit = 0
    return df, cost, total_holding, total_stockout, total_labor, total_production_cost, total_demand, total_produced, total_unfilled, avg_unit_cost, avg_labor_unit, avg_prod_unit, avg_other_unit

def model4_run(demand, working_days, holding_cost, hiring_cost, firing_cost, daily_hours, labor_per_unit, min_workers, max_workers, max_workforce_change, hourly_wage, stockout_cost=20, production_cost=30):
    months = len(demand)
    def prod_capacity(workers, t):
        return workers * working_days[t] * daily_hours / labor_per_unit
    cost_table = np.full((months+1, max_workers+1), np.inf)
    backtrack = np.full((months+1, max_workers+1), -1, dtype=int)
    cost_table[0, min_workers:max_workers+1] = 0
    for t in range(months):
        for prev_w in range(min_workers, max_workers+1):
            if cost_table[t, prev_w] < np.inf:
                for w in range(max(min_workers, prev_w-max_workforce_change), min(max_workers, prev_w+max_workforce_change)+1):
                    capacity = prod_capacity(w, t)
                    unmet = max(0, demand[t] - capacity)
                    inventory = max(0, capacity - demand[t])
                    hire = max(0, w - prev_w) * hiring_cost
                    fire = max(0, prev_w - w) * firing_cost
                    holding = inventory * holding_cost
                    stockout = unmet * stockout_cost
                    labor = w * working_days[t] * daily_hours * hourly_wage
                    actual_prod = min(capacity, demand[t])
                    prod_cost = actual_prod * production_cost
                    total_cost = cost_table[t, prev_w] + hire + fire + holding + stockout + labor + prod_cost
                    if total_cost < cost_table[t+1, w]:
                        cost_table[t+1, w] = total_cost
                        backtrack[t+1, w] = prev_w
    final_workers = np.argmin(cost_table[months])
    min_cost = cost_table[months, final_workers]
    workers_seq = []
    w = final_workers
    for t in range(months, 0, -1):
        workers_seq.append(w)
        w = backtrack[t, w]
    workers_seq = workers_seq[::-1]
    production_seq = []
    inventory_seq = []
    labor_cost_seq = []
    unmet_seq = []
    stockout_cost_seq = []
    prod_cost_seq = []
    hiring_seq = []
    firing_seq = []
    for t, w in enumerate(workers_seq):
        cap = prod_capacity(w, t)
        prod = min(cap, demand[t])
        unmet = max(0, demand[t] - cap)
        inventory = max(0, cap - demand[t])
        labor_cost = w * working_days[t] * daily_hours * hourly_wage
        prod_cost = prod * production_cost
        if t > 0:
            hire = max(0, w - workers_seq[t-1]) * hiring_cost
            fire = max(0, workers_seq[t-1] - w) * firing_cost
        else:
            hire = w * hiring_cost
            fire = 0
        production_seq.append(prod)
        inventory_seq.append(inventory)
        labor_cost_seq.append(labor_cost)
        unmet_seq.append(unmet)
        stockout_cost_seq.append(unmet * stockout_cost)
        prod_cost_seq.append(prod_cost)
        hiring_seq.append(hire)
        firing_seq.append(fire)
    results = []
    total_labor = 0
    total_production = 0
    total_holding = 0
    total_stockout = 0
    total_hiring = 0
    total_firing = 0
    for t in range(months):
        holding = inventory_seq[t] * holding_cost
        stockout = unmet_seq[t] * stockout_cost
        labor_cost = labor_cost_seq[t]
        prod_cost = prod_cost_seq[t]
        hire = hiring_seq[t]
        fire = firing_seq[t]
        total_labor += labor_cost
        total_production += prod_cost
        total_holding += holding
        total_stockout += stockout
        total_hiring += hire
        total_firing += fire
        results.append([
            t+1, workers_seq[t], production_seq[t], inventory_seq[t], unmet_seq[t],
            labor_cost, prod_cost, hire, fire, holding, stockout
        ])
    df = pd.DataFrame(results, columns=[
        'Ay', 'İşçi', 'Üretim', 'Stok', 'Karşılanmayan Talep',
        'İşçilik Maliyeti', 'Üretim Maliyeti', 'İşe Alım Maliyeti',
        'İşten Çıkarma Maliyeti', 'Stok Maliyeti', 'Stoksuzluk Maliyeti'
    ])
    total_demand = sum(demand)
    total_produced = sum(production_seq)
    total_unfilled = sum(unmet_seq)
    # Birim maliyetler
    if total_produced > 0:
        avg_unit_cost = min_cost / total_produced
        avg_labor_unit = total_labor / total_produced
        avg_prod_unit = total_production / total_produced
        avg_other_unit = (total_holding + total_hiring + total_firing) / total_produced
    else:
        avg_unit_cost = avg_labor_unit = avg_prod_unit = avg_other_unit = 0
    return df, min_cost, total_labor, total_production, total_holding, total_stockout, total_hiring, total_firing, total_demand, total_produced, total_unfilled, avg_unit_cost, avg_labor_unit, avg_prod_unit, avg_other_unit

def model5_run(demand, holding_cost, cost_supplier_A, cost_supplier_B, capacity_supplier_A, capacity_supplier_B, working_days, stockout_cost):
    months = len(demand)
    decision_model = pulp.LpProblem('Dis_Kaynak_Karsilastirma', pulp.LpMinimize)
    out_A = [pulp.LpVariable(f'out_A_{t}', lowBound=0, cat='Integer') for t in range(months)]
    out_B = [pulp.LpVariable(f'out_B_{t}', lowBound=0, cat='Integer') for t in range(months)]
    inventory = [pulp.LpVariable(f'inventory_{t}', lowBound=0, cat='Integer') for t in range(months)]
    stockout = [pulp.LpVariable(f'stockout_{t}', lowBound=0, cat='Integer') for t in range(months)]
    decision_model += pulp.lpSum([
        cost_supplier_A * out_A[t] +
        cost_supplier_B * out_B[t] +
        holding_cost * inventory[t] +
        stockout_cost * stockout[t]
        for t in range(months)
    ])
    for t in range(months):
        if t == 0:
            prev_inventory = 0
        else:
            prev_inventory = inventory[t-1]
        decision_model += (out_A[t] + out_B[t] + prev_inventory + stockout[t] == demand[t] + inventory[t])
        decision_model += (out_A[t] <= capacity_supplier_A)
        decision_model += (out_B[t] <= capacity_supplier_B)
        decision_model += (out_A[t] >= 0)
        decision_model += (out_B[t] >= 0)
        decision_model += (stockout[t] >= 0)
    solver = pulp.PULP_CBC_CMD(msg=0)
    decision_model.solve(solver)
    table = []
    for t in range(months):
        table.append([
            t+1,
            int(out_A[t].varValue),
            int(out_B[t].varValue),
            int(inventory[t].varValue),
            int(stockout[t].varValue)
        ])
    df = pd.DataFrame(table, columns=['Ay', 'Tedarikçi A', 'Tedarikçi B', 'Stok', 'Karşılanmayan Talep'])
    toplam_maliyet = pulp.value(decision_model.objective)
    return df, toplam_maliyet

def model6_run(demand, holding_cost, stockout_cost, production_cost, labor_per_unit, hourly_wage, daily_hours):
    demand = np.array(demand)  # Listeyi NumPy array'e çevir
    months = len(demand)
    max_production = int(np.mean(demand) + np.std(demand))  # Doğru formül
    avg_working_days = np.mean([22, 20, 23, 19, 21, 19, 22, 22, 22, 21, 21, 21])
    needed_workers = int(np.ceil(max_production * labor_per_unit / (daily_hours * np.mean(working_days))))
    monthly_labor_cost = needed_workers * np.mean(working_days) * daily_hours * hourly_wage
    model = pulp.LpProblem('Mevsimsel_Stok_Optimizasyonu', pulp.LpMinimize)
    y_production = [pulp.LpVariable(f'production_{t}', lowBound=0, cat='Integer') for t in range(months)]
    y_inventory = [pulp.LpVariable(f'inventory_{t}', lowBound=0, cat='Integer') for t in range(months)]
    y_stockout = [pulp.LpVariable(f'stockout_{t}', lowBound=0, cat='Integer') for t in range(months)]
    model += pulp.lpSum([
        production_cost * y_production[t] +
        holding_cost * y_inventory[t] +
        stockout_cost * y_stockout[t] +
        monthly_labor_cost
        for t in range(months)
    ]) + model6.hiring_cost * needed_workers  # Add hiring cost directly to objective
    for t in range(months):
        model += y_production[t] <= max_production
        if t == 0:
            prev_inventory = 0
        else:
            prev_inventory = y_inventory[t-1]
        model += prev_inventory + y_production[t] + y_stockout[t] == demand[t] + y_inventory[t]
        model += y_inventory[t] >= 0
        model += y_stockout[t] >= 0
    solver = pulp.PULP_CBC_CMD(msg=0)
    model.solve(solver)
    results = []
    total_holding = 0
    total_stockout = 0
    total_production_cost = 0
    total_labor_cost = 0
    for t in range(months):
        labor_cost = monthly_labor_cost
        holding = int(y_inventory[t].varValue) * holding_cost
        stockout = int(y_stockout[t].varValue) * stockout_cost
        prod_cost = int(y_production[t].varValue) * production_cost
        total_holding += holding
        total_stockout += stockout
        total_production_cost += prod_cost
        total_labor_cost += labor_cost
        results.append([
            t+1,
            demand[t],
            int(y_production[t].varValue),
            int(y_inventory[t].varValue),
            int(y_stockout[t].varValue),
            holding,
            stockout,
            prod_cost,
            labor_cost
        ])
    df = pd.DataFrame(results, columns=[
        'Ay', 'Talep', 'Üretim', 'Stok', 'Stoksuzluk', 'Stok Maliyeti', 'Stoksuzluk Maliyeti', 'Üretim Maliyeti', 'İşçilik Maliyeti'
    ])
    # Calculate the total cost from the model's objective value
    total_cost = pulp.value(model.objective)
    return df, total_cost, needed_workers, max_production

if model == "Karma Planlama (Model 1)":
    st.header("Karma Planlama (Model 1)")
    with st.sidebar:
        demand = st.text_input("Aylık Talep (virgülle ayrılmış)", ", ".join(map(str, model1.demand)), key="m1_demand")
        demand = [int(x.strip()) for x in demand.split(",") if x.strip()]
        working_days = st.text_input("Aylık Çalışma Günü (virgülle ayrılmış)", ", ".join(map(str, model1.working_days)), key="m1_days")
        working_days = [int(x.strip()) for x in working_days.split(",") if x.strip()]
        if len(demand) != len(working_days):
            st.error(f"Talep ve çalışma günü uzunlukları eşit olmalı. Şu an talep: {len(demand)}, çalışma günü: {len(working_days)}.")
            st.stop()
        holding_cost = st.number_input("Stok Maliyeti (TL)", min_value=1, max_value=100, value=int(model1.holding_cost), step=1, key="m1_holding")
        outsourcing_cost = st.number_input("Fason Maliyeti (TL)", min_value=1, max_value=100, value=int(model1.outsourcing_cost), step=1, key="m1_outsourcing")
        stockout_cost = st.number_input("Karşılanmayan Talep Maliyeti (TL/adet)", min_value=1, max_value=100, value=int(model1.stockout_cost), step=1, key="m1_stockout")
        labor_per_unit = st.number_input("Birim İşgücü (saat)", min_value=0.1, max_value=10.0, value=float(model1.labor_per_unit), step=0.1, key="m1_labor")
        hiring_cost = st.number_input("İşçi Alım Maliyeti (TL)", min_value=0, max_value=5000, value=int(model1.hiring_cost), step=1, key="m1_hire")
        firing_cost = st.number_input("İşçi Çıkarma Maliyeti (TL)", min_value=0, max_value=5000, value=int(model1.firing_cost), step=1, key="m1_fire")
        daily_hours = st.number_input("Günlük Çalışma Saati", min_value=1, max_value=24, value=int(model1.daily_hours), step=1, key="m1_daily")
        outsourcing_capacity = st.number_input("Fason Kapasitesi (adet)", min_value=1, max_value=10000, value=int(model1.outsourcing_capacity), step=1, key="m1_capacity")
        min_internal_ratio = st.slider("En Az İç Üretim Oranı (%)", min_value=0, max_value=100, value=int(model1.min_internal_ratio*100), key="m1_min_internal") / 100
        max_workforce_change = st.number_input("İşgücü Değişim Sınırı (kişi)", min_value=1, max_value=100, value=int(model1.max_workforce_change), step=1, key="m1_max_workforce")
        max_outsourcing_ratio = st.slider("En Fazla Fason Oranı (%)", min_value=0, max_value=100, value=int(model1.max_outsourcing_ratio*100), key="m1_max_outsourcing") / 100
        min_workers = st.number_input("Başlangıç Minimum İşçi Sayısı", min_value=1, max_value=100, value=10, step=1, key="m1_min_workers")
        overtime_wage_multiplier = st.number_input("Fazla Mesai Ücret Çarpanı", min_value=1.0, max_value=5.0, value=float(model1.overtime_wage_multiplier), step=0.1, key="m1_overtime_multiplier")
        max_overtime_per_worker = st.number_input("Maks. Fazla Mesai (saat/işçi)", min_value=0, max_value=100, value=int(model1.max_overtime_per_worker), step=1, key="m1_max_overtime")
        run_model = st.button("Modeli Çalıştır", key="m1_run")
    if run_model:
        df, toplam_maliyet = model1_run(
            demand, working_days, holding_cost, outsourcing_cost, labor_per_unit, hiring_cost, firing_cost, daily_hours,
            outsourcing_capacity, min_internal_ratio, max_workforce_change, max_outsourcing_ratio, stockout_cost, min_workers,
            hourly_wage=10, production_cost=30, overtime_wage_multiplier=overtime_wage_multiplier, max_overtime_per_worker=max_overtime_per_worker
        )
        st.subheader("Sonuç Tablosu")
        st.dataframe(df, use_container_width=True)
        st.success(f"Toplam Maliyet: {toplam_maliyet:,.2f} TL")
        st.subheader("Grafiksel Sonuçlar")
        fig, ax1 = plt.subplots(figsize=(10,6))
        months = df["Ay"].tolist()
        ax1.bar(months, df["İşçi"], color='skyblue', label='İşçi', alpha=0.7)
        ax1.set_xlabel('Ay')
        ax1.set_ylabel('İşçi Sayısı', color='skyblue')
        ax1.tick_params(axis='y', labelcolor='skyblue')
        ax2 = ax1.twinx()
        ax2.plot(months, df["İç Üretim"], marker='s', label='İç Üretim', color='green')
        ax2.plot(months, df["Fason"], marker='^', label='Fason', color='orange')
        ax2.plot(months, df["Stok"], marker='d', label='Stok', color='red')
        ax2.plot(months, df["Karşılanmayan Talep"], marker='x', label='Karşılanmayan Talep', color='black')
        ax2.set_ylabel('Adet', color='gray')
        ax2.tick_params(axis='y', labelcolor='gray')
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        plt.title('Karma Planlama Modeli Sonuçları')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        st.pyplot(fig)

        # Ayrıntılı Toplam Maliyetler ve Birim Maliyet Analizi
        detay = m1_ayrintili(
            len(demand), model1.internal_production, model1.outsourced_production, model1.inventory, model1.hired, model1.fired, model1.stockout, model1.overtime_hours, working_days, daily_hours, model1.hourly_wage, model1.production_cost, model1.outsourcing_cost, holding_cost, stockout_cost, hiring_cost, firing_cost, model1.overtime_wage_multiplier
        )
        st.subheader("Ayrıntılı Toplam Maliyetler")
        st.markdown(f"- İç Üretim İşçilik Maliyeti Toplamı: {detay['total_internal_labor']:,.2f} TL")
        st.markdown(f"- İç Üretim Birim Maliyeti Toplamı: {detay['total_internal_prod']:,.2f} TL")
        st.markdown(f"- Fason Üretim Maliyeti Toplamı: {detay['total_outsource']:,.2f} TL")
        st.markdown(f"- Stok Maliyeti Toplamı: {detay['total_holding']:,.2f} TL")
        st.markdown(f"- Karşılanmayan Talep Maliyeti Toplamı: {detay['total_stockout']:,.2f} TL")
        st.markdown(f"- İşe Alım Maliyeti Toplamı: {detay['total_hiring']:,.2f} TL")
        st.markdown(f"- İşten Çıkarma Maliyeti Toplamı: {detay['total_firing']:,.2f} TL")
        st.markdown(f"- Fazla Mesai Maliyeti Toplamı: {detay['total_overtime']:,.2f} TL")
        birim = m1_birim(
            len(demand), demand, model1.internal_production, model1.outsourced_production, model1.stockout, detay['total_internal_labor'], detay['total_internal_prod'], detay['total_outsource'], detay['total_holding'], detay['total_hiring'], detay['total_firing'], toplam_maliyet
        )
        st.subheader("Birim Maliyet Analizi")
        st.markdown(f"- Toplam Talep: {birim['total_demand']:,} birim")
        st.markdown(f"- Toplam İç Üretim: {birim['total_internal_produced']:,} birim ({birim['total_internal_produced']/birim['total_demand']*100:.2f}%)")
        st.markdown(f"- Toplam Fason Üretim: {birim['total_outsourced']:,} birim ({birim['total_outsourced']/birim['total_demand']*100:.2f}%)")
        st.markdown(f"- Toplam Üretim: {birim['total_produced']:,} birim ({birim['total_produced']/birim['total_demand']*100:.2f}%)")
        st.markdown(f"- Karşılanmayan Talep: {birim['total_unfilled']:,} birim ({birim['total_unfilled']/birim['total_demand']*100:.2f}%)")
        if birim['total_produced'] > 0:
            st.markdown(f"- Ortalama Birim Maliyet (Toplam): {birim['avg_unit_cost']:.2f} TL/birim")
            if birim['total_internal_produced'] > 0:
                st.markdown(f"- İç Üretim Birim Maliyeti: {birim['internal_unit_cost']:.2f} TL/birim")
                st.markdown(f"  * İşçilik Birim Maliyeti: {birim['internal_labor_unit_cost']:.2f} TL/birim")
                st.markdown(f"  * Üretim Birim Maliyeti: {birim['internal_prod_unit_cost']:.2f} TL/birim")
            if birim['total_outsourced'] > 0:
                st.markdown(f"- Fason Üretim Birim Maliyeti: {birim['outsourced_unit_cost']:.2f} TL/birim")
            st.markdown(f"- Diğer Maliyetler (Stok, İşe Alım/Çıkarma): {birim['other_unit_cost']:.2f} TL/birim")
        else:
            st.markdown("- Ortalama Birim Maliyet: Hesaplanamadı (0 birim üretildi)")

if model == "Fazla Mesaili Üretim (Model 2)":
    st.header("Fazla Mesaili Üretim (Model 2)")
    with st.sidebar:
        demand = st.text_input("Aylık Talep (virgülle ayrılmış)", ", ".join(map(str, model2.demand)), key="m2_demand")
        demand = np.array([int(x.strip()) for x in demand.split(",") if x.strip()])
        working_days = st.text_input("Aylık Çalışma Günü (virgülle ayrılmış)", ", ".join(map(str, model2.working_days)), key="m2_days")
        working_days = np.array([int(x.strip()) for x in working_days.split(",") if x.strip()])
        if len(demand) != len(working_days):
            st.error(f"Talep ve çalışma günü uzunlukları eşit olmalı. Şu an talep: {len(demand)}, çalışma günü: {len(working_days)}.")
            st.stop()
        holding_cost = st.number_input("Stok Maliyeti (TL)", min_value=1, max_value=100, value=int(model2.holding_cost), key="m2_holding", step=1)
        labor_per_unit = st.number_input("Birim İşgücü (saat)", min_value=0.1, max_value=10.0, value=float(model2.labor_per_unit), key="m2_labor", step=0.1)
        fixed_workers = st.number_input("Sabit İşçi Sayısı", min_value=1, max_value=200, value=model2.fixed_workers, key="m2_workers", step=1)
        daily_hours = st.number_input("Günlük Çalışma Saati", min_value=1, max_value=24, value=model2.daily_hours, key="m2_daily", step=1)
        overtime_wage_multiplier = st.number_input("Fazla Mesai Ücret Çarpanı", min_value=1.0, max_value=5.0, value=model2.overtime_wage_multiplier, step=0.1, key="m2_overtime_multiplier")
        max_overtime_per_worker = st.number_input("Maks. Fazla Mesai (saat/işçi)", min_value=0, max_value=100, value=model2.max_overtime_per_worker, step=1, key="m2_max_overtime")
        stockout_cost = st.number_input("Stoksuzluk Maliyeti (TL/adet)", min_value=1, max_value=100, value=model2.stockout_cost, step=1, key="m2_stockout")
        normal_hourly_wage = st.number_input("Normal Saatlik İşçilik Maliyeti (TL)", min_value=1, max_value=1000, value=model2.normal_hourly_wage, step=1, key="m2_normal_wage")
        production_cost = st.number_input("Üretim Maliyeti (TL)", min_value=1, max_value=100, value=model2.production_cost, key="m2_prod_cost", step=1)
        run_model = st.button("Modeli Çalıştır", key="m2_run")
    if run_model:
        df_table, total_cost = model2_run(
            demand, working_days, holding_cost, labor_per_unit, fixed_workers, daily_hours,
            overtime_wage_multiplier, max_overtime_per_worker, stockout_cost, normal_hourly_wage, production_cost
        )
        st.subheader("Sonuç Tablosu")
        st.dataframe(df_table, use_container_width=True)
        st.success(f"Toplam Maliyet: {total_cost:,.2f} TL")
        st.subheader("Grafiksel Sonuçlar")
        fig, ax1 = plt.subplots(figsize=(10,6))
        months_list = df_table["Ay"].tolist()
        # Fazla mesai saatlerini bar olarak göster
        ax1.bar(months_list, df_table["Fazla Mesai"], color='orange', label='Fazla Mesai (saat)', alpha=0.7)
        ax1.set_xlabel('Ay')
        ax1.set_ylabel('Fazla Mesai (saat)', color='orange')
        ax1.tick_params(axis='y', labelcolor='orange')
        ax2 = ax1.twinx()
        ax2.plot(months_list, df_table["Üretim"], marker='s', label='Üretim', color='green')
        ax2.plot(months_list, df_table["Stok"], marker='d', label='Stok', color='red')
        ax2.set_ylabel('Adet', color='gray')
        ax2.tick_params(axis='y', labelcolor='gray')
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        plt.title('Fazla Mesaili Üretim Modeli Sonuçları')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        st.pyplot(fig)

        # Ayrıntılı Toplam Maliyetler ve Birim Maliyet Analizi
        detay = m2_ayrintili(
            df_table['Stok Maliyeti'].sum(), df_table['Stoksuzluk Maliyeti'].sum(), df_table['Fazla Mesai Maliyeti'].sum() if 'Fazla Mesai Maliyeti' in df_table.columns else 0, df_table['Normal İşçilik Maliyeti'].sum(), df_table['Üretim Maliyeti'].sum()
        )
        st.subheader("Ayrıntılı Toplam Maliyetler")
        st.markdown(f"- Stok Maliyeti Toplamı: {detay['total_holding']:,.2f} TL")
        st.markdown(f"- Stoksuzluk Maliyeti Toplamı: {detay['total_stockout']:,.2f} TL")
        st.markdown(f"- Fazla Mesai Maliyeti Toplamı: {detay['total_overtime']:,.2f} TL")
        st.markdown(f"- Normal İşçilik Maliyeti Toplamı: {detay['total_normal_labor']:,.2f} TL")
        st.markdown(f"- Üretim Maliyeti Toplamı: {detay['total_production']:,.2f} TL")
        birim = m2_birim(
            demand, df_table['Üretim'], df_table['Stok'], total_cost, detay['total_normal_labor'], detay['total_overtime'], detay['total_production'], detay['total_holding'], detay['total_stockout'], production_cost
        )
        st.subheader("Birim Maliyet Analizi")
        st.markdown(f"- Toplam Talep: {birim['total_demand']:,} birim")
        st.markdown(f"- Toplam Üretim: {birim['total_produced']:,} birim ({birim['total_produced']/birim['total_demand']*100:.2f}%)")
        if 'Fazla Mesai' in df_table.columns:
            st.markdown(f"- Toplam Fazla Mesai: {df_table['Fazla Mesai'].sum():,.2f} saat")
        if birim['total_produced'] > 0:
            st.markdown(f"- Ortalama Birim Maliyet (Toplam): {birim['avg_unit_cost']:.2f} TL/birim")
            st.markdown(f"- İşçilik Birim Maliyeti: {birim['labor_unit_cost']:.2f} TL/birim")
            st.markdown(f"  * Normal İşçilik: {birim['normal_labor_unit_cost']:.2f} TL/birim")
            st.markdown(f"  * Fazla Mesai: {birim['overtime_unit_cost']:.2f} TL/birim")
            st.markdown(f"- Üretim Birim Maliyeti: {birim['prod_unit_cost']:.2f} TL/birim")
            st.markdown(f"- Diğer Maliyetler (Stok, Stoksuzluk): {birim['other_unit_cost']:.2f} TL/birim")
        else:
            st.markdown("- Ortalama Birim Maliyet: Hesaplanamadı (0 birim üretildi)")

if model == "Toplu Üretim ve Stoklama (Model 3)":
    st.header("Toplu Üretim ve Stoklama (Model 3)")
    with st.sidebar:
        demand = st.text_input("Aylık Talep (virgülle ayrılmış)", ", ".join(map(str, model3.demand)), key="m3_demand")
        demand = np.array([int(x.strip()) for x in demand.split(",") if x.strip()])
        working_days = st.text_input("Aylık Çalışma Günü (virgülle ayrılmış)", ", ".join(map(str, model3.working_days)), key="m3_days")
        working_days = np.array([int(x.strip()) for x in working_days.split(",") if x.strip()])
        if len(demand) != len(working_days):
            st.error(f"Talep ve çalışma günü uzunlukları eşit olmalı. Şu an talep: {len(demand)}, çalışma günü: {len(working_days)}.")
            st.stop()
        holding_cost = st.number_input("Stok Maliyeti (TL)", min_value=1, max_value=100, value=int(model3.holding_cost), key="m3_holding", step=1)
        stockout_cost = st.number_input("Stoksuzluk Maliyeti (TL)", min_value=1, max_value=100, value=model3.stockout_cost, key="m3_stockout", step=1)
        fixed_workers = st.number_input("Sabit İşçi Sayısı", min_value=1, max_value=200, value=model3.fixed_workers, key="m3_workers", step=1)
        production_rate = st.number_input("Üretim Hızı (adet/saat)", min_value=0.1, max_value=10.0, value=model3.production_rate, step=0.1, key="m3_prod_rate")
        daily_hours = st.number_input("Günlük Çalışma Saati", min_value=1, max_value=24, value=model3.daily_hours, key="m3_daily", step=1)
        worker_monthly_cost = st.number_input("Aylık İşçi Ücreti (TL)", min_value=1, max_value=100000, value=model3.worker_monthly_cost, key="m3_worker_monthly_cost", step=1)
        production_cost = st.number_input("Üretim Maliyeti (TL)", min_value=1, max_value=100, value=model3.production_cost, key="m3_prod_cost", step=1)
        run_model = st.button("Modeli Çalıştır", key="m3_run")
    if run_model:
        df, cost, total_holding, total_stockout, total_labor, total_production_cost, total_demand, total_produced, total_unfilled, avg_unit_cost, avg_labor_unit, avg_prod_unit, avg_other_unit = model3_run(demand, working_days, holding_cost, stockout_cost, fixed_workers, production_rate, daily_hours, worker_monthly_cost, production_cost)
        st.subheader("Sonuç Tablosu")
        st.dataframe(df, use_container_width=True)
        st.success(f"Toplam Maliyet: {cost:,.2f} TL")
        st.subheader("Grafiksel Sonuçlar")
        months_list = df["Ay"].tolist()
        fig, ax = plt.subplots(figsize=(10,6))
        ax.bar(months_list, df["Üretim"], color='skyblue', label='Üretim', alpha=0.7)
        ax.plot(months_list, df["Stok"], marker='d', label='Stok', color='red')
        ax.set_xlabel('Ay')
        ax.set_ylabel('Adet')
        ax.legend()
        plt.title('Toplu Üretim ve Stoklama Modeli Sonuçları')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        st.pyplot(fig)

        # Ayrıntılı toplam maliyetler ve Birim Maliyet Analizi
        detay = m3_ayrintili(df)
        st.subheader("Ayrıntılı Toplam Maliyetler")
        st.markdown(f"- Stok Maliyeti Toplamı: {detay['total_holding']:,.2f} TL")
        st.markdown(f"- Stoksuzluk Maliyeti Toplamı: {detay['total_stockout']:,.2f} TL")
        st.markdown(f"- İşçilik Maliyeti Toplamı: {detay['total_labor']:,.2f} TL")
        st.markdown(f"- Üretim Maliyeti Toplamı: {detay['total_production_cost']:,.2f} TL")
        birim = m3_birim(demand, df['Üretim'], df['Stok'], cost, df, fixed_workers, len(demand))
        st.subheader("Birim Maliyet Analizi")
        st.markdown(f"- Toplam Talep: {birim['total_demand']:,} birim")
        st.markdown(f"- Toplam Üretim: {birim['total_produced']:,} birim ({birim['total_produced']/birim['total_demand']*100:.2f}%)")
        st.markdown(f"- Karşılanmayan Talep: {birim['total_unfilled']:,} birim ({birim['total_unfilled']/birim['total_demand']*100:.2f}%)")
        if birim['total_produced'] > 0:
            st.markdown(f"- Ortalama Birim Maliyet (Toplam): {birim['avg_unit_cost']:.2f} TL/birim")
            st.markdown(f"- İşçilik Birim Maliyeti: {birim['labor_unit_cost']:.2f} TL/birim")
            st.markdown(f"- Üretim Birim Maliyeti: {birim['prod_unit_cost']:.2f} TL/birim")
            st.markdown(f"- Diğer Maliyetler (Stok, Stoksuzluk): {birim['other_unit_cost']:.2f} TL/birim")
        else:
            st.markdown("- Ortalama Birim Maliyet: Hesaplanamadı (0 birim üretildi)")

if model == "Dinamik Programlama (Model 4)":
    st.header("Dinamik Programlama Tabanlı (Model 4)")
    with st.sidebar:
        demand = st.text_input("Aylık Talep (virgülle ayrılmış)", ", ".join(map(str, model4.demand)), key="m4_demand")
        demand = np.array([int(x.strip()) for x in demand.split(",") if x.strip()])
        working_days = st.text_input("Aylık Çalışma Günü (virgülle ayrılmış)", ", ".join(map(str, model4.working_days)), key="m4_days")
        working_days = np.array([int(x.strip()) for x in working_days.split(",") if x.strip()])
        if len(demand) != len(working_days):
            st.error(f"Talep ve çalışma günü uzunlukları eşit olmalı. Şu an talep: {len(demand)}, çalışma günü: {len(working_days)}.")
            st.stop()
        holding_cost = st.number_input("Stok Maliyeti (TL)", 1, 100, int(model4.holding_cost), key="m4_holding")
        stockout_cost = st.number_input("Karşılanmayan Talep Maliyeti (TL/adet)", 1, 100, int(model4.stockout_cost), key="m4_stockout")
        hiring_cost = st.number_input("İşçi Alım Maliyeti (TL)", 0, 5000, int(model4.hiring_cost), key="m4_hire")
        firing_cost = st.number_input("İşçi Çıkarma Maliyeti (TL)", 0, 5000, int(model4.firing_cost), key="m4_fire")
        daily_hours = st.number_input("Günlük Çalışma Saati", 1, 24, int(model4.daily_hours), key="m4_daily")
        labor_per_unit = st.number_input("Birim İşgücü (saat)", 0.1, 10.0, float(model4.labor_per_unit), key="m4_labor")
        min_workers = st.number_input("Minimum İşçi", 1, 100, int(model4.min_workers), key="m4_min_workers")
        max_workers = st.number_input("Maksimum İşçi", 1, 100, int(model4.max_workers), key="m4_max_workers")
        max_workforce_change = st.number_input("Aylık İşgücü Değişim Sınırı", 1, 100, int(model4.max_workforce_change), key="m4_max_workforce")
        run_model = st.button("Modeli Çalıştır", key="m4_run")
    if run_model:
        df, min_cost, total_labor, total_production, total_holding, total_stockout, total_hiring, total_firing, total_demand, total_produced, total_unfilled, avg_unit_cost, avg_labor_unit, avg_prod_unit, avg_other_unit = model4_run(demand, working_days, holding_cost, hiring_cost, firing_cost, daily_hours, labor_per_unit, min_workers, max_workers, max_workforce_change, hourly_wage=10, stockout_cost=stockout_cost)
        st.subheader("Sonuç Tablosu")
        st.dataframe(df, use_container_width=True)
        st.success(f"Toplam Maliyet: {min_cost:,.2f} TL")
        st.subheader("Grafiksel Sonuçlar")
        months_list = list(range(1, len(demand)+1))
        fig, ax1 = plt.subplots(figsize=(10,6))
        ax1.bar(months_list, df['İşçi'], color='skyblue', label='İşçi', alpha=0.7)
        ax1.set_xlabel('Ay')
        ax1.set_ylabel('İşçi Sayısı', color='skyblue')
        ax1.tick_params(axis='y', labelcolor='skyblue')
        ax2 = ax1.twinx()
        ax2.plot(months_list, df['Üretim'], marker='s', label='Üretim', color='green')
        ax2.plot(months_list, df['Stok'], marker='d', label='Stok', color='red')
        ax2.plot(months_list, df['Karşılanmayan Talep'], marker='x', label='Karşılanmayan Talep', color='black')
        ax2.set_ylabel('Adet', color='gray')
        ax2.tick_params(axis='y', labelcolor='gray')
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        plt.title('Dinamik Programlama Tabanlı Model Sonuçları')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        st.pyplot(fig)

        # Ayrıntılı Toplam Maliyetler ve Birim Maliyet Analizi
        detay = m4_ayrintili(total_labor, total_production, total_holding, total_stockout, total_hiring, total_firing)
        st.subheader("Ayrıntılı Toplam Maliyetler")
        st.markdown(f"- İşçilik Maliyeti Toplamı: {detay['total_labor']:,.2f} TL")
        st.markdown(f"- Üretim Maliyeti Toplamı: {detay['total_production']:,.2f} TL")
        st.markdown(f"- Stok Maliyeti Toplamı: {detay['total_holding']:,.2f} TL")
        st.markdown(f"- Stoksuzluk Maliyeti Toplamı: {detay['total_stockout']:,.2f} TL")
        st.markdown(f"- İşe Alım Maliyeti Toplamı: {detay['total_hiring']:,.2f} TL")
        st.markdown(f"- İşten Çıkarma Maliyeti Toplamı: {detay['total_firing']:,.2f} TL")
        birim = m4_birim(total_demand, total_produced, total_unfilled, min_cost, total_labor, total_production, total_holding, total_hiring, total_firing)
        st.subheader("Birim Maliyet Analizi")
        st.markdown(f"- Toplam Talep: {birim['total_demand']:,} birim")
        st.markdown(f"- Toplam Üretim: {birim['total_produced']:,} birim ({(birim['total_produced']/birim['total_demand']*100 if birim['total_demand'] else 0):.2f}%)")
        st.markdown(f"- Karşılanmayan Talep: {birim['total_unfilled']:,} birim ({(birim['total_unfilled']/birim['total_demand']*100 if birim['total_demand'] else 0):.2f}%)")
        if birim['total_produced'] > 0:
            st.markdown(f"- Ortalama Birim Maliyet (Toplam): {birim['avg_unit_cost']:.2f} TL/birim")
            st.markdown(f"- İşçilik Birim Maliyeti: {birim['avg_labor_unit']:.2f} TL/birim")
            st.markdown(f"- Üretim Birim Maliyeti: {birim['avg_prod_unit']:.2f} TL/birim")
            st.markdown(f"- Diğer Birim Maliyetler (Stok+Alım+Çıkış): {birim['avg_other_unit']:.2f} TL/birim")
        else:
            st.markdown("- Ortalama Birim Maliyet: Hesaplanamadı (0 birim üretildi)")

if model == "Dış Kaynak Karşılaştırma (Model 5)":
    st.header("Dış Kaynak Kullanımı Karşılaştırma (Model 5)")
    with st.sidebar:
        demand = st.text_input("Aylık Talep (virgülle ayrılmış)", ", ".join(map(str, model5.demand)), key="m5_demand")
        demand = [int(x.strip()) for x in demand.split(",") if x.strip()]
        working_days = st.text_input("Aylık Çalışma Günü (virgülle ayrılmış)", ", ".join(map(str, model5.working_days)), key="m5_days")
        working_days = [int(x.strip()) for x in working_days.split(",") if x.strip()]
        if len(demand) != len(working_days):
            st.error(f"Talep ve çalışma günü uzunlukları eşit olmalı. Şu an talep: {len(demand)}, çalışma günü: {len(working_days)}.")
            st.stop()
        holding_cost = st.number_input("Stok Maliyeti (TL)", 1, 100, int(model5.holding_cost), key="m5_holding")
        cost_supplier_A = st.number_input("Tedarikçi A Maliyeti (TL)", 1, 1000, int(model5.cost_supplier_A), key="m5_cost_A")
        cost_supplier_B = st.number_input("Tedarikçi B Maliyeti (TL)", 1, 1000, int(model5.cost_supplier_B), key="m5_cost_B")
        capacity_supplier_A = st.number_input("Tedarikçi A Kapasitesi (adet)", 1, 10000, int(model5.capacity_supplier_A), key="m5_cap_A")
        capacity_supplier_B = st.number_input("Tedarikçi B Kapasitesi (adet)", 1, 10000, int(model5.capacity_supplier_B), key="m5_cap_B")
        stockout_cost = st.number_input("Stoksuzluk Maliyeti (TL/adet)", 1, 9999, int(model5.stockout_cost), key="m5_stockout")
        run_model = st.button("Modeli Çalıştır", key="m5_run")
    if run_model:
        df, toplam_maliyet = model5_run(
            demand, holding_cost, cost_supplier_A, cost_supplier_B, capacity_supplier_A, capacity_supplier_B, working_days, stockout_cost
        )
        st.subheader("Sonuç Tablosu")
        st.dataframe(df, use_container_width=True)
        st.success(f"Toplam Maliyet: {toplam_maliyet:,.2f} TL")

        # Grafiksel Sonuçlar
        st.subheader("Grafiksel Sonuçlar")
        months_list = [t+1 for t in range(len(demand))]
        fig, ax = plt.subplots(figsize=(12,6))
        ax.bar(months_list, df['Tedarikçi A'], color='#3498db', label='Tedarikçi A (Düşük Maliyet)', alpha=0.7)
        ax.bar(months_list, df['Tedarikçi B'], bottom=df['Tedarikçi A'], color='#e74c3c', label='Tedarikçi B (Yüksek Maliyet)', alpha=0.7)
        ax.plot(months_list, demand, marker='o', label='Talep', color='#2c3e50', linewidth=2, linestyle='--')
        ax.plot(months_list, df['Stok'], marker='d', label='Stok', color='#27ae60', linewidth=2)
        ax.plot(months_list, df['Karşılanmayan Talep'], marker='x', label='Karşılanmayan Talep', color='#8e44ad', linewidth=2)
        ax.axhline(y=capacity_supplier_A, color='#f39c12', linestyle='-.', label=f'Tedarikçi A Kapasite Limiti ({capacity_supplier_A})')
        ax.set_xlabel('Ay')
        ax.set_ylabel('Adet')
        ax.set_title('Dış Kaynak Kullanımı Karşılaştırması')
        ax.legend(loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        st.pyplot(fig)
        # Ayrıntılı Toplam Maliyetler
        total_cost_A = df['Tedarikçi A'].sum() * cost_supplier_A
        total_cost_B = df['Tedarikçi B'].sum() * cost_supplier_B
        total_holding = df['Stok'].sum() * holding_cost
        total_stockout = df['Karşılanmayan Talep'].sum() * stockout_cost
        detay = m5_ayrintili(total_cost_A, total_cost_B, total_holding, total_stockout)
        st.subheader("Ayrıntılı Toplam Maliyetler")
        st.markdown(f"- Tedarikçi A Toplam Maliyet: {detay['total_cost_A']:,.2f} TL")
        st.markdown(f"- Tedarikçi B Toplam Maliyet: {detay['total_cost_B']:,.2f} TL")
        st.markdown(f"- Stok Tutma Toplam Maliyet: {detay['total_holding']:,.2f} TL")
        st.markdown(f"- Karşılanmayan Talep Toplam Maliyet: {detay['total_stockout']:,.2f} TL")
        # Birim Maliyet Analizi
        total_demand = sum(demand)
        total_fulfilled = df['Tedarikçi A'].sum() + df['Tedarikçi B'].sum()
        birim = m5_birim(sum(demand), df['Tedarikçi A'].sum() + df['Tedarikçi B'].sum(), toplam_maliyet, cost_supplier_A, cost_supplier_B)
        st.subheader("Birim Maliyet Analizi")
        st.markdown(f"- Toplam Talep: {birim['total_demand']:,} birim")
        st.markdown(f"- Karşılanan Talep: {birim['total_fulfilled']:,} birim ({(birim['total_fulfilled']/birim['total_demand']*100 if birim['total_demand'] else 0):.2f}%)")
        if birim['total_fulfilled'] > 0:
            st.markdown(f"- Ortalama Birim Maliyet: {birim['avg_unit_cost']:.2f} TL/birim")
        else:
            st.markdown("- Ortalama Birim Maliyet: Hesaplanamadı (0 birim karşılandı)")
        st.markdown(f"- Tedarikçi A Birim Maliyeti: {birim['cost_supplier_A']:.2f} TL/birim")
        st.markdown(f"- Tedarikçi B Birim Maliyeti: {birim['cost_supplier_B']:.2f} TL/birim")

if model == "Mevsimsellik ve Dalga (Model 6)":
    st.header("Mevsimsel Talep Dalgaları ve Stok Optimizasyonu (Model 6)")
    with st.sidebar:
        demand = st.text_input("Aylık Talep (virgülle ayrılmış)", ", ".join(map(str, model6.demand)), key="m6_demand")
        demand = [int(x.strip()) for x in demand.split(",") if x.strip()]
        working_days = st.text_input("Aylık Çalışma Günü (virgülle ayrılmış)", ", ".join(map(str, model6.working_days)), key="m6_days")
        working_days = [int(x.strip()) for x in working_days.split(",") if x.strip()]
        if len(demand) != 12 or len(working_days) != 12:
            st.error(f"Talep tahminleri ve çalışma takvimi 12 ay olmalı. Şu an talep: {len(demand)}, çalışma takvimi: {len(working_days)}.")
            st.stop()
        holding_cost = st.number_input("Stok Maliyeti (TL)", 1, 100, int(model6.holding_cost), key="m6_holding")
        stockout_cost = st.number_input("Stoksuzluk Maliyeti (TL/adet)", 1, 100, int(model6.stockout_cost), key="m6_stockout")
        production_cost = st.number_input("Üretim Maliyeti (TL)", 1, 100, int(model6.production_cost), key="m6_prod_cost")
        labor_per_unit = st.number_input("Birim İşgücü (saat)", 0.1, 10.0, float(model6.labor_per_unit), key="m6_labor_per_unit")
        hourly_wage = st.number_input("Saatlik Ücret (TL)", 1, 100, int(model6.hourly_wage), key="m6_hourly_wage")
        daily_hours = st.number_input("Günlük Çalışma Saati", 1, 24, 8, key="m6_daily_hours")
        run_model = st.button("Modeli Çalıştır", key="m6_run")
    if run_model:
        df, total_cost, needed_workers, max_production = model6_run(
            demand, holding_cost, stockout_cost, production_cost, labor_per_unit, hourly_wage, daily_hours
        )
        st.info(f"Maksimum Üretim Kapasitesi: {max_production} adet/ay | Gerekli Optimum İşçi Sayısı: {needed_workers}")
        st.subheader("Sonuç Tablosu")
        st.dataframe(df, use_container_width=True)
        st.success(f"Toplam Maliyet: {total_cost:,.2f} TL")
        st.subheader("Grafiksel Sonuçlar")
        months_list = list(range(1, len(demand)+1))
        fig, ax = plt.subplots(figsize=(12,6))
        ax.plot(months_list, df['Talep'], marker='o', label='Talep', color='orange')
        ax.bar(months_list, df['Üretim'], color='skyblue', label='Üretim', alpha=0.7)
        ax.plot(months_list, df['Stok'], marker='d', label='Stok', color='red')
        ax.plot(months_list, df['Stoksuzluk'], marker='x', label='Stoksuzluk', color='black')
        ax.set_xlabel('Ay')
        ax.set_ylabel('Adet')
        ax.set_title('Mevsimsellik ve Stok Optimizasyonu Sonuçları')
        ax.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        st.pyplot(fig)
        # Ayrıntılı Toplam Maliyetler
        total_holding = df['Stok Maliyeti'].sum()
        total_stockout = df['Stoksuzluk Maliyeti'].sum()
        total_production_cost = df['Üretim Maliyeti'].sum()
        total_labor_cost = df['İşçilik Maliyeti'].sum()
        total_hiring_cost = model6.hiring_cost * needed_workers
        detay = m6_ayrintili(total_holding, total_stockout, total_production_cost, total_labor_cost, total_hiring_cost)
        st.subheader("Ayrıntılı Toplam Maliyetler")
        st.markdown(f"- Stok Maliyeti Toplamı: {detay['total_holding']:,.2f} TL")
        st.markdown(f"- Stoksuzluk Maliyeti Toplamı: {detay['total_stockout']:,.2f} TL")
        st.markdown(f"- Üretim Maliyeti Toplamı: {detay['total_production_cost']:,.2f} TL")
        st.markdown(f"- İşçilik Maliyeti Toplamı: {detay['total_labor_cost']:,.2f} TL")
        st.markdown(f"- İşe Alım Maliyeti Toplamı: {detay['total_hiring_cost']:,.2f} TL")
        # Birim Maliyet Analizi
        total_demand = df['Talep'].sum()
        total_produced = df['Üretim'].sum()
        total_unfilled = df['Stoksuzluk'].sum()
        birim = m6_birim(total_demand, total_produced, total_unfilled, total_cost, total_labor_cost, total_production_cost, total_holding, total_stockout)
        st.subheader("Birim Maliyet Analizi")
        st.markdown(f"- Toplam Talep: {birim['total_demand']:,} birim")
        st.markdown(f"- Toplam Üretim: {birim['total_produced']:,} birim ({(birim['total_produced']/birim['total_demand']*100 if birim['total_demand'] else 0):.2f}%)")
        st.markdown(f"- Karşılanmayan Talep: {birim['total_unfilled']:,} birim ({(birim['total_unfilled']/birim['total_demand']*100 if birim['total_demand'] else 0):.2f}%)")
        if birim['total_produced'] > 0:
            st.markdown(f"- Ortalama Birim Maliyet (Toplam): {birim['avg_unit_cost']:.2f} TL/birim")
            st.markdown(f"- İşçilik Birim Maliyeti: {birim['avg_labor_unit']:.2f} TL/birim")
            st.markdown(f"- Üretim Birim Maliyeti: {birim['avg_prod_unit']:.2f} TL/birim")
            st.markdown(f"- Diğer Birim Maliyetler (Stok, Stoksuzluk): {birim['avg_other_unit']:.2f} TL/birim")
        else:
            st.markdown("- Ortalama Birim Maliyet: Hesaplanamadı (0 birim üretildi)")

if model == "Karşılaştırma Tablosu":
    st.header("Tüm Modeller İçin Karşılaştırma Tablosu")
    model_names = [
        ("Model 1", "Karma Planlama (Model 1)", model1.maliyet_analizi, "Yüksek", "Karma planlama, işgücü ve fason esnekliği"),
        ("Model 2", "Fazla Mesaili Üretim (Model 2)", model2.maliyet_analizi, "Orta", "Fazla mesai ile esneklik"),
        ("Model 3", "Toplu Üretim ve Stoklama (Model 3)", model3.maliyet_analizi, "Düşük", "Sabit işgücü, toplu üretim"),
        ("Model 4", "Dinamik Programlama (Model 4)", model4.maliyet_analizi, "Düşük", "Sabit işgücü, belirli üretim"),
        ("Model 5", "Dış Kaynak Karşılaştırma (Model 5)", model5.maliyet_analizi, "Yok", "Tam fason kullanımı"),
        ("Model 6", "Mevsimsellik ve Dalga (Model 6)", model6.maliyet_analizi, "Orta", "Mevsimsellik ve stok optimizasyonu"),
    ]
    summary_rows = []
    for idx, (short_name, display_name, func, flex, scenario) in enumerate(model_names, 1):
        try:
            res = func()
            cost = res.get("Toplam Maliyet", None)
            labor_cost = res.get("İşçilik Maliyeti", None)
            total_prod = res.get("Toplam Üretim", None)
            total_demand = res.get("Toplam Talep", None)
            stockout = res.get("Karşılanmayan Talep", 0)
            stockout_rate = (stockout / total_demand * 100) if (total_demand and total_demand > 0) else 0
            summary_rows.append([
                cost, labor_cost, total_prod, stockout_rate, flex, scenario
            ])
        except Exception as e:
            summary_rows.append([None, None, None, None, flex, f"Hata: {str(e)}"])
    summary_df = pd.DataFrame(
        summary_rows,
        columns=["Toplam Maliyet (₺)", "Toplam İşçilik Maliyeti (₺)", "Toplam Üretim", "Stoksuzluk Oranı (%)", "İşgücü Esnekliği", "Uygun Senaryo"],
        index=[m[0] for m in model_names]
    )
    # Sayısal sütunları okunaklı formatla (sıralama bozulmasın diye display_format ile)
    def format_number(val):
        if pd.isnull(val):
            return ""
        if isinstance(val, (int, float)):
            return f"{val:,.0f}".replace(",", ".")
        return val
    st.subheader("Özet Karşılaştırma Tablosu")
    st.dataframe(
        summary_df,
        use_container_width=True
    )
    st.markdown("---")
    # Detaylı tabloyu da göster
    st.subheader("Detaylı Karşılaştırma Tablosu")
    results = []
    for name, display_name, func, _, _ in model_names:
        try:
            res = func()
            res["Model"] = display_name
            results.append(res)
        except Exception as e:
            results.append({"Model": display_name, "Hata": str(e)})
    df = pd.DataFrame(results)

    cols = ["Model"] + [c for c in df.columns if c != "Model"]
    st.dataframe(df[cols], use_container_width=True)
