import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pulp

st.set_page_config(page_title="Üretim Planlama Modelleri", layout="wide")
st.title("Üretim Planlama Modelleri Karar Destek Arayüzü")

model = st.sidebar.selectbox("Model Seçiniz", [
    "Karma Planlama Modeli (Model 1)",
    "Fazla Mesaili Üretim (Model 2)",
    "Toplu Üretim ve Stoklama (Model 3)",
    "Dinamik Programlama (Model 4)",
    "Dış Kaynak Karşılaştırma (Model 5)",
    "Mevsimsellik ve Dalga (Model 6)"
])

st.sidebar.markdown("---")

def model1_run(demand, working_days, holding_cost, outsourcing_cost, labor_per_unit, hiring_cost, firing_cost, daily_hours, outsourcing_capacity, min_internal_ratio, max_workforce_change, max_outsourcing_ratio):
    T = len(demand)
    decision_model = pulp.LpProblem('Karma_Planlama_Modeli', pulp.LpMinimize)
    workers = [pulp.LpVariable(f'workers_{t}', lowBound=0, cat='Integer') for t in range(T)]
    internal_production = [pulp.LpVariable(f'internal_prod_{t}', lowBound=0, cat='Integer') for t in range(T)]
    outsourced_production = [pulp.LpVariable(f'outsourced_prod_{t}', lowBound=0, cat='Integer') for t in range(T)]
    inventory = [pulp.LpVariable(f'inventory_{t}', lowBound=0, cat='Integer') for t in range(T)]
    hired = [pulp.LpVariable(f'hired_{t}', lowBound=0, cat='Integer') for t in range(T)]
    fired = [pulp.LpVariable(f'fired_{t}', lowBound=0, cat='Integer') for t in range(T)]
    cost = (
        pulp.lpSum([
            holding_cost * inventory[t] +
            outsourcing_cost * outsourced_production[t] +
            hiring_cost * hired[t] +
            firing_cost * fired[t]
            for t in range(T)
        ])
    )
    decision_model += cost
    for t in range(T):
        if t == 0:
            prev_inventory = 0
        else:
            prev_inventory = inventory[t-1]
        decision_model += (internal_production[t] + outsourced_production[t] + prev_inventory - demand[t] == inventory[t])
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
        decision_model += (workers[t] <= internal_production[t] * labor_per_unit / (working_days[t] * daily_hours) + 1)
        decision_model += (workers[t] >= 0)
    solver = pulp.PULP_CBC_CMD(msg=0)
    decision_model.solve(solver)
    results = []
    for t in range(T):
        results.append([
            t+1,
            int(workers[t].varValue),
            int(internal_production[t].varValue),
            int(outsourced_production[t].varValue),
            int(inventory[t].varValue),
            int(hired[t].varValue),
            int(fired[t].varValue)
        ])
    df = pd.DataFrame(results, columns=["Ay", "İşçi", "İç Üretim", "Fason", "Stok", "Alım", "Çıkış"])
    toplam_maliyet = pulp.value(decision_model.objective)
    return df, toplam_maliyet

def model2_run(demand, working_days, holding_cost, labor_per_unit, fixed_workers, daily_hours, overtime_wage_multiplier, max_overtime_per_worker, overtime_cost_per_hour, stockout_cost, normal_hourly_wage):
    months = len(demand)
    production = np.zeros(months)
    overtime_hours = np.zeros(months)
    inventory = np.zeros(months)
    prev_inventory = 0
    results = []
    total_cost = 0
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
        overtime = max(ot_hours, 0) * overtime_wage_multiplier * overtime_cost_per_hour
        normal_labor_cost = fixed_workers * working_days[t] * daily_hours * normal_hourly_wage
        total_cost += holding + stockout + overtime + normal_labor_cost
        results.append([
            t+1, fixed_workers, prod, ot_hours, inventory[t], holding, stockout, overtime, normal_labor_cost
        ])
        prev_inventory = inventory[t]
    headers = [
        'Ay', 'İşçi', 'Üretim', 'Fazla Mesai', 'Stok', 'Stok Maliyeti', 'Stoksuzluk Maliyeti', 'Fazla Mesai Maliyeti', 'Normal İşçilik Maliyeti'
    ]
    df = pd.DataFrame(results, columns=headers)
    return df, total_cost

if model == "Karma Planlama Modeli (Model 1)":
    st.header("Karma Planlama Modeli (Model 1)")
    demand = st.text_input("Aylık Talep (virgülle ayrılmış)", "10000, 7200, 7500, 2500, 2000, 1000")
    demand = [int(x.strip()) for x in demand.split(",") if x.strip()]
    working_days = st.text_input("Aylık Çalışma Günü (virgülle ayrılmış)", "22, 20, 23, 21, 22, 20")
    working_days = [int(x.strip()) for x in working_days.split(",") if x.strip()]
    holding_cost = st.number_input("Stok Maliyeti (TL)", 1, 100, 5)
    outsourcing_cost = st.number_input("Fason Maliyeti (TL)", 1, 100, 15)
    labor_per_unit = st.number_input("Birim İşgücü (saat)", 0.1, 10.0, 0.5)
    hiring_cost = st.number_input("İşçi Alım Maliyeti (TL)", 0, 5000, 1000)
    firing_cost = st.number_input("İşçi Çıkarma Maliyeti (TL)", 0, 5000, 800)
    daily_hours = st.number_input("Günlük Çalışma Saati", 1, 24, 8)
    outsourcing_capacity = st.number_input("Fason Kapasitesi (adet)", 1, 10000, 500)
    min_internal_ratio = st.slider("En Az İç Üretim Oranı (%)", 0, 100, 70) / 100
    max_workforce_change = st.number_input("İşgücü Değişim Sınırı (kişi)", 1, 100, 5)
    max_outsourcing_ratio = st.slider("En Fazla Fason Oranı (%)", 0, 100, 30) / 100
    if st.button("Modeli Çalıştır"):
        df, toplam_maliyet = model1_run(
            demand, working_days, holding_cost, outsourcing_cost, labor_per_unit, hiring_cost, firing_cost, daily_hours,
            outsourcing_capacity, min_internal_ratio, max_workforce_change, max_outsourcing_ratio
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
        ax2.set_ylabel('Adet', color='gray')
        ax2.tick_params(axis='y', labelcolor='gray')
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        plt.title('Karma Planlama Modeli Sonuçları')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        st.pyplot(fig)

if model == "Fazla Mesaili Üretim (Model 2)":
    st.header("Fazla Mesaili Üretim Modeli (Model 2)")
    demand = st.text_input("Aylık Talep (virgülle ayrılmış)", "2500, 2000, 2500, 3500, 3000, 2000")
    demand = np.array([int(x.strip()) for x in demand.split(",") if x.strip()])
    working_days = st.text_input("Aylık Çalışma Günü (virgülle ayrılmış)", "22, 20, 23, 21, 22, 20")
    working_days = np.array([int(x.strip()) for x in working_days.split(",") if x.strip()])
    holding_cost = st.number_input("Stok Maliyeti (TL)", 1, 100, 5, key="m2_holding")
    labor_per_unit = st.number_input("Birim İşgücü (saat)", 0.1, 10.0, 0.5, key="m2_labor")
    fixed_workers = st.number_input("Sabit İşçi Sayısı", 1, 200, 15, key="m2_workers")
    daily_hours = st.number_input("Günlük Çalışma Saati", 1, 24, 8, key="m2_daily")
    overtime_wage_multiplier = st.number_input("Fazla Mesai Ücret Çarpanı", 1.0, 5.0, 1.5)
    max_overtime_per_worker = st.number_input("Maks. Fazla Mesai (saat/işçi)", 0, 100, 20)
    overtime_cost_per_hour = st.number_input("Fazla Mesai Saatlik Maliyeti (TL)", 1, 1000, 10)
    stockout_cost = st.number_input("Stoksuzluk Maliyeti (TL/adet)", 1, 100, 20)
    normal_hourly_wage = st.number_input("Normal Saatlik İşçilik Maliyeti (TL)", 1, 1000, 10)
    if st.button("Modeli Çalıştır", key="m2_run"):
        df_table, total_cost = model2_run(
            demand, working_days, holding_cost, labor_per_unit, fixed_workers, daily_hours,
            overtime_wage_multiplier, max_overtime_per_worker, overtime_cost_per_hour, stockout_cost, normal_hourly_wage
        )
        st.subheader("Sonuç Tablosu")
        st.dataframe(df_table, use_container_width=True)
        st.success(f"Toplam Maliyet: {total_cost:,.2f} TL")
        st.subheader("Grafiksel Sonuçlar")
        fig, ax1 = plt.subplots(figsize=(10,6))
        months_list = df_table["Ay"].tolist()
        ax1.bar(months_list, df_table["İşçi"], color='skyblue', label='İşçi (Sabit)', alpha=0.7)
        ax1.set_xlabel('Ay')
        ax1.set_ylabel('İşçi Sayısı', color='skyblue')
        ax1.tick_params(axis='y', labelcolor='skyblue')
        ax2 = ax1.twinx()
        ax2.plot(months_list, df_table["Üretim"], marker='s', label='Üretim', color='green')
        ax2.plot(months_list, df_table["Fazla Mesai"], marker='^', label='Fazla Mesai (saat)', color='orange')
        ax2.plot(months_list, df_table["Stok"], marker='d', label='Stok', color='red')
        ax2.set_ylabel('Adet / Saat', color='gray')
        ax2.tick_params(axis='y', labelcolor='gray')
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        plt.title('Fazla Mesaili Üretim Modeli Sonuçları')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        st.pyplot(fig)

if model == "Toplu Üretim ve Stoklama (Model 3)":
    st.header("Toplu Üretim ve Stoklama Modeli (Model 3)")
    demand = st.text_input("Aylık Talep (virgülle ayrılmış)", "1000, 1200, 1500, 1300, 1100, 1400")
    demand = np.array([int(x.strip()) for x in demand.split(",") if x.strip()])
    working_days = st.text_input("Aylık Çalışma Günü (virgülle ayrılmış)", "22, 20, 23, 21, 22, 20")
    working_days = np.array([int(x.strip()) for x in working_days.split(",") if x.strip()])
    holding_cost = st.number_input("Stok Maliyeti (TL)", 1, 100, 5, key="m3_holding")
    fixed_workers = st.number_input("Sabit İşçi Sayısı", 1, 200, 10, key="m3_workers")
    production_rate = st.number_input("Üretim Hızı (adet/saat)", 0.1, 10.0, 1.0)
    daily_hours = st.number_input("Günlük Çalışma Saati", 1, 24, 8, key="m3_daily")
    months = len(demand)
    if st.button("Modeli Çalıştır", key="m3_run"):
        monthly_capacity = fixed_workers * daily_hours * working_days * production_rate
        production = monthly_capacity
        inventory = np.zeros(months)
        cost = 0
        prev_inventory = 0
        results = []
        for t in range(months):
            inventory[t] = prev_inventory + production[t] - demand[t]
            holding = max(inventory[t], 0) * holding_cost
            cost += holding
            results.append([
                t+1, production[t], inventory[t], holding
            ])
            prev_inventory = inventory[t]
        df = pd.DataFrame(results, columns=["Ay", "Üretim", "Stok", "Stok Maliyeti"])
        st.subheader("Sonuç Tablosu")
        st.dataframe(df, use_container_width=True)
        st.success(f"Toplam Maliyet: {cost:,.2f} TL")
        st.subheader("Grafiksel Sonuçlar")
        fig, ax = plt.subplots(figsize=(10,6))
        months_list = df["Ay"].tolist()
        ax.bar(months_list, df["Üretim"], color='skyblue', label='Üretim', alpha=0.7)
        ax.plot(months_list, df["Stok"], marker='d', label='Stok', color='red')
        ax.set_xlabel('Ay')
        ax.set_ylabel('Adet')
        ax.legend()
        plt.title('Toplu Üretim ve Stoklama Modeli Sonuçları')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        st.pyplot(fig)

if model == "Dinamik Programlama (Model 4)":
    st.header("Dinamik Programlama Tabanlı Model (Model 4)")
    demand = st.text_input("Aylık Talep (virgülle ayrılmış)", "3000, 3200, 8500, 5300, 1100, 1400")
    demand = np.array([int(x.strip()) for x in demand.split(",") if x.strip()])
    working_days = st.text_input("Aylık Çalışma Günü (virgülle ayrılmış)", "22, 20, 23, 21, 22, 20", key="m4_days")
    working_days = np.array([int(x.strip()) for x in working_days.split(",") if x.strip()])
    holding_cost = st.number_input("Stok Maliyeti (TL)", 1, 100, 5, key="m4_holding")
    hiring_cost = st.number_input("İşçi Alım Maliyeti (TL)", 0, 5000, 1000, key="m4_hire")
    firing_cost = st.number_input("İşçi Çıkarma Maliyeti (TL)", 0, 5000, 800, key="m4_fire")
    daily_hours = st.number_input("Günlük Çalışma Saati", 1, 24, 8, key="m4_daily")
    labor_per_unit = st.number_input("Birim İşgücü (saat)", 0.1, 10.0, 0.5, key="m4_labor")
    min_workers = st.number_input("Minimum İşçi", 1, 200, 30)
    max_workers = st.number_input("Maksimum İşçi", 1, 200, 120)
    max_workforce_change = st.number_input("Aylık İşgücü Değişim Sınırı", 1, 100, 20)
    months = len(demand)
    if st.button("Modeli Çalıştır", key="m4_run"):
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
                        if capacity < demand[t]:
                            continue
                        inventory = capacity - demand[t]
                        hire = max(0, w - prev_w) * hiring_cost
                        fire = max(0, prev_w - w) * firing_cost
                        holding = inventory * holding_cost
                        total_cost = cost_table[t, prev_w] + hire + fire + holding
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
        for t, w in enumerate(workers_seq):
            cap = prod_capacity(w, t)
            production_seq.append(min(cap, demand[t]))
            inventory_seq.append(cap - demand[t])
        results = []
        for t in range(months):
            results.append([
                t+1, workers_seq[t], production_seq[t], inventory_seq[t]
            ])
        df = pd.DataFrame(results, columns=['Ay', 'İşçi', 'Üretim', 'Stok'])
        st.subheader("Sonuç Tablosu")
        st.dataframe(df, use_container_width=True)
        st.success(f"Toplam Maliyet: {min_cost:,.2f} TL")
        st.subheader("Grafiksel Sonuçlar")
        months_list = list(range(1, months+1))
        fig, ax1 = plt.subplots(figsize=(10,6))
        ax1.bar(months_list, workers_seq, color='skyblue', label='İşçi', alpha=0.7)
        ax1.set_xlabel('Ay')
        ax1.set_ylabel('İşçi Sayısı', color='skyblue')
        ax1.tick_params(axis='y', labelcolor='skyblue')
        ax2 = ax1.twinx()
        ax2.plot(months_list, production_seq, marker='s', label='Üretim', color='green')
        ax2.plot(months_list, inventory_seq, marker='d', label='Stok', color='red')
        ax2.set_ylabel('Adet', color='gray')
        ax2.tick_params(axis='y', labelcolor='gray')
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        plt.title('Dinamik Programlama Tabanlı Model Sonuçları')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        st.pyplot(fig)

if model == "Dış Kaynak Karşılaştırma (Model 5)":
    st.header("Dış Kaynak Kullanımı Karşılaştırma Modeli (Model 5)")
    demand = st.text_input("Aylık Talep (virgülle ayrılmış)", "2000, 4200, 3500, 1300, 1100, 1400")
    demand = [int(x.strip()) for x in demand.split(",") if x.strip()]
    holding_cost = st.number_input("Stok Maliyeti (TL)", 1, 100, 5, key="m5_holding")
    internal_production_cost = st.number_input("İç Üretim Maliyeti (TL)", 1, 100, 10)
    cost_supplier_A = st.number_input("Tedarikçi A Maliyeti (TL)", 1, 100, 15)
    cost_supplier_B = st.number_input("Tedarikçi B Maliyeti (TL)", 1, 100, 18)
    capacity_supplier_A = st.number_input("Tedarikçi A Kapasitesi (adet)", 1, 10000, 500)
    capacity_supplier_B = st.number_input("Tedarikçi B Kapasitesi (adet)", 1, 10000, 500)
    max_internal_production = st.number_input("İç Üretim Kapasitesi (adet)", 1, 20000, 1500)
    stockout_cost = st.number_input("Stoksuzluk Maliyeti (TL/adet)", 1, 100, 20, key="m5_stockout")
    months = len(demand)
    if st.button("Modeli Çalıştır", key="m5_run"):
        decision_model = pulp.LpProblem('Dis_Kaynak_Karsilastirma', pulp.LpMinimize)
        internal_production = [pulp.LpVariable(f'internal_prod_{t}', lowBound=0, cat='Integer') for t in range(months)]
        out_A = [pulp.LpVariable(f'out_A_{t}', lowBound=0, cat='Integer') for t in range(months)]
        out_B = [pulp.LpVariable(f'out_B_{t}', lowBound=0, cat='Integer') for t in range(months)]
        inventory = [pulp.LpVariable(f'inventory_{t}', lowBound=0, cat='Integer') for t in range(months)]
        stockout = [pulp.LpVariable(f'stockout_{t}', lowBound=0, cat='Integer') for t in range(months)]
        decision_model += pulp.lpSum([
            internal_production_cost * internal_production[t] +
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
            decision_model += (internal_production[t] + out_A[t] + out_B[t] + prev_inventory + stockout[t] == demand[t] + inventory[t])
            decision_model += (internal_production[t] <= max_internal_production)
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
                int(internal_production[t].varValue),
                int(out_A[t].varValue),
                int(out_B[t].varValue),
                int(inventory[t].varValue),
                int(stockout[t].varValue)
            ])
        df = pd.DataFrame(table, columns=['Ay', 'İç Üretim', 'Tedarikçi A', 'Tedarikçi B', 'Stok', 'Karşılanmayan Talep'])
        toplam_maliyet = pulp.value(decision_model.objective)
        st.subheader("Sonuç Tablosu")
        st.dataframe(df, use_container_width=True)
        st.success(f"Toplam Maliyet: {toplam_maliyet:,.2f} TL")
        st.subheader("Grafiksel Sonuçlar")
        months_list = [t+1 for t in range(months)]
        fig, ax = plt.subplots(figsize=(10,6))
        ax.bar(months_list, df['İç Üretim'], color='skyblue', label='İç Üretim', alpha=0.7)
        ax.bar(months_list, df['Tedarikçi A'], bottom=df['İç Üretim'], color='orange', label='Tedarikçi A', alpha=0.7)
        bottom_sum = df['İç Üretim'] + df['Tedarikçi A']
        ax.bar(months_list, df['Tedarikçi B'], bottom=bottom_sum, color='green', label='Tedarikçi B', alpha=0.7)
        ax.plot(months_list, df['Stok'], marker='d', label='Stok', color='red', linewidth=2)
        ax.plot(months_list, df['Karşılanmayan Talep'], marker='x', label='Karşılanmayan Talep', color='black', linewidth=2)
        ax.set_xlabel('Ay')
        ax.set_ylabel('Adet')
        ax.set_title('Dış Kaynak Kullanımı Karşılaştırması')
        ax.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        st.pyplot(fig)

if model == "Mevsimsellik ve Dalga (Model 6)":
    st.header("Mevsimsellik ve Talep Dalgaları Modeli (Model 6)")
    seasonal_demand = st.text_input("Aylık Talep (virgülle ayrılmış)", "1000, 1200, 1600, 2000, 2500, 3000, 2800, 2200, 1800, 1400, 1100, 900")
    seasonal_demand = np.array([int(x.strip()) for x in seasonal_demand.split(",") if x.strip()])
    holding_cost = st.number_input("Stok Maliyeti (TL)", 1, 100, 5, key="m6_holding")
    production_cost = st.number_input("Üretim Maliyeti (TL)", 1, 100, 12)
    max_production = st.number_input("Maksimum Üretim Kapasitesi (adet)", 1, 10000, 2200)
    months = len(seasonal_demand)
    if st.button("Modeli Çalıştır", key="m6_run"):
        results = []
        production = np.zeros(months)
        inventory = np.zeros(months)
        cost = 0
        prev_inventory = 0
        for t in range(months):
            if seasonal_demand[t] > max_production:
                production[t] = max_production
            else:
                production[t] = seasonal_demand[t]
            inventory[t] = prev_inventory + production[t] - seasonal_demand[t]
            holding = max(inventory[t], 0) * holding_cost
            prod_cost = production[t] * production_cost
            cost += holding + prod_cost
            results.append([
                t+1, seasonal_demand[t], production[t], inventory[t], holding, prod_cost
            ])
            prev_inventory = inventory[t]
        df = pd.DataFrame(results, columns=[
            'Ay', 'Talep', 'Üretim', 'Stok', 'Stok Maliyeti', 'Üretim Maliyeti'
        ])
        st.subheader("Sonuç Tablosu")
        st.dataframe(df, use_container_width=True)
        st.success(f"Toplam Maliyet: {cost:,.2f} TL")
        st.subheader("Grafiksel Sonuçlar")
        months_list = list(range(1, months+1))
        fig, ax = plt.subplots(figsize=(12,6))
        ax.plot(months_list, df['Talep'], marker='o', label='Talep', color='orange')
        ax.bar(months_list, df['Üretim'], color='skyblue', label='Üretim', alpha=0.7)
        ax.plot(months_list, df['Stok'], marker='d', label='Stok', color='red')
        ax.set_xlabel('Ay')
        ax.set_ylabel('Adet')
        ax.set_title('Mevsimsellik ve Talep Dalgaları Modeli Sonuçları')
        ax.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        st.pyplot(fig)

