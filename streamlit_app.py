import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yaml
import inspect

# Import default parameters from model files
import model1_mixed_planning as model1
import model2_overtime_production as model2
import model3_batch_production as model3
import model4_dynamic_programming as model4
import model5_outsourcing_comparison as model5
import model6_seasonal_planning as model6

# Import ayrintili_toplam_maliyetler and birim_maliyet_analizi from each model
from model1_mixed_planning import ayrintili_toplam_maliyetler as m1_ayrintili, birim_maliyet_analizi as m1_birim, solve_model as model1_solver
from model2_overtime_production import ayrintili_toplam_maliyetler as m2_ayrintili, birim_maliyet_analizi as m2_birim, solve_model as model2_solver
from model3_batch_production import ayrintili_toplam_maliyetler as m3_ayrintili, birim_maliyet_analizi as m3_birim, solve_model as model3_solver
from model4_dynamic_programming import ayrintili_toplam_maliyetler as m4_ayrintili, birim_maliyet_analizi as m4_birim, solve_model as model4_solver
from model5_outsourcing_comparison import ayrintili_toplam_maliyetler as m5_ayrintili, birim_maliyet_analizi as m5_birim, solve_model as model5_solver
from model6_seasonal_planning import ayrintili_toplam_maliyetler as m6_ayrintili, birim_maliyet_analizi as m6_birim, solve_model as model6_solver

st.set_page_config(page_title="Üretim Planlama Modelleri", layout="wide", initial_sidebar_state="expanded")
st.title("Üretim Planlama Modelleri Karar Destek Arayüzü")

# Load parameters from YAML
with open("parametreler.yaml", "r") as f:
    params = yaml.safe_load(f)

model = st.sidebar.selectbox("Model Seçiniz", params['models'])

st.sidebar.markdown("---")

def get_param(key, subkey=None, default=None):
    try:
        if subkey:
            return params[key][subkey]
        return params[key]
    except Exception:
        return default

def model1_run(demand, working_days, holding_cost, outsourcing_cost, labor_per_unit, hiring_cost, firing_cost, daily_hours, outsourcing_capacity, min_internal_ratio, max_workforce_change, max_outsourcing_ratio, stockout_cost, hourly_wage, production_cost, overtime_wage_multiplier, max_overtime_per_worker):
    # Use the shared model solver function from model1_mixed_planning
    model_vars = model1_solver(
        demand, working_days, holding_cost, outsourcing_cost, labor_per_unit,
        hiring_cost, firing_cost, daily_hours, outsourcing_capacity, min_internal_ratio,
        max_workforce_change, max_outsourcing_ratio, stockout_cost, hourly_wage,
        production_cost, overtime_wage_multiplier, max_overtime_per_worker
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

    # Create results dataframe
    T = len(demand)
    results = []
    for t in range(T):
        internal_prod_cost = int(internal_production[t].varValue) * production_cost
        outsourcing_cost_val = int(outsourced_production[t].varValue) * outsourcing_cost
        labor_cost = int(workers[t].varValue) * working_days[t] * daily_hours * hourly_wage
        overtime_cost = int(overtime_hours[t].varValue) * hourly_wage * overtime_wage_multiplier
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
            labor_cost,
            overtime_cost
        ])
    df = pd.DataFrame(results, columns=["Ay", "İşçi", "İç Üretim", "Fason", "Stok", "Alım", "Çıkış", "Karşılanmayan Talep", "İç Üretim Maliyeti", "Fason Üretim Maliyeti", "İşçilik Maliyeti", "Fazla Mesai Maliyeti"])

    # Model değişkenlerini dictionary olarak döndür
    model_results = {
        'internal_production': internal_production,
        'outsourced_production': outsourced_production,
        'inventory': inventory,
        'hired': hired,
        'fired': fired,
        'stockout': stockout,
        'overtime_hours': overtime_hours
    }

    return df, toplam_maliyet, model_results

def model2_run(demand, working_days, holding_cost, labor_per_unit, workers, daily_hours, overtime_wage_multiplier, max_overtime_per_worker, stockout_cost, normal_hourly_wage, production_cost):
    # Use the shared model solver function
    model_results = model2_solver(
        demand, working_days, holding_cost, labor_per_unit, workers,
        daily_hours, overtime_wage_multiplier, max_overtime_per_worker,
        stockout_cost, normal_hourly_wage, production_cost
    )

    # Get the dataframe and total_cost from model_results
    df = model_results['df']
    total_cost = model_results['total_cost']

    return df, total_cost, model_results.get('optimal_workers', workers)

def model3_run(demand, working_days, holding_cost, stockout_cost, workers, labor_per_unit, daily_hours, production_cost, worker_monthly_cost=None):
    # Use the shared model solver function
    model_results = model3_solver(
        demand, working_days, holding_cost, stockout_cost,
        workers, labor_per_unit, daily_hours,
        production_cost, worker_monthly_cost
    )

    # Use the optimized worker count for all cost calculations
    optimized_workers = model_results.get('optimized_workers', workers)
    original_workers = model_results.get('original_workers', workers)
    optimal_workers = model_results.get('optimal_workers', optimized_workers)

    # Get the results from the model
    df = model_results['df']
    cost = model_results['total_cost']
    total_holding = model_results['total_holding']
    total_stockout = model_results['total_stockout']
    total_labor = model_results['total_labor']
    total_production_cost = model_results['total_production_cost']
    total_demand = sum(demand)
    total_produced = model_results['total_produced']
    total_unfilled = model_results['total_unfilled']

    # Add hiring cost to the total cost (use optimized_workers)
    hiring_cost = params['costs']['hiring_cost']
    total_hiring_cost = hiring_cost * optimized_workers
    adjusted_cost = cost + total_hiring_cost

    # Calculate unit costs
    if total_produced > 0:
        avg_unit_cost = adjusted_cost / total_produced
        avg_labor_unit = total_labor / total_produced
        avg_prod_unit = total_production_cost / total_produced
        avg_other_unit = (total_holding + total_stockout + total_hiring_cost) / total_produced
    else:
        avg_unit_cost = avg_labor_unit = avg_prod_unit = avg_other_unit = 0

    return df, adjusted_cost, total_holding, total_stockout, total_labor, total_production_cost, total_demand, total_produced, total_unfilled, avg_unit_cost, avg_labor_unit, avg_prod_unit, avg_other_unit, total_hiring_cost, optimized_workers, original_workers, optimal_workers

def model4_run(demand, working_days, holding_cost, hiring_cost, firing_cost, daily_hours, labor_per_unit, workers, max_workers, max_workforce_change, hourly_wage, stockout_cost, production_cost):
    # Use the shared model solver function
    model_results = model4_solver(
        demand, working_days, holding_cost, stockout_cost, hiring_cost, firing_cost,
        daily_hours, labor_per_unit, workers, max_workers, max_workforce_change,
        hourly_wage, production_cost
    )

    # Extract results from model_results
    df = model_results['df']
    min_cost = model_results['min_cost']
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
        avg_unit_cost = min_cost / total_produced
        avg_labor_unit = total_labor / total_produced
        avg_prod_unit = total_production / total_produced
        avg_other_unit = (total_holding + total_hiring + total_firing) / total_produced
    else:
        avg_unit_cost = avg_labor_unit = avg_prod_unit = avg_other_unit = 0

    return df, min_cost, total_labor, total_production, total_holding, total_stockout, total_hiring, total_firing, total_demand, total_produced, total_unfilled, avg_unit_cost, avg_labor_unit, avg_prod_unit, avg_other_unit

def model5_run(demand, holding_cost, cost_supplier_A, cost_supplier_B, capacity_supplier_A, capacity_supplier_B, working_days, stockout_cost):
    # Use the shared model solver function
    model_results = model5_solver(
        demand, working_days, holding_cost, stockout_cost,
        cost_supplier_A, cost_supplier_B,
        capacity_supplier_A, capacity_supplier_B
    )

    # Extract the dataframe and total cost from the results
    df = model_results['df'][['Ay', 'Tedarikçi A', 'Tedarikçi B', 'Stok', 'Karşılanmayan Talep']]
    toplam_maliyet = model_results['toplam_maliyet']

    return df, toplam_maliyet

def model6_run(demand, working_days, holding_cost, stockout_cost, production_cost, labor_per_unit, hourly_wage, daily_hours, hiring_cost, firing_cost, workers, max_workers, max_workforce_change):
    # Use the shared model solver function
    model_results = model6_solver(
        demand, holding_cost, stockout_cost, production_cost,
        labor_per_unit, hourly_wage, daily_hours, working_days,
        hiring_cost, firing_cost, workers, max_workers, max_workforce_change
    )

    # Extract the results from the model
    df = model_results['df']
    total_cost = model_results['total_cost']

    # Get the average number of workers from the results
    needed_workers = int(df['İşçi'].mean())

    # Get the maximum production capacity
    max_production = int(df['Üretim'].max())

    return df, total_cost, needed_workers, max_production

def select_demand_type_and_workers(model_key):
    """
    Creates demand type selection interface and returns demand values and worker count

    Args:
        model_key (str): Unique key prefix for Streamlit widgets (e.g., "m1", "cmp")

    Returns:
        tuple: (demand, workers, working_days, selected_demand_type)
    """
    # Demand type selection
    demand_type_names = {
        "normal": "Normal Talep",
        "high": "Yüksek Talep",
        "veryHigh": "Aşırı Yüksek Talep",
        "seasonal": "Mevsimsel Talep"
    }
    # Get the demand types from params
    demand_types = list(params['demand'].keys())

    # Create options list with Turkish display names
    display_options = [demand_type_names[key] for key in demand_types]

    # Show Turkish names in dropdown
    selected_display = st.selectbox("Talep Tipi Seçiniz", display_options, index=0, key=f"{model_key}_demand_type")

    # Convert back to internal key
    selected_demand_type = next(key for key, value in demand_type_names.items() if value == selected_display)

    default_demand = params['demand'][selected_demand_type]
    manual_demand = st.text_input("Aylık Talep (virgülle ayrılmış, opsiyonel)", ", ".join(map(str, default_demand)),
                                  key=f"{model_key}_manual_demand")
    if manual_demand.strip():
        try:
            demand = [int(x.strip()) for x in manual_demand.split(",") if x.strip()]
        except Exception:
            st.error("Talep formatı hatalı. Lütfen sayıları virgülle ayırınız.")
            st.stop()
    else:
        demand = default_demand

    # Define working days properly to fix the unresolved reference error
    working_days = get_param('workforce', 'working_days', [22] * 12)
    if len(demand) != len(working_days):
        st.error(f"Talep 12 aylık girilmeli. Şu an talep: {len(demand)} Ay")
        st.stop()

    # Calculate worker multiplier based on demand type
    worker_multiplier = 1
    if selected_demand_type == "high":
        worker_multiplier = 5
    elif selected_demand_type == "veryHigh":
        worker_multiplier = 10

    # Get base worker count from params
    base_workers = int(get_param('workforce', 'workers', 8))

    # Calculate adjusted worker count
    adjusted_workers = base_workers * worker_multiplier

    # Display worker count with adjusted default
    workers = st.number_input(
        "İşçi Sayısı",
        min_value=1,
        max_value=200,
        value=adjusted_workers,
        step=1,
        key=f"{model_key}_workers",
        help=f"Talep tipine göre önerilen işçi sayısı: {base_workers}×{worker_multiplier}={adjusted_workers}"
    )

    return demand, workers, working_days, selected_demand_type

if model == "Karma Planlama (Model 1)":
    st.header("Karma Planlama (Model 1)")
    with st.sidebar:
        demand, workers, working_days, selected_demand_type = select_demand_type_and_workers("m1")
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
        hourly_wage = st.number_input("Saatlik Ücret (TL)", min_value=1, max_value=1000, value=int(model1.hourly_wage), step=1, key="m1_hourly_wage")
        production_cost = st.number_input("Birim Üretim Maliyeti (TL)", min_value=1, max_value=1000, value=int(model1.production_cost), step=1, key="m1_production_cost")
        overtime_wage_multiplier = st.number_input("Fazla Mesai Ücret Çarpanı", min_value=1.0, max_value=5.0, value=float(model1.overtime_wage_multiplier), step=0.1, key="m1_overtime_multiplier")
        max_overtime_per_worker = st.number_input("Maks. Fazla Mesai (saat/işçi)", min_value=0, max_value=100, value=int(model1.max_overtime_per_worker), step=1, key="m1_max_overtime")
        run_model = st.button("Modeli Çalıştır", key="m1_run")
    if run_model:
        df, toplam_maliyet, model_vars = model1_run(
            demand, working_days, holding_cost, outsourcing_cost, labor_per_unit, hiring_cost, firing_cost, daily_hours,
            outsourcing_capacity, min_internal_ratio, max_workforce_change, max_outsourcing_ratio, stockout_cost,
            hourly_wage, production_cost, overtime_wage_multiplier=overtime_wage_multiplier, max_overtime_per_worker=max_overtime_per_worker
        )
        st.subheader("Sonuç Tablosu")
        st.dataframe(df, use_container_width=True, hide_index=True)
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
            len(demand), model_vars['internal_production'], model_vars['outsourced_production'],
            model_vars['inventory'], model_vars['hired'], model_vars['fired'],
            model_vars['stockout'], model_vars['overtime_hours'],
            working_days, daily_hours, hourly_wage, production_cost,
            outsourcing_cost, holding_cost, stockout_cost, hiring_cost,
            firing_cost, overtime_wage_multiplier
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
            len(demand), demand, model_vars['internal_production'], model_vars['outsourced_production'],
            model_vars['stockout'], detay['total_internal_labor'], detay['total_internal_prod'],
            detay['total_outsource'], detay['total_holding'], detay['total_hiring'],
            detay['total_firing'], toplam_maliyet
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
                st.markdown(f"- İşçilik Birim Maliyeti: {birim['internal_labor_unit_cost']:.2f} TL/birim")
                # Replace the separate production costs with the weighted average
                avg_prod_unit = (detay['total_internal_prod'] + detay['total_outsource']) / birim['total_produced']
                st.markdown(f"- Üretim Birim Maliyeti (Ağırlıklı Ortalama): {avg_prod_unit:.2f} TL/birim")
                # Add note about the weighted average
                st.info(f"Bu maliyet, iç üretim ({production_cost} TL/birim) ve fason üretimin ({outsourcing_cost} TL/birim) ağırlıklı ortalamasıdır.")
            st.markdown(f"- Diğer Maliyetler (Stok, İşe Alım/Çıkarma): {birim['other_unit_cost']:.2f} TL/birim")
        else:
            st.markdown("- Ortalama Birim Maliyet: Hesaplanamadı (0 birim üretildi)")

if model == "Fazla Mesaili Üretim (Model 2)":
    st.header("Fazla Mesaili Üretim (Model 2)")
    with st.sidebar:
        demand, workers_input, working_days, selected_demand_type = select_demand_type_and_workers("m2")

        # Optimal işçi sayısını hesapla
        from model2_overtime_production import calculate_optimal_workers
        optimal_workers = calculate_optimal_workers(
            demand, working_days,
            st.session_state.get("m2_daily", model2.daily_hours),  # Fixed: Use model2.daily_hours instead of daily_hours
            st.session_state.get("m2_labor", model2.labor_per_unit),
            st.session_state.get("m2_max_overtime", model2.max_overtime_per_worker)
        )

        # İşçi sayısını ekranda gösterirken optimum değeri de belirt
        st.info(f"Talep ve parametrelere göre optimal işçi sayısı: {optimal_workers}")

        # Kullanıcının girdiği değer mi yoksa optimal değer mi kullanılacak?
        use_optimal = st.checkbox("Optimal işçi sayısını kullan", value=True)
        if use_optimal:
            workers = optimal_workers
        else:
            workers = workers_input

        holding_cost = st.number_input("Stok Maliyeti (TL)", min_value=1, max_value=100, value=int(model2.holding_cost), key="m2_holding", step=1)
        labor_per_unit = st.number_input("Birim İşgücü (saat)", min_value=0.1, max_value=10.0, value=float(model2.labor_per_unit), key="m2_labor", step=0.1)
        daily_hours = st.number_input("Günlük Çalışma Saati", min_value=1, max_value=24, value=model2.daily_hours, key="m2_daily", step=1)
        overtime_wage_multiplier = st.number_input("Fazla Mesai Ücret Çarpanı", min_value=1.0, max_value=5.0, value=model2.overtime_wage_multiplier, step=0.1, key="m2_overtime_multiplier")
        max_overtime_per_worker = st.number_input("Maks. Fazla Mesai (saat/işçi)", min_value=0, max_value=100, value=model2.max_overtime_per_worker, step=1, key="m2_max_overtime")
        stockout_cost = st.number_input("Stoksuzluk Maliyeti (TL/adet)", min_value=1, max_value=100, value=model2.stockout_cost, step=1, key="m2_stockout")
        normal_hourly_wage = st.number_input("Normal Saatlik İşçilik Maliyeti (TL)", min_value=1, max_value=1000, value=model2.normal_hourly_wage, step=1, key="m2_normal_wage")
        production_cost = st.number_input("Üretim Maliyeti (TL)", min_value=1, max_value=100, value=model2.production_cost, key="m2_prod_cost", step=1)
        hiring_cost = st.number_input("İşçi Alım Maliyeti (TL)", min_value=0, max_value=5000, value=int(model2.hiring_cost), step=1, key="m2_hire")
        run_model = st.button("Modeli Çalıştır", key="m2_run")
    if run_model:
        df_table, total_cost, optimal_workers = model2_run(
            demand, working_days, holding_cost, labor_per_unit, workers, daily_hours,
            overtime_wage_multiplier, max_overtime_per_worker, stockout_cost, normal_hourly_wage, production_cost
        )

        # Kullanıcıya optimal işçi sayısını göster
        if workers != optimal_workers:
            st.warning(f"Kullanılan işçi sayısı ({workers}) optimal seviyeden ({optimal_workers}) farklı. Model verimliliği etkilenebilir.")

        st.subheader("Sonuç Tablosu")
        st.dataframe(df_table, use_container_width=True, hide_index=True)
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
            df_table['Stok Maliyeti'].sum(), df_table['Stoksuzluk Maliyeti'].sum(), df_table['Fazla Mesai Maliyeti'].sum() if 'Fazla Mesai Maliyeti' in df_table.columns else 0, df_table['Normal İşçilik Maliyeti'].sum(), df_table['Üretim Maliyeti'].sum(), hiring_cost * workers
        )
        st.subheader("Ayrıntılı Toplam Maliyetler")
        st.markdown(f"- Stok Maliyeti Toplamı: {detay['total_holding']:,.2f} TL")
        st.markdown(f"- Stoksuzluk Maliyeti Toplamı: {detay['total_stockout']:,.2f} TL")
        st.markdown(f"- Fazla Mesai Maliyeti Toplamı: {detay['total_overtime']:,.2f} TL")
        st.markdown(f"- Normal İşçilik Maliyeti Toplamı: {detay['total_normal_labor']:,.2f} TL")
        st.markdown(f"- Üretim Maliyeti Toplamı: {detay['total_production']:,.2f} TL")
        st.markdown(f"- İşe Alım Maliyeti Toplamı: {detay['total_hiring_cost']:,.2f} TL")
        birim = m2_birim(
            demand, df_table['Üretim'], df_table['Stok'], total_cost, detay['total_normal_labor'], detay['total_overtime'], detay['total_production'], detay['total_holding'], detay['total_stockout'], production_cost, hiring_cost * workers
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
            st.markdown(f"- Diğer Maliyetler (Stok, Stoksuzluk, İşe Alım): {birim['other_unit_cost']:.2f} TL/birim")
        else:
            st.markdown("- Ortalama Birim Maliyet: Hesaplanamadı (0 birim üretildi)")

if model == "Toplu Üretim ve Stoklama (Model 3)":
    st.header("Toplu Üretim ve Stoklama (Model 3)")
    with st.sidebar:
        demand, workers, working_days, selected_demand_type = select_demand_type_and_workers("m3")
        holding_cost = st.number_input("Stok Maliyeti (TL)", min_value=1, max_value=100, value=int(model3.holding_cost), key="m3_holding", step=1)
        stockout_cost = st.number_input("Stoksuzluk Maliyeti (TL)", min_value=1, max_value=100, value=model3.stockout_cost, key="m3_stockout", step=1)
        labor_per_unit = st.number_input("Birim İşgücü (saat)", min_value=0.1, max_value=10.0, value=float(model3.labor_per_unit), step=0.1, key="m3_labor_per_unit")
        daily_hours = st.number_input("Günlük Çalışma Saati", min_value=1, max_value=24, value=int(model3.daily_hours), key="m3_daily", step=1)
        worker_monthly_cost = st.number_input("Aylık İşçi Ücreti (TL)", min_value=1, max_value=100000, value=int(model3.worker_monthly_cost), key="m3_worker_monthly_cost", step=1)
        production_cost = st.number_input("Üretim Maliyeti (TL)", min_value=1, max_value=100, value=int(model3.production_cost), key="m3_prod_cost", step=1)
        hiring_cost = st.number_input("İşçi Alım Maliyeti (TL)", min_value=0, max_value=5000, value=int(params['costs']['hiring_cost']), key="m3_hire", step=1)
        run_model = st.button("Modeli Çalıştır", key="m3_run")
    if run_model:
        df, cost, total_holding, total_stockout, total_labor, total_production_cost, total_demand, total_produced, total_unfilled, avg_unit_cost, avg_labor_unit, avg_prod_unit, avg_other_unit, total_hiring_cost, optimized_workers, original_workers, optimal_workers = model3_run(demand, working_days, holding_cost, stockout_cost, workers, labor_per_unit, daily_hours, production_cost, worker_monthly_cost)
        # Inform user if worker count was optimized
        if optimized_workers != original_workers:
            st.warning(f"Girdiğiniz işçi sayısı ({original_workers}) modele göre optimize edilerek {optimized_workers} olarak ayarlandı. (Optimal: {optimal_workers}, izin verilen aralık: {int(optimal_workers*0.9)} - {int(optimal_workers*1.1)})\nİşe alım maliyeti ve tüm hesaplamalar optimize edilen işçi sayısına göre yapılmıştır.")
        else:
            st.info(f"Kullandığınız işçi sayısı ({optimized_workers}) optimal aralıkta.")
        st.subheader("Sonuç Tablosu")
        st.dataframe(df, use_container_width=True, hide_index=True)
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
        st.markdown(f"- İşe Alım Maliyeti Toplamı: {total_hiring_cost:,.2f} TL")
        birim = m3_birim(demand, df['Üretim'], df['Stok'], cost, df, workers, hiring_cost)
        st.subheader("Birim Maliyet Analizi")
        st.markdown(f"- Toplam Talep: {birim['total_demand']:,} birim")
        st.markdown(f"- Toplam Üretim: {birim['total_produced']:,} birim ({birim['total_produced']/birim['total_demand']*100:.2f}%)")
        st.markdown(f"- Karşılanmayan Talep: {birim['total_unfilled']:,} birim ({birim['total_unfilled']/birim['total_demand']*100:.2f}%)")
        if birim['total_produced'] > 0:
            st.markdown(f"- Ortalama Birim Maliyet (Toplam): {birim['avg_unit_cost']:.2f} TL/birim")
            st.markdown(f"- İşçilik Birim Maliyeti: {birim['labor_unit_cost']:.2f} TL/birim")
            st.markdown(f"- Üretim Birim Maliyeti: {birim['prod_unit_cost']:.2f} TL/birim")
            st.markdown(f"- Diğer Maliyetler (Stok, Stoksuzluk, İşe Alım): {birim['other_unit_cost']:.2f} TL/birim")
            st.markdown(f"- İşe Alım Maliyeti: {birim['hiring_cost']:,} TL (İşçi başına {hiring_cost:,} TL)")
        else:
            st.markdown("- Ortalama Birim Maliyet: Hesaplanamadı (0 birim üretildi)")

if model == "Dinamik Programlama (Model 4)":
    st.header("Dinamik Programlama Tabanlı (Model 4)")
    with st.sidebar:
        demand, workers, working_days, selected_demand_type = select_demand_type_and_workers("m4")
        holding_cost = st.number_input("Stok Maliyeti (TL)", 1, 100, int(model4.holding_cost), key="m4_holding")
        stockout_cost = st.number_input("Karşılanmayan Talep Maliyeti (TL/adet)", 1, 100, int(model4.stockout_cost), key="m4_stockout")
        hiring_cost = st.number_input("İşçi Alım Maliyeti (TL)", 0, 5000, int(model4.hiring_cost), key="m4_hire")
        firing_cost = st.number_input("İşçi Çıkarma Maliyeti (TL)", 0, 5000, int(model4.firing_cost), key="m4_fire")
        daily_hours = st.number_input("Günlük Çalışma Saati", 1, 24, int(model4.daily_hours), key="m4_daily")
        labor_per_unit = st.number_input("Birim İşgücü (saat)", 0.1, 10.0, float(model4.labor_per_unit), key="m4_labor")
        production_cost = st.number_input("Üretim Maliyeti (TL)", min_value=1, max_value=100, value=int(model4.production_cost), step=1, key="m4_prod_cost")
        max_workers = st.number_input("Maksimum İşçi", 1, 100, int(model4.max_workers), key="m4_max_workers")
        max_workforce_change = st.number_input("Aylık İşgücü Değişim Sınırı", 1, 100, int(model4.max_workforce_change), key="m4_max_workforce")
        hourly_wage = st.number_input("Saatlik Ücret (TL)", min_value=1, max_value=1000, value=int(get_param('costs', 'hourly_wage', 10)), step=1, key="m4_hourly_wage")
        run_model = st.button("Modeli Çalıştır", key="m4_run")
    if run_model:
        df, min_cost, total_labor, total_production, total_holding, total_stockout, total_hiring, total_firing, total_demand, total_produced, total_unfilled, avg_unit_cost, avg_labor_unit, avg_prod_unit, avg_other_unit = model4_run(demand, working_days, holding_cost, hiring_cost, firing_cost, daily_hours, labor_per_unit, workers, max_workers, max_workforce_change, hourly_wage, stockout_cost,production_cost)
        st.subheader("Sonuç Tablosu")
        st.dataframe(df, use_container_width=True, hide_index=True)
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
        birim = m4_birim(total_demand, total_produced, total_unfilled, min_cost, total_labor, total_production, total_holding, total_stockout, total_hiring, total_firing)
        st.subheader("Birim Maliyet Analizi")
        st.markdown(f"- Toplam Talep: {birim['total_demand']:,} birim")
        st.markdown(f"- Toplam Üretim: {birim['total_produced']:,} birim ({(birim['total_produced']/birim['total_demand']*100 if birim['total_demand'] else 0):.2f}%)")
        st.markdown(f"- Karşılanmayan Talep: {birim['total_unfilled']:,} birim ({(birim['total_unfilled']/birim['total_demand']*100 if birim['total_demand'] else 0):.2f}%)")
        if birim['total_produced'] > 0:
            st.markdown(f"- Ortalama Birim Maliyet (Toplam): {birim['avg_unit_cost']:.2f} TL/birim")
            st.markdown(f"- İşçilik Birim Maliyeti: {birim['avg_labor_unit']:.2f} TL/birim")
            st.markdown(f"- Üretim Birim Maliyeti: {birim['avg_prod_unit']:.2f} TL/birim")
            st.markdown(f"- Diğer Birim Maliyetler (Stok+Alım+Çıkarma): {birim['avg_other_unit']:.2f} TL/birim")
        else:
            st.markdown("- Ortalama Birim Maliyet: Hesaplanamadı (0 birim üretildi)")

if model == "Dış Kaynak Karşılaştırma (Model 5)":
    st.header("Dış Kaynak Kullanımı Karşılaştırma (Model 5)")
    with st.sidebar:
        demand, workers, working_days, selected_demand_type = select_demand_type_and_workers("m5")
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
        st.dataframe(df, use_container_width=True, hide_index=True)
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
        demand, workers, working_days, selected_demand_type = select_demand_type_and_workers("m6")
        holding_cost = st.number_input("Stok Maliyeti (TL)", 1, 100, int(model6.holding_cost), key="m6_holding")
        stockout_cost = st.number_input("Stoksuzluk Maliyeti (TL/adet)", 1, 100, int(model6.stockout_cost), key="m6_stockout")
        production_cost = st.number_input("Üretim Maliyeti (TL)", 1, 100, int(model6.production_cost), key="m6_prod_cost")
        labor_per_unit = st.number_input("Birim İşgücü (saat)", 0.1, 10.0, float(model6.labor_per_unit), key="m6_labor_per_unit")
        hourly_wage = st.number_input("Saatlik Ücret (TL)", 1, 100, int(model6.hourly_wage), key="m6_hourly_wage")
        daily_hours = st.number_input("Günlük Çalışma Saati", 1, 24, 8, key="m6_daily_hours")
        hiring_cost = st.number_input("İşçi Alım Maliyeti (TL)", 0, 5000, int(model6.hiring_cost), key="m6_hire")
        firing_cost = st.number_input("İşçi Çıkarma Maliyeti (TL)", 0, 5000, int(model6.firing_cost), key="m6_fire")
        max_workers = st.number_input("Maksimum İşçi", 1, 100, int(model6.max_workers), key="m6_max_workers")
        max_workforce_change = st.number_input("Aylık İşgücü Değişim Sınırı", 1, 100, int(model6.max_workforce_change), key="m6_max_workforce")
        run_model = st.button("Modeli Çalıştır", key="m6_run")
    if run_model:
        df, total_cost, needed_workers, max_production = model6_run(
            demand, working_days, holding_cost, stockout_cost, production_cost, labor_per_unit, hourly_wage, daily_hours, hiring_cost, firing_cost, workers, max_workers, max_workforce_change
        )
        st.info(f"Maksimum Üretim Kapasitesi: {max_production} adet/ay | Gerekli Optimum İşçi Sayısı: {needed_workers}")
        st.subheader("Sonuç Tablosu")
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Display the total cost from the solver, which now includes hiring costs
        st.success(f"Toplam Maliyet: {total_cost:,.2f} TL")

        st.subheader("Grafiksel Sonuçlar")
        months_list = list(range(1, len(demand)+1))
        fig, ax = plt.subplots(figsize=(12,6))
        ax.plot(months_list, df['Talep'], marker='o', label='Talep', color='orange')
        ax.bar(months_list, df['Üretim'], color='skyblue', label='Üretim', alpha=0.7)
        ax.plot(months_list, df['Stok'], marker='d', label='Stok', color='red')
        ax.plot(months_list, df['Stoksuzluk'], marker='x', label='Stoksuzluk', color='black')
        if 'İşçi' in df.columns:
            ax.plot(months_list, df['İşçi'], marker='s', label='İşçi Sayısı', color='green')
        ax.set_xlabel('Ay')
        ax.set_ylabel('Adet')
        ax.set_title('Mevsimsellik ve Stok Optimizasyonu Sonuçları')
        ax.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        st.pyplot(fig)

        # Ayrıntılı Toplam Maliyetler
        # Convert string formatted costs to float for calculations
        total_holding = df['Stok Maliyeti'].astype(str).str.replace(' TL', '').str.replace(',', '').astype(float).sum()
        total_stockout = df['Stoksuzluk Maliyeti'].astype(str).str.replace(' TL', '').str.replace(',', '').astype(float).sum()
        total_production_cost = df['Üretim Maliyeti'].astype(str).str.replace(' TL', '').str.replace(',', '').astype(float).sum()
        total_labor_cost = df['İşçilik Maliyeti'].astype(str).str.replace(' TL', '').str.replace(',', '').astype(float).sum()

        # For hiring and firing costs, extract directly from the dataframe
        total_hiring_cost = df['İşe Alım Maliyeti'].astype(str).str.replace(' TL', '').str.replace(',', '').astype(float).sum()
        total_firing_cost = df['İşten Çıkarma Maliyeti'].astype(str).str.replace(' TL', '').str.replace(',', '').astype(float).sum()

        detay = m6_ayrintili(total_holding, total_stockout, total_production_cost, total_labor_cost, total_hiring_cost, total_firing_cost)
        st.subheader("Ayrıntılı Toplam Maliyetler")
        st.markdown(f"- Stok Maliyeti Toplamı: {detay['total_holding']:,.2f} TL")
        st.markdown(f"- Stoksuzluk Maliyeti Toplamı: {detay['total_stockout']:,.2f} TL")
        st.markdown(f"- Üretim Maliyeti Toplamı: {detay['total_production_cost']:,.2f} TL")
        st.markdown(f"- İşçilik Maliyeti Toplamı: {detay['total_labor_cost']:,.2f} TL")
        st.markdown(f"- İşe Alım Maliyeti Toplamı: {detay['total_hiring_cost']:,.2f} TL")
        st.markdown(f"- İşten Çıkarma Maliyeti Toplamı: {detay['total_firing_cost']:,.2f} TL")

        # Birim Maliyet Analizi
        total_demand = df['Talep'].sum()
        total_produced = df['Üretim'].sum()
        total_unfilled = df['Stoksuzluk'].sum()
        birim = m6_birim(total_demand, total_produced, total_unfilled, total_cost, total_labor_cost, total_production_cost, total_holding, total_stockout, total_hiring_cost, total_firing_cost)

        st.subheader("Birim Maliyet Analizi")
        st.markdown(f"- Toplam Talep: {birim['total_demand']:,} birim")
        st.markdown(f"- Toplam Üretim: {birim['total_produced']:,} birim ({(birim['total_produced']/birim['total_demand']*100 if birim['total_demand'] else 0):.2f}%)")
        st.markdown(f"- Karşılanmayan Talep: {birim['total_unfilled']:,} birim ({(birim['total_unfilled']/birim['total_demand']*100 if birim['total_demand'] else 0):.2f}%)")
        if birim['total_produced'] > 0:
            st.markdown(f"- Ortalama Birim Maliyet (Toplam): {birim['avg_unit_cost']:.2f} TL/birim")
            st.markdown(f"- İşçilik Birim Maliyeti: {birim['avg_labor_unit']:.2f} TL/birim")
            st.markdown(f"- Üretim Birim Maliyeti: {birim['avg_prod_unit']:.2f} TL/birim")
            st.markdown(f"- Diğer Birim Maliyetler (Stok, Stoksuzluk, İşe Alım/Çıkarma): {birim['avg_other_unit']:.2f} TL/birim")
        else:
            st.markdown("- Ortalama Birim Maliyet: Hesaplanamadı (0 birim üretildi)")

if model == "Modelleri Karşılaştır":
    st.header("Modelleri Karşılaştır")
    with st.sidebar:
        demand, workers, working_days, selected_demand_type = select_demand_type_and_workers("cmp")
        holding_cost = st.number_input("Stok Maliyeti (TL)", min_value=1, max_value=100, value=int(get_param('costs', 'holding_cost', 5)), step=1, key="cmp_holding")
        stockout_cost = st.number_input("Karşılanmayan Talep Maliyeti (TL/adet)", min_value=1, max_value=100, value=int(get_param('costs', 'stockout_cost', 80)), step=1, key="cmp_stockout")
        production_cost = st.number_input("Birim Üretim Maliyeti (TL)", min_value=1, max_value=1000, value=int(get_param('costs', 'production_cost', 30)), step=1, key="cmp_production_cost")
        hiring_cost = st.number_input("İşçi Alım Maliyeti (TL)", min_value=0, max_value=5000, value=int(get_param('costs', 'hiring_cost', 1800)), step=1, key="cmp_hire")
        firing_cost = st.number_input("İşçi Çıkarma Maliyeti (TL)", min_value=0, max_value=5000, value=int(get_param('costs', 'firing_cost', 1500)), step=1, key="cmp_fire")
        hourly_wage = st.number_input("Saatlik Ücret (TL)", min_value=1, max_value=1000, value=int(get_param('costs', 'hourly_wage', 10)), step=1, key="cmp_hourly_wage")
        daily_hours = st.number_input("Günlük Çalışma Saati", min_value=1, max_value=24, value=int(get_param('workforce', 'daily_hours', 8)), step=1, key="cmp_daily_hours")
        labor_per_unit = st.number_input("Birim İşgücü (saat)", min_value=0.1, max_value=10.0, value=float(get_param('workforce', 'labor_per_unit', 4)), step=0.1, key="cmp_labor")
        max_overtime_per_worker = st.number_input("Maks. Fazla Mesai (saat/işçi)", min_value=0, max_value=100, value=int(get_param('costs', 'max_overtime_per_worker', 20)), step=1, key="cmp_max_overtime")
        overtime_wage_multiplier = st.number_input("Fazla Mesai Ücret Çarpanı", min_value=1.0, max_value=5.0, value=float(get_param('costs', 'overtime_wage_multiplier', 1.5)), step=0.1, key="cmp_overtime_multiplier")

        # Add more fields as needed for other models
        compare_btn = st.button("Modelleri Karşılaştır", key="cmp_compare_btn")

    # Run comparison on first load or when button pressed
    if compare_btn or st.session_state.get("cmp_first_run", True):
        st.session_state["cmp_first_run"] = False

        # Collect current parameters from sidebar inputs and session state
        # 'demand' and 'working_days' are already defined in the scope
        current_params = {
            "demand": demand,
            "holding_cost": st.session_state.cmp_holding,
            "stockout_cost": st.session_state.cmp_stockout,
            "production_cost": st.session_state.cmp_production_cost,
            "hiring_cost": st.session_state.cmp_hire,
            "firing_cost": st.session_state.cmp_fire,
            "hourly_wage": st.session_state.cmp_hourly_wage,
            "daily_hours": st.session_state.cmp_daily_hours,
            "labor_per_unit": st.session_state.cmp_labor,
            "max_overtime_per_worker": st.session_state.cmp_max_overtime,
            "overtime_wage_multiplier": st.session_state.cmp_overtime_multiplier,
            "working_days": working_days,
            "workers": st.session_state.cmp_workers
        }

        model_names = [
            ("Model 1", "Karma Planlama (Model 1)", model1.maliyet_analizi, "Yüksek", "Karma planlama, işgücü ve fason esnekliği"),
            ("Model 2", "Fazla Mesaili Üretim (Model 2)", model2.maliyet_analizi, "Orta", "Fazla mesai ile esneklik"),
            ("Model 3", "Toplu Üretim ve Stoklama (Model 3)", model3.maliyet_analizi, "Düşük", "Sabit işgücü, toplu üretim"),
            ("Model 4", "Dinamik Programlama (Model 4)", model4.maliyet_analizi, "Orta", "Dinamik işgücü, sınırlı üretim"),
            ("Model 5", "Dış Kaynak Karşılaştırma (Model 5)", model5.maliyet_analizi, "Yok", "Tam fason kullanımı"),
            ("Model 6", "Mevsimsellik ve Dalga (Model 6)", model6.maliyet_analizi, "Orta", "Mevsimsellik ve stok optimizasyonu"),
        ]
        summary_rows = []
        for idx, (short_name, display_name, func, flex, scenario) in enumerate(model_names, 1):
            try:
                sig_params = inspect.signature(func).parameters
                params_to_pass = {}

                for p_name, p_value in current_params.items():
                    if short_name == "Model 2" and p_name == "hourly_wage":
                        if "normal_hourly_wage" in sig_params:
                            params_to_pass["normal_hourly_wage"] = p_value
                    elif p_name in sig_params:
                        params_to_pass[p_name] = p_value

                # For Model 3, make sure we're including the hiring_cost
                if short_name == "Model 3" and "hiring_cost" in sig_params:
                    params_to_pass["hiring_cost"] = current_params.get("hiring_cost")

                res = func(**params_to_pass)
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
                st.error(f"{short_name} hata: {str(e)}")

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
        # --- Grafiksel Karşılaştırma ---
        st.subheader("Ana Metriklerde Grafiksel Karşılaştırma")
        # Sadece sayısal metrikleri al ve ortalama birim maliyeti hesapla
        metrics = ["Toplam Maliyet (₺)", "Ortalama Birim Maliyet", "Toplam Üretim", "Stoksuzluk Oranı (%)"]
        plot_df = summary_df[["Toplam Maliyet (₺)", "Toplam Üretim", "Stoksuzluk Oranı (%)"]].copy()
        # Ortalama birim maliyet hesapla
        plot_df["Ortalama Birim Maliyet"] = plot_df["Toplam Maliyet (₺)"] / plot_df["Toplam Üretim"]
        # Sütun sırasını ayarla
        plot_df = plot_df[["Toplam Maliyet (₺)", "Ortalama Birim Maliyet", "Toplam Üretim", "Stoksuzluk Oranı (%)"]]
        plot_df = plot_df.apply(pd.to_numeric, errors='coerce')
        fig, axes = plt.subplots(1, len(metrics), figsize=(4*len(metrics), 5))
        if len(metrics) == 1:
            axes = [axes]
        for i, metric in enumerate(metrics):
            ax = axes[i]
            ax.bar(plot_df.index, plot_df[metric], color='steelblue', alpha=0.8)
            ax.set_title(metric)
            ax.set_xticklabels(plot_df.index, rotation=30, ha='right')
            ax.grid(axis='y', linestyle='--', alpha=0.5)
            if metric == "Stoksuzluk Oranı (%)":
                ax.set_ylim(0, 100)
        plt.tight_layout()
        st.pyplot(fig)
        st.markdown("---")
        # Detaylı tabloyu da göster
        st.subheader("Detaylı Karşılaştırma Tablosu")
        results_detail_list = []
        for short_name, display_name, func, _, _ in model_names:
            try:
                sig_params = inspect.signature(func).parameters
                params_to_pass_detail = {}
                for p_name, p_value in current_params.items():
                    if short_name == "Model 2" and p_name == "hourly_wage":
                        if "normal_hourly_wage" in sig_params:
                            params_to_pass_detail["normal_hourly_wage"] = p_value
                    elif p_name in sig_params:
                        params_to_pass_detail[p_name] = p_value

                res_detail = func(**params_to_pass_detail)
                res_detail["Model"] = display_name
                results_detail_list.append(res_detail)
            except Exception as e:
                results_detail_list.append({"Model": display_name, "Hata": str(e)})
                # Optionally, display this error in the detailed section as well
                # st.error(f"{display_name} için detaylı tablo hatası: {str(e)}")
        df_detail = pd.DataFrame(results_detail_list)

        cols = ["Model"] + [c for c in df_detail.columns if c != "Model"]
        st.dataframe(df_detail[cols], use_container_width=True, hide_index=True)

        # Add explanation about Model 1's production cost
        if any(name[0] == "Model 1" for name in model_names):
            st.info("Model 1'de Üretim Birim Maliyeti, iç üretim ve fason üretimin ağırlıklı ortalamasıdır.")
