import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yaml
import inspect
import gc  # Garbage collection için
import psutil  # Memory monitoring için (opsiyonel)
import time
import weakref
from contextlib import contextmanager

# Configure matplotlib for memory optimization
plt.ioff()  # Turn off interactive mode
plt.rcParams['figure.max_open_warning'] = 0  # Disable warning for too many figures
plt.rcParams['font.size'] = 8  # Smaller font size to save memory

# Configure pandas for memory optimization
pd.set_option('display.max_columns', None)
pd.set_option('display.precision', 2)

# Configure streamlit for memory optimization
st.set_page_config(
    page_title="Üretim Planlama Modelleri", 
    layout="wide", 
    initial_sidebar_state="expanded",
    menu_items={
        'About': "Üretim Planlama Modelleri - Bellek Optimize Edilmiş Versiyon"
    }
)

# Memory optimization: Limit cache size globally
if 'cache_cleanup_counter' not in st.session_state:
    st.session_state.cache_cleanup_counter = 0

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

# Memory management utilities
@contextmanager
def memory_context():
    """Context manager for memory cleanup"""
    try:
        yield
    finally:
        safe_memory_cleanup()

def clear_memory():
    """Enhanced memory clearing function"""
    # Clear matplotlib figures
    plt.close('all')
    
    # Clear streamlit cache
    if hasattr(st, 'cache_data'):
        try:
            st.cache_data.clear()
        except:
            pass
    
    # Clear session state of large objects
    keys_to_clear = [k for k in st.session_state.keys() if k.startswith('model_results_')]
    for key in keys_to_clear:
        del st.session_state[key]
    
    # Force garbage collection multiple times
    for _ in range(3):
        gc.collect()
    
    # Clear numpy cache if available
    try:
        np.clear_cache()
    except:
        pass

def monitor_memory():
    """Memory monitoring with error handling"""
    try:
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        return memory_mb
    except:
        return None

def safe_memory_cleanup():
    """Enhanced safe memory cleanup"""
    try:
        # Close all matplotlib figures
        plt.close('all')
        
        # Clear matplotlib's font cache
        if hasattr(plt, 'fontManager'):
            try:
                plt.fontManager._rebuild()
            except:
                pass
        
        # Force garbage collection
        gc.collect()
        
        # Clear pandas cache
        try:
            pd.set_option('mode.chained_assignment', None)
        except:
            pass
            
    except Exception as e:
        # Silent fail - don't interrupt user experience
        pass

# Optimized cache settings - reduced TTL and entries
@st.cache_data(ttl=120, max_entries=1, show_spinner=False)  # 2 minutes TTL, max 1 entry
def run_model1_solver(*args, **kwargs):
    with memory_context():
        result = model1_solver(*args, **kwargs)
        return result

@st.cache_data(ttl=120, max_entries=1, show_spinner=False)
def run_model2_solver(*args, **kwargs):
    with memory_context():
        result = model2_solver(*args, **kwargs)
        return result

@st.cache_data(ttl=120, max_entries=1, show_spinner=False)
def run_model3_solver(*args, **kwargs):
    with memory_context():
        result = model3_solver(*args, **kwargs)
        return result

@st.cache_data(ttl=120, max_entries=1, show_spinner=False)
def run_model4_solver(*args, **kwargs):
    with memory_context():
        result = model4_solver(*args, **kwargs)
        return result

@st.cache_data(ttl=120, max_entries=1, show_spinner=False)
def run_model5_solver(*args, **kwargs):
    with memory_context():
        result = model5_solver(*args, **kwargs)
        return result

@st.cache_data(ttl=120, max_entries=1, show_spinner=False)
def run_model6_solver(*args, **kwargs):
    with memory_context():
        result = model6_solver(*args, **kwargs)
        return result

st.title("Üretim Planlama Modelleri Karar Destek Arayüzü")
# st.markdown("*Bellek Optimize Edilmiş Versiyon*")

# Load parameters from YAML - optimized caching
@st.cache_data(ttl=300, max_entries=1, show_spinner=False)  # 5 minutes TTL, only 1 entry
def load_params():
    with open("parametreler.yaml", "r") as f:
        return yaml.safe_load(f)

# Initialize app with memory optimization
if 'memory_check_count' not in st.session_state:
    st.session_state.memory_check_count = 0

# Periodic memory cleanup
st.session_state.memory_check_count += 1
if st.session_state.memory_check_count % 10 == 0:  # Every 10 interactions
    safe_memory_cleanup()

params = load_params()

model = st.sidebar.selectbox("Model Seçiniz", params['models'])

st.sidebar.markdown("---")

def get_param(key, subkey=None, default=None):
    try:
        if subkey:
            return params[key][subkey]
        return params[key]
    except Exception:
        return default

def model1_run(demand, working_days, holding_cost, outsourcing_cost, labor_per_unit, hiring_cost, firing_cost, daily_hours, outsourcing_capacity, min_internal_ratio, max_workforce_change, max_outsourcing_ratio, stockout_cost, hourly_wage, production_cost, overtime_wage_multiplier, max_overtime_per_worker, initial_inventory, safety_stock_ratio, fixed_workers=None):
    """Optimized Model 1 runner with memory management"""
    with memory_context():
        # Use the shared model solver function from model1_mixed_planning
        model_vars = run_model1_solver(
            demand, working_days, holding_cost, outsourcing_cost, labor_per_unit,
            hiring_cost, firing_cost, daily_hours, outsourcing_capacity, min_internal_ratio,
            max_workforce_change, max_outsourcing_ratio, stockout_cost, hourly_wage,
            production_cost, overtime_wage_multiplier, max_overtime_per_worker, initial_inventory, safety_stock_ratio, fixed_workers
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

        # Create results dataframe - optimized
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

        # Model değişkenlerini dictionary olarak döndür - only essential data
        model_results = {
            'internal_production': internal_production,
            'outsourced_production': outsourced_production,
            'inventory': inventory,
            'hired': hired,
            'fired': fired,
            'stockout': stockout,
            'overtime_hours': overtime_hours
        }

        # Calculate optimal workers if using fixed workers
        if fixed_workers is not None:
            # Run optimization mode to get the true optimal worker count
            optimal_model_vars = run_model1_solver(
                demand, working_days, holding_cost, outsourcing_cost, labor_per_unit,
                hiring_cost, firing_cost, daily_hours, outsourcing_capacity, min_internal_ratio,
                max_workforce_change, max_outsourcing_ratio, stockout_cost, hourly_wage,
                production_cost, overtime_wage_multiplier, max_overtime_per_worker, initial_inventory, safety_stock_ratio, None  # No fixed workers
            )
            optimal_workers = [int(optimal_model_vars['workers'][t].varValue) for t in range(len(demand))]
            optimal_avg_workers = sum(optimal_workers) / len(optimal_workers)
            # Clean up optimal results
            del optimal_model_vars
            safe_memory_cleanup()
        else:
            # If not using fixed workers, the result is already optimal
            optimal_avg_workers = sum([int(workers[t].varValue) for t in range(len(demand))]) / len(demand)

        # Clear intermediate variables
        del model_vars, results, workers, internal_production, outsourced_production, inventory, hired, fired, stockout, overtime_hours

        return df, toplam_maliyet, model_results, optimal_avg_workers

def model2_run(demand, working_days, holding_cost, labor_per_unit, workers, daily_hours, overtime_wage_multiplier, max_overtime_per_worker, stockout_cost, normal_hourly_wage, production_cost, initial_inventory, safety_stock_ratio):
    """Optimized Model 2 runner with memory management"""
    with memory_context():
        # Use the shared model solver function
        model_results = run_model2_solver(
            demand, working_days, holding_cost, labor_per_unit, workers,
            daily_hours, overtime_wage_multiplier, max_overtime_per_worker,
            stockout_cost, normal_hourly_wage, production_cost,
            initial_inventory, safety_stock_ratio
        )

        # Get the dataframe and total_cost from model_results
        df = model_results['df'].copy()  # Make a copy to avoid reference issues
        total_cost = model_results['total_cost']
        optimal_workers = model_results.get('optimal_workers', workers)

        # Clear large objects immediately
        del model_results
        
        return df, total_cost, optimal_workers

def model3_run(demand, working_days, holding_cost, stockout_cost, workers, labor_per_unit, daily_hours, production_cost, worker_monthly_cost, initial_inventory, safety_stock_ratio):
    # Use the shared model solver function
    model_results = run_model3_solver(
        demand, working_days, holding_cost, stockout_cost,
        workers, labor_per_unit, daily_hours,
        production_cost, worker_monthly_cost, initial_inventory, safety_stock_ratio
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

    # Bellek temizliği
    del model_results
    safe_memory_cleanup()

    return df, adjusted_cost, total_holding, total_stockout, total_labor, total_production_cost, total_demand, total_produced, total_unfilled, avg_unit_cost, avg_labor_unit, avg_prod_unit, avg_other_unit, total_hiring_cost, optimized_workers, original_workers, optimal_workers

def model4_run(demand, working_days, holding_cost, hiring_cost, firing_cost, daily_hours, labor_per_unit, workers, max_workers, max_workforce_change, hourly_wage, stockout_cost, production_cost,initial_inventory, safety_stock_ratio):
    # Use the shared model solver function
    model_results = run_model4_solver(
        demand, working_days, holding_cost, stockout_cost, hiring_cost, firing_cost,
        daily_hours, labor_per_unit, workers, max_workers, max_workforce_change,
        hourly_wage, production_cost,initial_inventory, safety_stock_ratio
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

    # Bellek temizliği
    del model_results
    safe_memory_cleanup()

    return df, min_cost, total_labor, total_production, total_holding, total_stockout, total_hiring, total_firing, total_demand, total_produced, total_unfilled, avg_unit_cost, avg_labor_unit, avg_prod_unit, avg_other_unit

def model5_run(demand, holding_cost, cost_supplier_A, cost_supplier_B, capacity_supplier_A, capacity_supplier_B, working_days, stockout_cost,initial_inventory, safety_stock_ratio):
    # Use the shared model solver function
    model_results = run_model5_solver(
        demand, working_days, holding_cost, stockout_cost,
        cost_supplier_A, cost_supplier_B,
        capacity_supplier_A, capacity_supplier_B,initial_inventory, safety_stock_ratio
    )

    # Extract the dataframe and total cost from the results
    df = model_results['df'][['Ay', 'Tedarikçi A', 'Tedarikçi B', 'Stok', 'Karşılanmayan Talep']]
    toplam_maliyet = model_results['toplam_maliyet']

    # Bellek temizliği
    del model_results
    safe_memory_cleanup()

    return df, toplam_maliyet

def model6_run(demand, working_days, holding_cost, stockout_cost, production_cost, labor_per_unit, hourly_wage, daily_hours, hiring_cost, firing_cost, workers, max_workers, max_workforce_change,initial_inventory, safety_stock_ratio, fixed_workers=None):
    # Use the shared model solver function with fixed workers
    model_results = run_model6_solver(
        demand, holding_cost, stockout_cost, production_cost,
        labor_per_unit, hourly_wage, daily_hours, working_days,
        hiring_cost, firing_cost, workers, max_workers, max_workforce_change,initial_inventory, safety_stock_ratio, fixed_workers
    )

    # Extract the results from the model
    df = model_results['df']
    total_cost = model_results['total_cost']

    # Get the maximum production capacity
    max_production = int(df['Üretim'].max())
    
    # If using fixed workers, also run optimization to get the true optimal worker count
    if fixed_workers is not None:
        # Run optimization mode to get the true optimal worker count
        optimal_results = run_model6_solver(
            demand, holding_cost, stockout_cost, production_cost,
            labor_per_unit, hourly_wage, daily_hours, working_days,
            hiring_cost, firing_cost, workers, max_workers, max_workforce_change,initial_inventory, safety_stock_ratio, None  # No fixed workers for optimization
        )
        needed_workers = int(optimal_results['df']['İşçi'].mean())
        # Clean up optimal results
        del optimal_results
        safe_memory_cleanup()
    else:
        # If not using fixed workers, the result is already optimal
        needed_workers = int(df['İşçi'].mean())

    # Bellek temizliği
    del model_results
    safe_memory_cleanup()

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
        "seasonal": "Mevsimsel Talep",
        "highSeasonal": "Mevsimsel Yüksek Talep"
    }
    # Get the demand types from params
    demand_types = list(params['demand'].keys())

    # Create options list with Turkish display names
    display_options = [demand_type_names[key] for key in demand_types]

    # Set default index for Model 6 to "Mevsimsel Talep"
    default_index = 0
    if model_key == "m6":
        try:
            default_index = display_options.index("Mevsimsel Talep")
        except ValueError:
            default_index = 0

    selected_display = st.selectbox("Talep Tipi Seçiniz", display_options, index=default_index, key=f"{model_key}_demand_type")

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
    if selected_demand_type == "high" or selected_demand_type == "highSeasonal":
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
    st.info("💡 **Yeni Özellik**: Bu modelde işçi sayısını sabitleyebilirsiniz. Belirlediğiniz işçi sayısı kullanılacak ve model bu kısıt altında en optimal çözümü bulacaktır.")
    with st.sidebar:
        demand, workers, working_days, selected_demand_type = select_demand_type_and_workers("m1")
        holding_cost = st.number_input("Stok Maliyeti (TL)", min_value=1, max_value=100, value=int(model1.holding_cost), step=1, key="m1_holding")
        outsourcing_cost = st.number_input("Fason Maliyeti (TL)", min_value=1, max_value=100, value=int(model1.outsourcing_cost), step=1, key="m1_outsourcing")
        stockout_cost = st.number_input("Karşılanmayan Talep Maliyeti (TL/adet)", min_value=1, max_value=500, value=int(model1.stockout_cost), step=1, key="m1_stockout")
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
        initial_inventory = 0  # Kullanıcıdan istemiyoruz, default 0 gönderiyoruz.
        safety_stock_ratio = st.slider("Güvenlik Stoku Oranı (%)", min_value=0, max_value=100, value=5, key="m1_safety_stock_ratio") / 100
        run_model = st.button("Modeli Çalıştır", key="m1_run")
    if run_model or st.session_state.get("m1_first_run", True):
        st.session_state["m1_first_run"] = False
        df, toplam_maliyet, model_vars, optimal_avg_workers = model1_run(
            demand, working_days, holding_cost, outsourcing_cost, labor_per_unit, hiring_cost, firing_cost, daily_hours,
            outsourcing_capacity, min_internal_ratio, max_workforce_change, max_outsourcing_ratio, stockout_cost,
            hourly_wage, production_cost, overtime_wage_multiplier=overtime_wage_multiplier, max_overtime_per_worker=max_overtime_per_worker,
            initial_inventory=initial_inventory, safety_stock_ratio=safety_stock_ratio, fixed_workers=workers
        )
        
        # Show worker info
        current_avg_workers = df['İşçi'].mean()
        if abs(current_avg_workers - optimal_avg_workers) > 0.5:  # Allow small rounding differences
            st.info(f"Kullanılan Ortalama İşçi Sayısı: {current_avg_workers:.1f} | Gerçek Optimum Ortalama İşçi Sayısı: {optimal_avg_workers:.1f}")
            if current_avg_workers > optimal_avg_workers:
                st.warning(f"⚠️ Ortalama {current_avg_workers - optimal_avg_workers:.1f} fazla işçi kullanıyorsunuz.")
            else:
                st.warning(f"⚠️ Optimum için ortalama {optimal_avg_workers - current_avg_workers:.1f} daha fazla işçi gerekli.")
        else:
            st.success(f"✅ Optimum Ortalama İşçi Sayısı Kullanılıyor: {current_avg_workers:.1f}")
        
        st.subheader("Sonuç Tablosu")
        # Display the results table with formatted numbers
        cols = ["Ay"] + [c for c in df.columns if c != "Ay"]
        number_cols = df[cols].select_dtypes(include=["number"]).columns
        st.dataframe(
            df[cols].style.format({col: "{:,.0f}".format for col in number_cols}, thousands=".", decimal=","),
            use_container_width=True, hide_index=True
        )
        st.success(f"Toplam Maliyet: {toplam_maliyet:,.2f} TL")
        st.subheader("Grafiksel Sonuçlar")
        with memory_context():
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
            plt.close(fig)
            del fig, ax1, ax2, months

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
            st.session_state.get("m2_daily", model2.daily_hours),
            st.session_state.get("m2_labor", model2.labor_per_unit),
            st.session_state.get("m2_max_overtime", model2.max_overtime_per_worker)
        )
        st.info(f"Talep ve parametrelere göre optimal işçi sayısı: {optimal_workers}")
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
        stockout_cost = st.number_input("Stoksuzluk Maliyeti (TL/adet)", min_value=1, max_value=500, value=model2.stockout_cost, step=1, key="m2_stockout")
        normal_hourly_wage = st.number_input("Normal Saatlik İşçilik Maliyeti (TL)", min_value=1, max_value=1000, value=model2.normal_hourly_wage, step=1, key="m2_normal_wage")
        production_cost = st.number_input("Üretim Maliyeti (TL)", min_value=1, max_value=100, value=model2.production_cost, key="m2_prod_cost", step=1)
        hiring_cost = st.number_input("İşçi Alım Maliyeti (TL)", min_value=0, max_value=5000, value=int(model2.hiring_cost), step=1, key="m2_hire")
        initial_inventory = 0
        safety_stock_ratio = st.slider("Güvenlik Stoku Oranı (%)", min_value=0, max_value=100, value=5, key="m2_safety_stock_ratio") / 100
        run_model = st.button("Modeli Çalıştır", key="m2_run")
    if run_model or st.session_state.get("m2_first_run", True):
        st.session_state["m2_first_run"] = False
        df_table, total_cost, optimal_workers = model2_run(
            demand, working_days, holding_cost, labor_per_unit, workers, daily_hours,
            overtime_wage_multiplier, max_overtime_per_worker, stockout_cost, normal_hourly_wage, production_cost,
            initial_inventory=initial_inventory, safety_stock_ratio=safety_stock_ratio
        )
        if workers != optimal_workers:
            st.warning(f"Kullanılan işçi sayısı ({workers}) optimal seviyeden ({optimal_workers}) farklı. Model verimliliği etkilenebilir.")
        st.subheader("Sonuç Tablosu")
        cols = ["Ay"] + [c for c in df_table.columns if c != "Ay"]
        number_cols = df_table[cols].select_dtypes(include=["number"]).columns
        st.dataframe(
            df_table[cols].style.format({col: "{:,.0f}".format for col in number_cols}, thousands=".", decimal=","),
            use_container_width=True, hide_index=True
        )
        st.success(f"Toplam Maliyet: {total_cost:,.2f} TL")
        st.subheader("Grafiksel Sonuçlar")
        months_list = df_table["Ay"].tolist()
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax1.bar(months_list, df_table["Üretim"], color='skyblue', label='Üretim', alpha=0.7)
        ax1.set_xlabel('Ay')
        ax1.set_ylabel('Üretim (adet)', color='skyblue')
        ax1.tick_params(axis='y', labelcolor='skyblue')
        ax2 = ax1.twinx()
        ax2.plot(months_list, df_table["Fazla Mesai"], marker='o', label='Fazla Mesai (saat)', color='#FF8C00',
                 linewidth=2)
        ax2.plot(months_list, df_table["Stok"], marker='d', label='Stok', color='red', linestyle='--')
        ax2.set_ylabel('Fazla Mesai (saat)', color='#FF8C00')
        ax2.tick_params(axis='y', labelcolor='#FF8C00')
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
        plt.title('Fazla Mesaili Üretim Modeli Sonuçları')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

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
        stockout_cost = st.number_input("Stoksuzluk Maliyeti (TL)", min_value=1, max_value=500, value=model3.stockout_cost, key="m3_stockout", step=1)
        labor_per_unit = st.number_input("Birim İşgücü (saat)", min_value=0.1, max_value=10.0, value=float(model3.labor_per_unit), step=0.1, key="m3_labor_per_unit")
        daily_hours = st.number_input("Günlük Çalışma Saati", min_value=1, max_value=24, value=int(model3.daily_hours), key="m3_daily", step=1)
        worker_monthly_cost = st.number_input("Aylık İşçi Ücreti (TL)", min_value=1, max_value=100000, value=int(model3.worker_monthly_cost), key="m3_worker_monthly_cost", step=1)
        production_cost = st.number_input("Üretim Maliyeti (TL)", min_value=1, max_value=100, value=int(model3.production_cost), key="m3_prod_cost", step=1)
        hiring_cost = st.number_input("İşçi Alım Maliyeti (TL)", min_value=0, max_value=5000, value=int(params['costs']['hiring_cost']), key="m3_hire", step=1)
        initial_inventory = 0
        safety_stock_ratio = st.slider("Güvenlik Stoku Oranı (%)", min_value=0, max_value=100, value=5, key="m3_safety_stock_ratio") / 100
        run_model = st.button("Modeli Çalıştır", key="m3_run")
    if run_model or st.session_state.get("m3_first_run", True):
        st.session_state["m3_first_run"] = False
        (df, cost, total_holding, total_stockout, total_labor, total_production_cost, total_demand, total_produced, total_unfilled, avg_unit_cost, avg_labor_unit, avg_prod_unit, avg_other_unit, total_hiring_cost, optimized_workers,
         original_workers, optimal_workers) = model3_run(demand, working_days, holding_cost, stockout_cost, workers, labor_per_unit, daily_hours, production_cost, worker_monthly_cost,initial_inventory=initial_inventory, safety_stock_ratio=safety_stock_ratio)
        # Inform user if worker count was optimized
        if optimized_workers != original_workers:
            st.warning(f"Girdiğiniz işçi sayısı ({original_workers}) modele göre optimize edilerek {optimized_workers} olarak ayarlandı. (Optimal: {optimal_workers}, izin verilen aralık: {int(optimal_workers*0.9)} - {int(optimal_workers*1.1)})\nİşe alım maliyeti ve tüm hesaplamalar optimize edilen işçi sayısına göre yapılmıştır.")
        else:
            st.info(f"Kullandığınız işçi sayısı ({optimized_workers}) optimal aralıkta.")
        st.subheader("Sonuç Tablosu")
        cols = ["Ay"] + [c for c in df.columns if c != "Ay"]
        number_cols = df[cols].select_dtypes(include=["number"]).columns
        st.dataframe(
            df[cols].style.format({col: "{:,.0f}".format for col in number_cols}, thousands=".", decimal=","),
            use_container_width=True, hide_index=True
        )
        st.success(f"Toplam Maliyet: {cost:,.2f} TL")
        st.subheader("Grafiksel Sonuçlar")
        months_list = df["Ay"].tolist()
        fig, ax = plt.subplots(figsize=(10,6))
        ax.bar(months_list, df["Üretim"], color='skyblue', label='Üretim', alpha=0.7)
        ax.plot(months_list, df["Stok"], marker='d', label='Stok', color='red')
        ax.plot(months_list, df["Karşılanmayan Talep"], marker='x', label='Karşılanmayan Talep', color='black')
        ax.set_xlabel('Ay')
        ax.set_ylabel('Adet')
        ax.legend()
        plt.title('Toplu Üretim ve Stoklama Modeli Sonuçları')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

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
        else:
            st.markdown("- Ortalama Birim Maliyet: Hesaplanamadı (0 birim üretildi)")

if model == "Dinamik Programlama (Model 4)":
    st.header("Dinamik Programlama Tabanlı (Model 4)")
    with st.sidebar:
        demand, workers, working_days, selected_demand_type = select_demand_type_and_workers("m4")
        holding_cost = st.number_input("Stok Maliyeti (TL)", 1, 100, int(model4.holding_cost), key="m4_holding")
        stockout_cost = st.number_input("Karşılanmayan Talep Maliyeti (TL/adet)", 1, 500, int(model4.stockout_cost), key="m4_stockout")
        hiring_cost = st.number_input("İşçi Alım Maliyeti (TL)", 0, 5000, int(model4.hiring_cost), key="m4_hire")
        firing_cost = st.number_input("İşçi Çıkarma Maliyeti (TL)", 0, 5000, int(model4.firing_cost), key="m4_fire")
        daily_hours = st.number_input("Günlük Çalışma Saati", 1, 24, int(model4.daily_hours), key="m4_daily")
        labor_per_unit = st.number_input("Birim İşgücü (saat)", 0.1, 10.0, float(model4.labor_per_unit), key="m4_labor")
        production_cost = st.number_input("Üretim Maliyeti (TL)", min_value=1, max_value=100, value=int(model4.production_cost), step=1, key="m4_prod_cost")
        max_workers = st.number_input("Maksimum İşçi", 1, 100, int(model4.max_workers), key="m4_max_workers")
        max_workforce_change = st.number_input("Aylık İşgücü Değişim Sınırı", 1, 100, int(model4.max_workforce_change), key="m4_max_workforce")
        hourly_wage = st.number_input("Saatlik Ücret (TL)", min_value=1, max_value=1000, value=int(get_param('costs', 'hourly_wage', 10)), step=1, key="m4_hourly_wage")
        initial_inventory = 0
        safety_stock_ratio = st.slider("Güvenlik Stoku Oranı (%)", min_value=0, max_value=100, value=5, key="m4_safety_stock_ratio") / 100
        run_model = st.button("Modeli Çalıştır", key="m4_run")
    if run_model or st.session_state.get("m4_first_run", True):
        st.session_state["m4_first_run"] = False
        (df, min_cost, total_labor, total_production, total_holding, total_stockout, total_hiring, total_firing, total_demand, total_produced,
         total_unfilled, avg_unit_cost, avg_labor_unit, avg_prod_unit, avg_other_unit) = model4_run(
            demand, working_days, holding_cost, hiring_cost, firing_cost, daily_hours, labor_per_unit,
            workers, max_workers, max_workforce_change, hourly_wage, stockout_cost,production_cost,
            initial_inventory=initial_inventory, safety_stock_ratio=safety_stock_ratio)
        st.subheader("Sonuç Tablosu")
        cols = ["Ay"] + [c for c in df.columns if c != "Ay"]
        number_cols = df[cols].select_dtypes(include=["number"]).columns
        st.dataframe(
            df[cols].style.format({col: "{:,.0f}".format for col in number_cols}, thousands=".", decimal=","),
            use_container_width=True, hide_index=True
        )
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
        plt.close(fig)

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
        cost_supplier_A = st.number_input("Tedarikçi A Maliyeti (TL)", 1, 500, int(model5.cost_supplier_A), key="m5_cost_A")
        cost_supplier_B = st.number_input("Tedarikçi B Maliyeti (TL)", 1, 1000, int(model5.cost_supplier_B), key="m5_cost_B")
        capacity_supplier_A = st.number_input("Tedarikçi A Kapasitesi (adet)", 1, 10000, int(model5.capacity_supplier_A), key="m5_cap_A")
        capacity_supplier_B = st.number_input("Tedarikçi B Kapasitesi (adet)", 1, 100000, int(model5.capacity_supplier_B), key="m5_cap_B")
        stockout_cost = st.number_input("Stoksuzluk Maliyeti (TL/adet)", 1, 500, int(model5.stockout_cost), key="m5_stockout")
        initial_inventory = 0
        safety_stock_ratio = st.slider("Güvenlik Stoku Oranı (%)", min_value=0, max_value=100, value=5,
                                       key="m5_safety_stock_ratio") / 100
        run_model = st.button("Modeli Çalıştır", key="m5_run")
    if run_model or st.session_state.get("m5_first_run", True):
        st.session_state["m5_first_run"] = False
        df, toplam_maliyet = model5_run(
            demand, holding_cost, cost_supplier_A, cost_supplier_B, capacity_supplier_A, capacity_supplier_B, working_days, stockout_cost,initial_inventory=initial_inventory, safety_stock_ratio=safety_stock_ratio
        )
        st.subheader("Sonuç Tablosu")
        cols = ["Ay"] + [c for c in df.columns if c != "Ay"]
        number_cols = df[cols].select_dtypes(include=["number"]).columns
        st.dataframe(
            df[cols].style.format({col: "{:,.0f}".format for col in number_cols}, thousands=".", decimal=","),
            use_container_width=True, hide_index=True
        )
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
        plt.close(fig)

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
    st.info("💡 **Yeni Özellik**: Bu modelde işçi sayısını sabitleyebilirsiniz. Belirlediğiniz işçi sayısı kullanılacak ve model bu kısıt altında en optimal çözümü bulacaktır.")
    with st.sidebar:
        demand, workers, working_days, selected_demand_type = select_demand_type_and_workers("m6")
        holding_cost = st.number_input("Stok Maliyeti (TL)", 1, 100, int(model6.holding_cost), key="m6_holding")
        stockout_cost = st.number_input("Stoksuzluk Maliyeti (TL/adet)", 1, 500, int(model6.stockout_cost), key="m6_stockout")
        production_cost = st.number_input("Üretim Maliyeti (TL)", 1, 100, int(model6.production_cost), key="m6_prod_cost")
        labor_per_unit = st.number_input("Birim İşgücü (saat)", 0.1, 10.0, float(model6.labor_per_unit), key="m6_labor_per_unit")
        hourly_wage = st.number_input("Saatlik Ücret (TL)", 1, 100, int(model6.hourly_wage), key="m6_hourly_wage")
        daily_hours = st.number_input("Günlük Çalışma Saati", 1, 24, 8, key="m6_daily_hours")
        hiring_cost = st.number_input("İşçi Alım Maliyeti (TL)", 0, 5000, int(model6.hiring_cost), key="m6_hire")
        firing_cost = st.number_input("İşçi Çıkarma Maliyeti (TL)", 0, 5000, int(model6.firing_cost), key="m6_fire")
        max_workers = st.number_input("Maksimum İşçi", 1, 100, int(model6.max_workers), key="m6_max_workers")
        max_workforce_change = st.number_input("Aylık İşgücü Değişim Sınırı", 1, 100, int(model6.max_workforce_change), key="m6_max_workforce")
        initial_inventory = 0
        safety_stock_ratio = st.slider("Güvenlik Stoku Oranı (%)", min_value=0, max_value=100, value=5,
                                       key="m6_safety_stock_ratio") / 100
        run_model = st.button("Modeli Çalıştır", key="m6_run")
    if run_model or st.session_state.get("m6_first_run", True):
        st.session_state["m6_first_run"] = False
        df, total_cost, needed_workers, max_production = model6_run(
            demand, working_days, holding_cost, stockout_cost, production_cost, labor_per_unit, hourly_wage, daily_hours,
            hiring_cost, firing_cost, workers, max_workers, max_workforce_change,initial_inventory=initial_inventory, safety_stock_ratio=safety_stock_ratio, fixed_workers=workers
        )
        # Show different info based on whether fixed workers are used
        current_workers = int(df['İşçi'].mean())
        if workers != needed_workers:
            st.info(f"Maksimum Üretim Kapasitesi: {max_production} adet/ay | Kullanılan İşçi Sayısı: {current_workers} | Gerçek Optimum İşçi Sayısı: {needed_workers}")
            if current_workers > needed_workers:
                st.warning(f"⚠️ Şu anda {current_workers - needed_workers} fazla işçi kullanıyorsunuz.")
            else:
                st.warning(f"⚠️ Optimum için {needed_workers - current_workers} daha fazla işçi gerekli.")
        else:
            st.success(f"✅ Maksimum Üretim Kapasitesi: {max_production} adet/ay | Optimum İşçi Sayısı Kullanılıyor: {current_workers}")
        st.subheader("Sonuç Tablosu")
        cols = ["Ay"] + [c for c in df.columns if c != "Ay"]
        number_cols = df[cols].select_dtypes(include=["number"]).columns
        st.dataframe(
            df[cols].style.format({col: "{:,.0f}".format for col in number_cols}, thousands=".", decimal=","),
            use_container_width=True, hide_index=True
        )

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
        plt.close(fig)

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
    
    # Memory monitoring widget (opsiyonel)
    memory_placeholder = st.empty()
    
    with st.sidebar:
        demand, workers, working_days, selected_demand_type = select_demand_type_and_workers("cmp")
        holding_cost = st.number_input("Stok Maliyeti (TL)", min_value=1, max_value=100, value=int(get_param('costs', 'holding_cost', 5)), step=1, key="cmp_holding")
        stockout_cost = st.number_input("Karşılanmayan Talep Maliyeti (TL/adet)", min_value=1, max_value=500, value=int(get_param('costs', 'stockout_cost', 80)), step=1, key="cmp_stockout")
        production_cost = st.number_input("Birim Üretim Maliyeti (TL)", min_value=1, max_value=1000, value=int(get_param('costs', 'production_cost', 30)), step=1, key="cmp_production_cost")
        hiring_cost = st.number_input("İşçi Alım Maliyeti (TL)", min_value=0, max_value=5000, value=int(get_param('costs', 'hiring_cost', 1800)), step=1, key="cmp_hire")
        firing_cost = st.number_input("İşçi Çıkarma Maliyeti (TL)", min_value=0, max_value=5000, value=int(get_param('costs', 'firing_cost', 1500)), step=1, key="cmp_fire")
        hourly_wage = st.number_input("Saatlik Ücret (TL)", min_value=1, max_value=1000, value=int(get_param('costs', 'hourly_wage', 10)), step=1, key="cmp_hourly_wage")
        daily_hours = st.number_input("Günlük Çalışma Saati", min_value=1, max_value=24, value=int(get_param('workforce', 'daily_hours', 8)), step=1, key="cmp_daily_hours")
        labor_per_unit = st.number_input("Birim İşgücü (saat)", min_value=0.1, max_value=10.0, value=float(get_param('workforce', 'labor_per_unit', 4)), step=0.1, key="cmp_labor")
        max_overtime_per_worker = st.number_input("Maks. Fazla Mesai (saat/işçi)", min_value=0, max_value=100, value=int(get_param('costs', 'max_overtime_per_worker', 20)), step=1, key="cmp_max_overtime")
        overtime_wage_multiplier = st.number_input("Fazla Mesai Ücret Çarpanı", min_value=1.0, max_value=5.0, value=float(get_param('costs', 'overtime_wage_multiplier', 1.5)), step=0.1, key="cmp_overtime_multiplier")
        initial_inventory = 0
        safety_stock_ratio = st.slider("Güvenlik Stoku Oranı (%)", min_value=0, max_value=100, value=5, key="cmp_safety_stock_ratio") / 100

        # Enhanced memory management controls
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Cache Temizle", key="clear_cache_btn"):
                clear_memory()
                st.success("Cache temizlendi!")
        
        with col2:
            show_memory = st.checkbox("Bellek İzleme", key="show_memory")

        compare_btn = st.button("Modelleri Karşılaştır", key="cmp_compare_btn")

    # Run comparison on first load or when button pressed
    if compare_btn or st.session_state.get("cmp_first_run", True):
        st.session_state["cmp_first_run"] = False

        # Aggressive memory cleanup before starting
        clear_memory()
        
        # Monitor memory if requested
        initial_memory = monitor_memory() if show_memory else None
        if initial_memory and show_memory:
            memory_placeholder.info(f"Başlangıç bellek kullanımı: {initial_memory:.1f} MB")

        # Collect current parameters - use locals to minimize memory
        current_params = {
            "demand": demand,
            "holding_cost": holding_cost,
            "stockout_cost": stockout_cost,
            "production_cost": production_cost,
            "hiring_cost": hiring_cost,
            "firing_cost": firing_cost,
            "hourly_wage": hourly_wage,
            "daily_hours": daily_hours,
            "labor_per_unit": labor_per_unit,
            "max_overtime_per_worker": max_overtime_per_worker,
            "overtime_wage_multiplier": overtime_wage_multiplier,
            "working_days": working_days,
            "workers": workers,
            "initial_inventory": initial_inventory,
            "safety_stock_ratio": safety_stock_ratio
        }

        # Optimized model definitions - reduced memory footprint
        model_configs = [
            ("Model 1", "Karma Planlama", model1.maliyet_analizi, "Yüksek", "Karma planlama"),
            ("Model 2", "Fazla Mesaili Üretim", model2.maliyet_analizi, "Orta", "Fazla mesai"),
            ("Model 3", "Toplu Üretim", model3.maliyet_analizi, "Düşük", "Toplu üretim"),
            ("Model 4", "Dinamik Programlama", model4.maliyet_analizi, "Orta", "Dinamik işgücü"),
            ("Model 5", "Dış Kaynak", model5.maliyet_analizi, "Yok", "Fason kullanımı"),
            ("Model 6", "Mevsimsellik", model6.maliyet_analizi, "Orta", "Mevsimsel optimizasyon"),
        ]
        
        # Use generator for memory efficiency
        def process_model(model_config):
            """Process a single model and return results"""
            short_name, display_name, func, flex, scenario = model_config
            
            try:
                # Aggressive cleanup before each model
                safe_memory_cleanup()
                
                # Get function signature
                sig_params = inspect.signature(func).parameters
                params_to_pass = {}

                # Build parameters efficiently
                for p_name, p_value in current_params.items():
                    if short_name == "Model 2" and p_name == "hourly_wage":
                        if "normal_hourly_wage" in sig_params:
                            params_to_pass["normal_hourly_wage"] = p_value
                    elif p_name in sig_params:
                        params_to_pass[p_name] = p_value

                # Add specific parameters for Model 3
                if short_name == "Model 3" and "hiring_cost" in sig_params:
                    params_to_pass["hiring_cost"] = current_params.get("hiring_cost")

                # Run the model function
                with memory_context():
                    res = func(**params_to_pass)
                
                # Extract only essential values
                result = {
                    'cost': res.get("Toplam Maliyet", 0),
                    'labor_cost': res.get("İşçilik Maliyeti", 0),
                    'total_prod': res.get("Toplam Üretim", 0),
                    'total_demand': res.get("Toplam Talep", 1),
                    'stockout': res.get("Karşılanmayan Talep", 0),
                    'flex': flex,
                    'scenario': scenario
                }
                
                # Calculate stockout rate
                result['stockout_rate'] = (result['stockout'] / result['total_demand'] * 100) if result['total_demand'] > 0 else 0
                
                # Cleanup immediately
                del res, params_to_pass
                
                return result
                
            except Exception as e:
                # Return error result
                return {
                    'cost': 0, 'labor_cost': 0, 'total_prod': 0, 'total_demand': 1,
                    'stockout': 0, 'stockout_rate': 0, 'flex': flex, 'scenario': f"Hata: {str(e)}"
                }

        # Process models with progress tracking
        summary_results = []
        progress_bar = st.progress(0)
        
        for idx, model_config in enumerate(model_configs, 1):
            progress_bar.progress(idx / len(model_configs))
            
            # Process single model
            result = process_model(model_config)
            summary_results.append([
                result['cost'], result['labor_cost'], result['total_prod'], 
                result['stockout_rate'], result['flex'], result['scenario']
            ])
            
            # Memory monitoring
            if show_memory and idx % 2 == 0:  # Check every 2 models
                current_memory = monitor_memory()
                if current_memory:
                    memory_placeholder.info(f"Model {idx} tamamlandı. Bellek: {current_memory:.1f} MB")
            
            # Small delay for memory cleanup
            time.sleep(0.01)

        # Clear progress bar
        progress_bar.empty()
        
        # Final comprehensive memory cleanup
        clear_memory()
        
        # Show final memory usage
        if show_memory:
            final_memory = monitor_memory()
            if initial_memory and final_memory:
                memory_placeholder.success(f"Tamamlandı! Bellek: {final_memory:.1f} MB (Başlangıç: {initial_memory:.1f} MB)")

        # Create summary dataframe efficiently
        summary_df = pd.DataFrame(
            summary_results,
            columns=["Toplam Maliyet (₺)", "Toplam İşçilik Maliyeti (₺)", "Toplam Üretim", "Stoksuzluk Oranı (%)", "İşgücü Esnekliği", "Uygun Senaryo"],
            index=[config[0] for config in model_configs]
        )
        
        # Clean up intermediate data
        del summary_results, model_configs
        
        # Replace NaN values
        summary_df = summary_df.fillna(0)
        
        st.subheader("Özet Karşılaştırma Tablosu")
        
        # Optimized formatting
        try:
            formatted_df = summary_df.style.format({
                "Toplam Maliyet (₺)": "{:,.0f}",
                "Toplam İşçilik Maliyeti (₺)": "{:,.0f}",
                "Toplam Üretim": "{:,.0f}",
                "Stoksuzluk Oranı (%)": "{:.1f}",
            }, na_rep="0")
            
            st.dataframe(formatted_df, use_container_width=True)
        except Exception:
            st.dataframe(summary_df, use_container_width=True)

        # Create optimized visualization
        st.subheader("Grafiksel Karşılaştırma")
        
        try:
            # Create plots with memory optimization
            with memory_context():
                fig, axes = plt.subplots(2, 2, figsize=(10, 8))
                axes = axes.flatten()
                
                # Colors
                colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
                
                # Plot data
                plot_data = {
                    'Toplam Maliyet (₺)': summary_df['Toplam Maliyet (₺)'],
                    'Toplam Üretim': summary_df['Toplam Üretim'],
                    'Stoksuzluk Oranı (%)': summary_df['Stoksuzluk Oranı (%)'],
                    'Ortalama Birim Maliyet': summary_df['Toplam Maliyet (₺)'] / summary_df['Toplam Üretim'].replace(0, 1)
                }
                
                for i, (metric, data) in enumerate(plot_data.items()):
                    ax = axes[i]
                    bars = ax.bar(summary_df.index, data, color=colors[:len(summary_df.index)], alpha=0.8)
                    
                    # Add value labels
                    for bar in bars:
                        height = bar.get_height()
                        if pd.notnull(height) and height > 0:
                            if metric in ["Toplam Maliyet (₺)", "Toplam Üretim"]:
                                text = f"{int(height):,}"
                            else:
                                text = f"{height:.1f}"
                            ax.text(bar.get_x() + bar.get_width()/2., height * 1.02,
                                   text, ha='center', va='bottom', fontsize=8)
                    
                    ax.set_title(metric, fontsize=10)
                    ax.set_xticks(range(len(summary_df.index)))
                    ax.set_xticklabels(summary_df.index, rotation=45, ha='right', fontsize=8)
                    ax.grid(axis='y', alpha=0.3)
                
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
                
                # Clear plot data
                del plot_data, bars
                
        except Exception as e:
            st.error(f"Grafik oluşturulurken hata: {str(e)}")
            
        # Final cleanup
        del summary_df
        
        st.markdown("---")

        # Detaylı Karşılaştırma Tablosu - Restored with memory optimization
        st.subheader("Detaylı Karşılaştırma Tablosu")
        
        with st.expander("Tüm Modellerin Detaylı Sonuçları", expanded=False):
            st.info("Bu tablo tüm modellerin detaylı maliyet analizlerini gösterir. Büyük veri seti için yavaş olabilir.")
            
            # Recreate model configs for detailed analysis
            detail_model_configs = [
                ("Model 1", "Karma Planlama", model1.maliyet_analizi, "Yüksek", "Karma planlama"),
                ("Model 2", "Fazla Mesaili Üretim", model2.maliyet_analizi, "Orta", "Fazla mesai"),
                ("Model 3", "Toplu Üretim", model3.maliyet_analizi, "Düşük", "Toplu üretim"),
                ("Model 4", "Dinamik Programlama", model4.maliyet_analizi, "Orta", "Dinamik işgücü"),
                ("Model 5", "Dış Kaynak", model5.maliyet_analizi, "Yok", "Fason kullanımı"),
                ("Model 6", "Mevsimsellik", model6.maliyet_analizi, "Orta", "Mevsimsel optimizasyon"),
            ]
            
            # Create detailed results with memory management
            detail_results = []
            detail_progress = st.progress(0)
            detail_status = st.empty()
            
            for idx, model_config in enumerate(detail_model_configs, 1):
                short_name, display_name, func, flex, scenario = model_config
                detail_progress.progress(idx / len(detail_model_configs))
                detail_status.text(f"İşleniyor: {display_name}...")
                
                try:
                    with memory_context():
                        # Get function signature
                        sig_params = inspect.signature(func).parameters
                        params_to_pass = {}
                        
                        # Build parameters for each model
                        for p_name, p_value in current_params.items():
                            if short_name == "Model 2" and p_name == "hourly_wage":
                                if "normal_hourly_wage" in sig_params:
                                    params_to_pass["normal_hourly_wage"] = p_value
                            elif p_name in sig_params:
                                params_to_pass[p_name] = p_value
                        
                        # Special handling for Model 3
                        if short_name == "Model 3" and "hiring_cost" in sig_params:
                            params_to_pass["hiring_cost"] = current_params.get("hiring_cost")
                        
                        # Run the model analysis function
                        result = func(**params_to_pass)
                        
                        # Add model name to result
                        result["Model"] = display_name
                        result["Esneklik"] = flex
                        result["Senaryo"] = scenario
                        
                        # Only keep essential columns for display
                        essential_result = {
                            "Model": result.get("Model", display_name),
                            "Toplam Maliyet": result.get("Toplam Maliyet", 0),
                            "İşçilik Maliyeti": result.get("İşçilik Maliyeti", 0),
                            "Üretim Maliyeti": result.get("Üretim Maliyeti", 0),
                            "Stok Maliyeti": result.get("Stok Maliyeti", 0),
                            "Stoksuzluk Maliyeti": result.get("Stoksuzluk Maliyeti", 0),
                            "Toplam Talep": result.get("Toplam Talep", 0),
                            "Toplam Üretim": result.get("Toplam Üretim", 0),
                            "Karşılanmayan Talep": result.get("Karşılanmayan Talep", 0),
                            "Ortalama Birim Maliyet": result.get("Ortalama Birim Maliyet", 0),
                            "İşgücü Esnekliği": flex,
                            "Uygun Senaryo": scenario
                        }
                        
                        detail_results.append(essential_result)
                        
                        # Clear large result object immediately
                        del result, params_to_pass
                
                except Exception as e:
                    st.warning(f"{display_name} hesaplanırken hata: {str(e)}")
                    # Add error placeholder
                    detail_results.append({
                        "Model": display_name,
                        "Toplam Maliyet": 0,
                        "İşçilik Maliyeti": 0,
                        "Üretim Maliyeti": 0,
                        "Stok Maliyeti": 0,
                        "Stoksuzluk Maliyeti": 0,
                        "Toplam Talep": 0,
                        "Toplam Üretim": 0,
                        "Karşılanmayan Talep": 0,
                        "Ortalama Birim Maliyet": 0,
                        "İşgücü Esnekliği": flex,
                        "Uygun Senaryo": f"Hata: {str(e)}"
                    })
                
                # Force cleanup after each model
                safe_memory_cleanup()
            
            # Clear progress indicators
            detail_progress.empty()
            detail_status.empty()
            
            # Create detailed DataFrame
            try:
                detail_df = pd.DataFrame(detail_results)
                
                # Fill NaN values
                detail_df = detail_df.fillna(0)
                
                # Reorder columns for better display
                column_order = [
                    "Model", "Toplam Maliyet", "İşçilik Maliyeti", "Üretim Maliyeti", 
                    "Stok Maliyeti", "Stoksuzluk Maliyeti", "Toplam Talep", "Toplam Üretim",
                    "Karşılanmayan Talep", "Ortalama Birim Maliyet", "İşgücü Esnekliği", "Uygun Senaryo"
                ]
                
                # Only use columns that exist in the dataframe
                available_columns = [col for col in column_order if col in detail_df.columns]
                detail_df = detail_df[available_columns]
                
                # Format numerical columns
                numerical_cols = detail_df.select_dtypes(include=['number']).columns
                
                try:
                    # Apply formatting with error handling
                    formatted_detail = detail_df.style.format({
                        col: "{:,.2f}" if col == "Ortalama Birim Maliyet" else "{:,.0f}" 
                        for col in numerical_cols
                    }, na_rep="0")
                    
                    st.dataframe(formatted_detail, use_container_width=True, hide_index=True)
                    
                except Exception:
                    # Fallback to simple display
                    st.dataframe(detail_df, use_container_width=True, hide_index=True)
                    st.warning("Tablo formatlamasında sorun oluştu, ham veriler gösteriliyor.")
                
                # Add summary statistics
                st.subheader("Karşılaştırma Özeti")
                
                try:
                    # Calculate summary statistics
                    valid_costs = detail_df[detail_df['Toplam Maliyet'] > 0]['Toplam Maliyet']
                    if len(valid_costs) > 0:
                        min_cost_idx = detail_df['Toplam Maliyet'].idxmin()
                        max_cost_idx = detail_df['Toplam Maliyet'].idxmax()
                        
                        min_cost_model = detail_df.loc[min_cost_idx, 'Model']
                        max_cost_model = detail_df.loc[max_cost_idx, 'Model']
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("En Düşük Maliyet", 
                                     f"{detail_df['Toplam Maliyet'].min():,.0f} TL",
                                     delta=f"{min_cost_model}")
                        with col2:
                            st.metric("En Yüksek Maliyet", 
                                     f"{detail_df['Toplam Maliyet'].max():,.0f} TL",
                                     delta=f"{max_cost_model}")
                        with col3:
                            avg_cost = detail_df[detail_df['Toplam Maliyet'] > 0]['Toplam Maliyet'].mean()
                            st.metric("Ortalama Maliyet", f"{avg_cost:,.0f} TL")
                            
                except Exception as e:
                    st.info("Özet istatistikler hesaplanamadı.")
                
                # Cleanup with error handling
                try:
                    if 'detail_df' in locals():
                        del detail_df
                    if 'detail_results' in locals():
                        del detail_results
                    if 'detail_model_configs' in locals():
                        del detail_model_configs
                except:
                    pass
                
            except Exception as e:
                st.error(f"Detaylı tablo oluşturulurken hata: {str(e)}")
                # Cleanup with error handling
                try:
                    if 'detail_results' in locals():
                        del detail_results
                    if 'detail_model_configs' in locals():
                        del detail_model_configs
                except:
                    pass
        
        # Model 1'e özel bilgi notu        
        st.info("**Not:** Model 1'de Üretim Birim Maliyeti, iç üretim ve fason üretimin ağırlıklı ortalamasıdır.")
        
        # Final cleanup
        try:
            del current_params
        except:
            pass
        clear_memory()


