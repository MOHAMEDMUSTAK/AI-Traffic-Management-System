import streamlit as st
import numpy as np

# --- Signal Timing Logic (Simple Rule-Based Model) ---
def calculate_signal_time(car_count, bike_count, total_pressure, W_C, W_B, MIN_TIME, MAX_TIME):
    """Calculates green light time based on weighted vehicle counts."""
    road_pressure = (car_count * W_C) + (bike_count * W_B)
    
    if total_pressure == 0:
        # Default to even split if no pressure
        return round((MIN_TIME + MAX_TIME) / 2) 

    # Scale green time proportionally between MIN and MAX
    normalized_pressure = road_pressure / total_pressure
    green_time = MIN_TIME + (MAX_TIME - MIN_TIME) * normalized_pressure
    
    return round(green_time)

# --- Configuration ---
st.set_page_config(layout="wide", page_title="Traffic Command Dashboard")

st.title("🛰️ Traffic Command Center Dashboard")
st.markdown("This dashboard displays the **real-time adaptive decisions** made by the AI based on traffic pressure.")
st.markdown("---")

# ==========================================================
# 1. INPUTS & ADAPTIVE PARAMETERS
# ==========================================================

col_logic, col_inputs = st.columns([1, 2.5])

with col_logic:
    st.header("⚙️ Adaptive Logic Settings")
    W_C = st.slider("Car Weight ($W_C$)", 0.5, 2.0, 1.0, 0.1, key='W_C')
    W_B = st.slider("Bike Weight ($W_B$)", 0.1, 1.0, 0.5, 0.1, key='W_B')
    MIN_TIME = st.slider("Min Green (s)", 5, 20, 15, key='MIN_TIME')
    MAX_TIME = st.slider("Max Green (s)", 30, 90, 60, key='MAX_TIME')
    YELLOW_TIME = 4
    st.metric("Fixed Yellow Time", f"{YELLOW_TIME}s")

# --- Traffic Load Inputs (Visual Steppers) ---
roads = ["North", "South", "East", "West"]
road_data = {}
with col_inputs:
    st.header("1. 🚗 Traffic Load Input (Vertical Steppers)")
    st.markdown("Use the controls below to simulate traffic queues for each road.")
    
    # Use 4 columns for the roads
    road_cols = st.columns(4)
    
    for i, road in enumerate(roads):
        with road_cols[i]:
            st.markdown(f"**--- {road} Approach ---**")
            
            # --- Car Input (SLIDER IS NOW VERTICAL) ---
            st.markdown("### 🚗 Car Queue") 
            cars = st.slider("Cars", 0, 100, 20, key=f'{road}_cars', label_visibility="collapsed")
                
            # --- Bike Input (SLIDER IS NOW VERTICAL) ---
            st.markdown("### 🏍️ Bike Queue") 
            bikes = st.slider("Bikes", 0, 50, 10, key=f'{road}_bikes', label_visibility="collapsed")
            
            # Display current count below the sliders
            st.markdown(f"Size: **{cars} Cars / {bikes} Bikes**")
            
            road_data[road] = {'cars': cars, 'bikes': bikes}

        
# --- Calculation ---
total_weighted_pressure = sum(
    (data['cars'] * W_C) + (data['bikes'] * W_B) 
    for data in road_data.values()
)

green_times = {}
for road in roads:
    cars = road_data[road]['cars']
    bikes = road_data[road]['bikes']
    green_times[road] = calculate_signal_time(
        cars, bikes, total_weighted_pressure, W_C, W_B, MIN_TIME, MAX_TIME
    )

# Phase Times
ns_time = max(green_times['North'], green_times['South'])
ew_time = max(green_times['East'], green_times['West'])
total_green_time = ns_time + ew_time
total_cycle_time = total_green_time + (2 * YELLOW_TIME)

# Pressures and Ratios for Visualization
ns_pressure = (road_data['North']['cars'] * W_C + road_data['North']['bikes'] * W_B) + \
              (road_data['South']['cars'] * W_C + road_data['South']['bikes'] * W_B)
ew_pressure = (road_data['East']['cars'] * W_C + road_data['East']['bikes'] * W_B) + \
              (road_data['West']['cars'] * W_C + road_data['West']['bikes'] * W_B)
total_pressure_vis = ns_pressure + ew_pressure

ns_percentage = (ns_time / total_green_time) * 100 if total_green_time > 0 else 50
ew_percentage = (ew_time / total_green_time) * 100 if total_green_time > 0 else 50


# ==========================================================
# 2. TRAFFIC ALLOCATION RING (INNOVATIVE VISUALIZATION)
# ==========================================================

st.markdown("---")
st.header("2. 💡 Adaptive Allocation Dashboard")

def draw_adaptive_ring(ns_percent, ew_percent):
    """Generates a dynamic HTML/CSS dual-color ring visualization."""
    
    # Gradient style for the ring (NS is Green, EW is Blue)
    ring_css = f"""
        <div style='
            width: 250px;
            height: 250px;
            border-radius: 50%;
            background: conic-gradient(
                #4CAF50 0% {ns_percent}%, 
                #2196F3 {ns_percent}% 100%
            );
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0 auto;
        '>
            <div style='
                width: 200px;
                height: 200px;
                border-radius: 50%;
                background: #1a1a1a;
                text-align: center;
                padding-top: 50px;
                color: white;
            '>
                <h3 style='margin: 0;'>{round(ns_percent)}% / {round(ew_percent)}%</h3>
                <p style='margin: 5px;'>ALLOCATED</p>
            </div>
        </div>
    """
    st.components.v1.html(ring_css, height=300)

col_kpis, col_ring, col_road_metrics = st.columns([1, 1, 1.5])

with col_kpis:
    st.subheader("Key Performance Indicators (KPIs)")
    st.metric("Total Cycle Time", f"**{total_cycle_time}s**", delta=f"+{2*YELLOW_TIME}s yellow time")
    st.metric("Total Green Time", f"**{total_green_time}s**", delta=f"{total_green_time/MAX_TIME:.0%} of Max Green")
    st.metric("Total System Pressure", f"**{round(total_pressure_vis, 1)}** units")

with col_ring:
    st.subheader("Phase Time Distribution")
    draw_adaptive_ring(ns_percentage, ew_percentage)
    
    st.markdown(f"""
        <div style="text-align:center;">
            <span style="color:#4CAF50;">■</span> N/S Phase: **{ns_time}s**
            <span style="color:#2196F3;">■</span> E/W Phase: **{ew_time}s**
        </div>
    """, unsafe_allow_html=True)

with col_road_metrics:
    st.subheader("Individual Road Allocations")
    road_metrics = {}
    for road in roads:
        pressure = (road_data[road]['cars'] * W_C) + (road_data[road]['bikes'] * W_B)
        road_metrics[road] = (f"{green_times[road]}s", f"Pressure: {round(pressure, 1)}")
    
    m_cols = st.columns(2)
    m_cols[0].metric(f"North Green Time", road_metrics['North'][0], delta=road_metrics['North'][1])
    m_cols[1].metric(f"South Green Time", road_metrics['South'][0], delta=road_metrics['South'][1])
    m_cols[0].metric(f"East Green Time", road_metrics['East'][0], delta=road_metrics['East'][1])
    m_cols[1].metric(f"West Green Time", road_metrics['West'][0], delta=road_metrics['West'][1])