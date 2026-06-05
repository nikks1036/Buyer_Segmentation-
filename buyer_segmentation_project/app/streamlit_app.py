import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import os
from datetime import datetime

# Page Configuration
st.set_page_config(
    page_title="Buyer Segmentation Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main { padding: 0rem 1rem; }
    h1 { color: #1f77b4; font-size: 2.5rem; margin-bottom: 0.5rem; }
    h2 { color: #1f77b4; font-size: 1.8rem; margin-top: 1.5rem; }
    .metric-card { 
        background-color: #f0f2f6; 
        padding: 1.5rem; 
        border-radius: 0.5rem; 
        border-left: 5px solid #1f77b4;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.title("🎯 Machine Learning Based Buyer Segmentation and Investment Profiling for Real Estate Market Intelligence")
st.markdown("Professional customer segmentation analysis using K-Means clustering")

try:
    # Get the absolute path to the CSV file
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(base_dir, "outputs", "segmented_clients.csv")
    
    if not os.path.exists(csv_path):
        st.error("❌ Data file not found!")
        st.warning(f"Expected path: {csv_path}")
        st.info("Please run the notebook first to generate segmented_clients.csv")
        st.stop()
    
    df = pd.read_csv(csv_path)
    st.success("✅ Data Loaded Successfully")
    
    # Sidebar Filters
    st.sidebar.header("🔍 Filters")
    selected_clusters = st.sidebar.multiselect(
        "Select Clusters",
        options=sorted(df['cluster'].unique()),
        default=sorted(df['cluster'].unique())
    )
    
    # Filter data
    filtered_df = df[df['cluster'].isin(selected_clusters)]
    
    # Key Metrics
    st.sidebar.markdown("---")
    st.sidebar.header("📊 Key Metrics")
    col1, col2, col3 = st.sidebar.columns(3)
    with col1:
        st.metric("Total Clients", f"{len(filtered_df):,}")
    with col2:
        st.metric("Avg Satisfaction", f"{filtered_df['satisfaction_score'].mean():.1f}/5")
    with col3:
        st.metric("Total Revenue", f"${filtered_df['total_spent'].sum()/1e6:.1f}M")
    
    # Main Content
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Overview", "👥 Segment Analysis", "📋 Customer Data", "📊 Advanced Analytics"])
    
    # Tab 1: Overview
    with tab1:
        st.subheader("Cluster Distribution & Key Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Cluster distribution histogram
            fig_dist = px.histogram(
                filtered_df, 
                x="cluster",
                title="Distribution of Customers Across Clusters",
                nbins=4,
                color='cluster',
                color_discrete_sequence=px.colors.qualitative.Plotly,
                labels={'cluster': 'Cluster ID', 'count': 'Number of Customers'}
            )
            fig_dist.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig_dist, use_container_width=True)
        
        with col2:
            # Satisfaction by cluster
            fig_sat = px.box(
                filtered_df,
                x='cluster',
                y='satisfaction_score',
                title="Satisfaction Score Distribution by Cluster",
                color='cluster',
                color_discrete_sequence=px.colors.qualitative.Plotly
            )
            fig_sat.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig_sat, use_container_width=True)
        
        # Spending Analysis
        col1, col2 = st.columns(2)
        
        with col1:
            fig_spend = px.box(
                filtered_df,
                x='cluster',
                y='total_spent',
                title="Total Spending by Cluster",
                color='cluster',
                color_discrete_sequence=px.colors.qualitative.Plotly
            )
            fig_spend.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig_spend, use_container_width=True)
        
        with col2:
            fig_age = px.violin(
                filtered_df,
                x='cluster',
                y='age',
                title="Age Distribution by Cluster",
                color='cluster',
                color_discrete_sequence=px.colors.qualitative.Plotly
            )
            fig_age.update_layout(showlegend=False, height=400)
            st.plotly_chart(fig_age, use_container_width=True)
    
    # Tab 2: Segment Analysis
    with tab2:
        st.subheader("Detailed Segment Profiles")
        
        # Cluster statistics
        cluster_stats = filtered_df.groupby('cluster').agg({
            'age': ['mean', 'median'],
            'satisfaction_score': ['mean', 'median'],
            'property_count': ['mean', 'sum'],
            'total_spent': ['mean', 'sum'],
            'avg_price': 'mean',
            'client_type': lambda x: (x == 'Company').sum(),
            'client_id': 'count'
        }).round(2)
        
        cluster_stats.columns = ['Avg Age', 'Median Age', 'Avg Satisfaction', 'Median Satisfaction',
                                  'Avg Properties', 'Total Properties', 'Avg Spent', 'Total Spent',
                                  'Avg Price', 'Companies', 'Total Clients']
        
        st.dataframe(cluster_stats, use_container_width=True)
        
        # Segment Characteristics
        st.markdown("### Segment Characteristics")
        
        for cluster_id in sorted(filtered_df['cluster'].unique()):
            cluster_data = filtered_df[filtered_df['cluster'] == cluster_id]
            
            with st.expander(f"📍 Cluster {cluster_id} - {len(cluster_data)} Customers"):
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Avg Age", f"{cluster_data['age'].mean():.1f} years")
                with col2:
                    st.metric("Avg Satisfaction", f"{cluster_data['satisfaction_score'].mean():.1f}/5")
                with col3:
                    st.metric("Avg Properties", f"{cluster_data['property_count'].mean():.1f}")
                with col4:
                    st.metric("Total Spent", f"${cluster_data['total_spent'].sum()/1e6:.2f}M")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Company Buyers:** {(cluster_data['client_type'] == 'Company').sum()} ({(cluster_data['client_type'] == 'Company').sum()/len(cluster_data)*100:.1f}%)")
                    st.write(f"**Individual Buyers:** {(cluster_data['client_type'] == 'Individual').sum()} ({(cluster_data['client_type'] == 'Individual').sum()/len(cluster_data)*100:.1f}%)")
                with col2:
                    st.write(f"**Avg Price per Property:** ${cluster_data['avg_price'].mean():,.0f}")
                    st.write(f"**Max Price:** ${cluster_data['avg_price'].max():,.0f}")
                    st.write(f"**Min Price:** ${cluster_data['avg_price'].min():,.0f}")
    
    # Tab 3: Customer Data
    with tab3:
        st.subheader("Customer Data Explorer")
        
        # Search functionality
        search_col = st.columns(3)
        with search_col[0]:
            search_client = st.text_input("Search by Client ID", "")
        with search_col[1]:
            min_spent = st.number_input("Min Spent ($)", value=0, step=100000)
        with search_col[2]:
            max_spent = st.number_input("Max Spent ($)", value=int(filtered_df['total_spent'].max()), step=100000)
        
        # Apply search filters
        display_df = filtered_df.copy()
        if search_client:
            display_df = display_df[display_df['client_id'].str.contains(search_client, case=False)]
        display_df = display_df[(display_df['total_spent'] >= min_spent) & (display_df['total_spent'] <= max_spent)]
        
        st.write(f"Showing {len(display_df)} customers")
        st.dataframe(display_df, use_container_width=True, height=500)
        
        # Download button
        csv = display_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Filtered Data (CSV)",
            data=csv,
            file_name=f"customer_segments_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    # Tab 4: Advanced Analytics
    with tab4:
        st.subheader("Advanced Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Scatter plot: Spending vs Satisfaction
            fig_scatter = px.scatter(
                filtered_df,
                x='satisfaction_score',
                y='total_spent',
                color='cluster',
                size='property_count',
                hover_name='client_id',
                title="Spending vs Satisfaction (bubble size = properties)",
                color_discrete_sequence=px.colors.qualitative.Plotly,
                labels={'satisfaction_score': 'Satisfaction Score', 'total_spent': 'Total Spent ($)'}
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
        
        with col2:
            # Bar chart: Average metrics by cluster
            metrics_df = filtered_df.groupby('cluster').agg({
                'satisfaction_score': 'mean',
                'age': 'mean',
                'property_count': 'mean'
            }).reset_index()
            
            fig_metrics = px.bar(
                metrics_df,
                x='cluster',
                y=['satisfaction_score', 'age'],
                barmode='group',
                title="Cluster Comparison: Satisfaction & Age",
                color_discrete_sequence=['#1f77b4', '#ff7f0e']
            )
            st.plotly_chart(fig_metrics, use_container_width=True)
        
        # Correlation heatmap
        st.markdown("### Correlation Analysis")
        numeric_cols = ['age', 'satisfaction_score', 'property_count', 'total_spent', 'avg_price']
        corr_matrix = filtered_df[numeric_cols].corr()
        
        fig_corr = px.imshow(
            corr_matrix,
            labels=dict(x="Features", y="Features", color="Correlation"),
            title="Feature Correlation Matrix",
            color_continuous_scale='RdBu_r',
            zmin=-1, zmax=1
        )
        st.plotly_chart(fig_corr, use_container_width=True)
    
    st.markdown("---")
    st.markdown(f"<p style='text-align: center; color: gray;'>Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>", unsafe_allow_html=True)
    
except FileNotFoundError:
    st.error("❌ Data file not found!")
    st.info("Please ensure the notebook has been executed and segmented_clients.csv exists.")
except Exception as e:
    st.error(f"❌ Error: {str(e)}")
    import traceback
    st.error(traceback.format_exc())
