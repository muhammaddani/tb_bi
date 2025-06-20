import dash
from dash import dcc, html, Input, Output, callback_context
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import sqlalchemy
from sqlalchemy import create_engine, text
import numpy as np
from datetime import datetime, timedelta
import os

# ML Libraries
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# Connection string
DATABASE_URL = f"postgresql://postgres.icrhhezhmudmobtztsat:Dani159901@aws-0-ap-southeast-1.pooler.supabase.com:6543/postgres"

class HotelDashboard:
    def __init__(self):
        self.engine = create_engine(DATABASE_URL)
        self.app = dash.Dash(__name__, suppress_callback_exceptions=True)
        # ML Models
        self.cancellation_model = None
        self.adr_model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.setup_layout()
        self.setup_callbacks()
        
    def get_data(self, filters=None):
        """Mengambil data dari database dengan filter opsional"""
        query = """
        SELECT 
            fb.*,
            dd_arrival.year as arrival_year,
            dd_arrival.month as arrival_month,
            dd_arrival.month_name as arrival_month_name,
            dd_arrival.season,
            dd_status.year as status_year,
            dd_status.month as status_month,
            dh.hotel,
            dms.market_segment,
            ddc.distribution_channel,
            dm.meal,
            ddt.deposit_type,
            dct.customer_type,
            dc.country,
            drs.reservation_status
        FROM fact_booking fb
        LEFT JOIN dim_date dd_arrival ON fb.arrival_date_key = dd_arrival.date_key
        LEFT JOIN dim_date dd_status ON fb.status_date_key = dd_status.date_key
        LEFT JOIN dim_hotel dh ON fb.hotel_key = dh.hotel_key
        LEFT JOIN dim_market_segment dms ON fb.market_segment_key = dms.market_segment_key
        LEFT JOIN dim_distribution_channel ddc ON fb.dist_channel_key = ddc.dist_channel_key
        LEFT JOIN dim_meal dm ON fb.meal_key = dm.meal_key
        LEFT JOIN dim_deposit_type ddt ON fb.deposit_type_key = ddt.deposit_type_key
        LEFT JOIN dim_customer_type dct ON fb.cust_type_key = dct.cust_type_key
        LEFT JOIN dim_country dc ON fb.country_key = dc.country_key
        LEFT JOIN dim_reservation_status drs ON fb.res_status_key = drs.res_status_key
        """
        
        where_conditions = []
        
        if filters:
            if 'hotel' in filters and filters['hotel']:
                hotels = "','".join(filters['hotel'])
                where_conditions.append(f"dh.hotel IN ('{hotels}')")
            if 'market_segment' in filters and filters['market_segment']:
                segments = "','".join(filters['market_segment'])
                where_conditions.append(f"dms.market_segment IN ('{segments}')")
            if 'country' in filters and filters['country']:
                countries = "','".join(filters['country'])
                where_conditions.append(f"dc.country IN ('{countries}')")
            if 'season' in filters and filters['season']:
                seasons = "','".join(filters['season'])
                where_conditions.append(f"dd_arrival.season IN ('{seasons}')")
        
        if where_conditions:
            query += " WHERE " + " AND ".join(where_conditions)
        
        if filters and 'period' in filters and filters['period']:
            yr, mon = filters['period'][0].split('-', 1)
            cond = f"dd_arrival.year = {int(yr)} AND dd_arrival.month_name = '{mon}'"
            query += (" AND " if where_conditions else " WHERE ") + cond
            
        return pd.read_sql(query, self.engine)

    def prepare_ml_data(self, data):
        """Mempersiapkan data untuk machine learning"""
        ml_data = data.copy()
        
        # Encode categorical variables
        categorical_cols = ['hotel', 'market_segment', 'distribution_channel', 'meal', 
                           'deposit_type', 'customer_type', 'country', 'season']
        
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                ml_data[f'{col}_encoded'] = self.label_encoders[col].fit_transform(ml_data[col].astype(str))
            else:
                ml_data[f'{col}_encoded'] = self.label_encoders[col].transform(ml_data[col].astype(str))
        
        return ml_data

    def train_cancellation_model(self, data):
        """Melatih model prediksi pembatalan"""
        ml_data = self.prepare_ml_data(data)
        
        # Features untuk prediksi pembatalan
        feature_cols = ['lead_time', 'total_stay_nights', 'adults', 'children', 'babies',
                       'previous_cancellations', 'booking_changes', 'total_of_special_requests',
                       'hotel_encoded', 'market_segment_encoded', 'deposit_type_encoded',
                       'customer_type_encoded', 'season_encoded']
        
        X = ml_data[feature_cols].fillna(0)
        y = ml_data['is_canceled']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.cancellation_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.cancellation_model.fit(X_train, y_train)
        
        # Evaluasi model
        y_pred = self.cancellation_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': self.cancellation_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return accuracy, feature_importance

    def train_adr_model(self, data):
        """Melatih model prediksi ADR"""
        ml_data = self.prepare_ml_data(data)
        
        # Filter data dengan ADR > 0
        ml_data = ml_data[ml_data['adr'] > 0]
        
        feature_cols = ['lead_time', 'total_stay_nights', 'adults', 'children', 'babies',
                       'booking_changes', 'total_of_special_requests', 'is_canceled',
                       'hotel_encoded', 'market_segment_encoded', 'meal_encoded',
                       'customer_type_encoded', 'season_encoded']
        
        X = ml_data[feature_cols].fillna(0)
        y = ml_data['adr']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        self.adr_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.adr_model.fit(X_train, y_train)
        
        # Evaluasi model
        y_pred = self.adr_model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': self.adr_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return mse, r2, feature_importance

    def customer_segmentation(self, data):
        """Melakukan customer segmentation dengan K-Means"""
        ml_data = self.prepare_ml_data(data)
        
        # Features untuk segmentasi
        segment_features = ['lead_time', 'total_stay_nights', 'adr', 'total_guests',
                           'booking_changes', 'total_of_special_requests']
        
        X = ml_data[segment_features].fillna(0)
        X_scaled = self.scaler.fit_transform(X)
        
        # K-Means clustering
        kmeans = KMeans(n_clusters=4, random_state=42)
        ml_data['customer_segment'] = kmeans.fit_predict(X_scaled)
        
        # Analisis segmen
        segment_analysis = ml_data.groupby('customer_segment').agg({
            'lead_time': 'mean',
            'total_stay_nights': 'mean',
            'adr': 'mean',
            'total_guests': 'mean',
            'is_canceled': 'mean',
            'booking_sk': 'count'
        }).round(2)
        
        segment_analysis.columns = ['Avg_Lead_Time', 'Avg_Stay_Nights', 'Avg_ADR', 
                                   'Avg_Guests', 'Cancellation_Rate', 'Count']
        
        return ml_data, segment_analysis

    def setup_layout(self):
        """Setup layout dashboard dengan sidebar"""
        self.app.layout = html.Div([
            dcc.Store(id='filter-store', data={}),
            
            # Sidebar
            html.Div([
                # Logo/Brand
                html.Div([
                    html.H2("🏨 Hotel Analytics", style={
                        'color': 'white', 
                        'margin': '0 0 2rem 0', 
                        'fontSize': '1.5rem',
                        'fontWeight': '600'
                    })
                ], style={'padding': '1.5rem 1rem', 'borderBottom': '1px solid rgba(255,255,255,0.1)'}),
                
                # Navigation Menu
                html.Div([
                    html.Div([
                        html.Div("📊", style={'fontSize': '1.2rem', 'marginRight': '0.75rem'}),
                        html.Span("Dashboard Eksekutif")
                    ], id='nav-executive', className='nav-item active', **{'data-tab': 'executive'}),
                    
                    html.Div([
                        html.Div("❌", style={'fontSize': '1.2rem', 'marginRight': '0.75rem'}),
                        html.Span("Analisis Pembatalan")
                    ], id='nav-cancellation', className='nav-item', **{'data-tab': 'cancellation'}),
                    
                    html.Div([
                        html.Div("👥", style={'fontSize': '1.2rem', 'marginRight': '0.75rem'}),
                        html.Span("Segmentasi & Profitabilitas")
                    ], id='nav-segmentation', className='nav-item', **{'data-tab': 'segmentation'}),
                ], style={'padding': '1rem 0'}),
                
                # Reset Filters Button
                html.Div([
                    html.Button([
                        html.Span("🔄", style={'marginRight': '0.5rem'}),
                        "Reset Filters"
                    ], id='clear-filters-btn', className='reset-btn')
                ], style={'padding': '1rem', 'marginTop': 'auto'})
                
            ], className='sidebar'),
            
            # Main Content
            html.Div([
                # Header dengan active tab indicator
                html.Div([
                    html.H1(id='page-title', children="Dashboard Eksekutif", style={
                        'margin': '0',
                        'fontSize': '2rem',
                        'fontWeight': '600',
                        'color': '#2c3e50'
                    })
                ], className='main-header'),
                
                # Dashboard Content Container
                html.Div(id='dashboard-content', className='content-container')
                
            ], className='main-content')
            
        ], className='app-container')

    def create_executive_dashboard(self, data, filters):
        """Dashboard 1: Eksekutif (tetap sama)"""
        # KPI Cards
        total_bookings = len(data)
        total_cancellations = data['is_canceled'].sum()
        avg_adr = data['adr'].mean()
        cancellation_rate = (total_cancellations / total_bookings * 100) if total_bookings > 0 else 0
        
        kpi_cards = html.Div([
            html.Div([
                html.H3(f"{total_bookings:,}", className='kpi-value'),
                html.P("Total Pemesanan", className='kpi-label')
            ], className="kpi-card kpi-blue"),
            
            html.Div([
                html.H3(f"{total_cancellations:,}", className='kpi-value'),
                html.P("Total Pembatalan", className='kpi-label')
            ], className="kpi-card kpi-red"),
            
            html.Div([
                html.H3(f"${avg_adr:.2f}", className='kpi-value'),
                html.P("ADR Rata-rata", className='kpi-label')
            ], className="kpi-card kpi-green"),
            
            html.Div([
                html.H3(f"{cancellation_rate:.1f}%", className='kpi-value'),
                html.P("Tingkat Pembatalan", className='kpi-label')
            ], className="kpi-card kpi-orange")
        ], className='kpi-grid')
        
        # Tren Pemesanan (Line Chart)
        monthly_trend = data.groupby(['arrival_year', 'arrival_month_name']).size().reset_index(name='bookings')
        monthly_trend['period'] = monthly_trend['arrival_year'].astype(str) + '-' + monthly_trend['arrival_month_name']
        
        trend_fig = px.line(monthly_trend, x='period', y='bookings',
                           title='📈 Tren Pemesanan Bulanan',
                           color_discrete_sequence=['#3498db'])
        trend_fig.update_layout(
            xaxis_title="Period", 
            yaxis_title="Jumlah Pemesanan",
            margin=dict(t=50, l=50, r=50, b=50),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        # Segmentasi Pasar (Bar Chart)
        market_data = data.groupby('market_segment').size().reset_index(name='count')
        market_fig = px.bar(market_data, x='market_segment', y='count',
                           title='🎯 Segmentasi Pasar',
                           color='count',
                           color_continuous_scale='viridis')
        market_fig.update_layout(
            xaxis_title="Segmen Pasar", 
            yaxis_title="Jumlah Pemesanan",
            margin=dict(t=50, l=50, r=50, b=50),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        # Jenis Hotel (Pie Chart)
        hotel_data = data.groupby('hotel').size().reset_index(name='count')
        hotel_fig = px.pie(hotel_data, values='count', names='hotel',
                          title='🏨 Distribusi Jenis Hotel',
                          color_discrete_sequence=px.colors.qualitative.Set3)
        hotel_fig.update_layout(
            margin=dict(t=50, l=50, r=50, b=50),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        return html.Div([
            kpi_cards,
            html.Div([
                dcc.Graph(figure=trend_fig, id='trend-chart', className='chart-container')
            ], className='chart-full'),
            
            html.Div([
                html.Div([
                    dcc.Graph(figure=market_fig, id='market-segment-chart', className='chart-container')
                ], className='chart-half'),
                
                html.Div([
                    dcc.Graph(figure=hotel_fig, id='hotel-pie-chart', className='chart-container')
                ], className='chart-half')
            ], className='chart-row')
        ])

    def create_cancellation_dashboard(self, data, filters):
        """Dashboard 2: Analisis Pembatalan + ML Model"""
        
        # KPI Cards dengan ML
        total_cancellations = data['is_canceled'].sum()
        cancellation_rate = (total_cancellations / len(data) * 100) if len(data) > 0 else 0
        
        kpi_cards = html.Div([
            html.Div([
                html.H3(f"{total_cancellations:,}", className='kpi-value'),
                html.P("Total Pembatalan", className='kpi-label')
            ], className="kpi-card kpi-red"),
            
            html.Div([
                html.H3(f"{cancellation_rate:.1f}%", className='kpi-value'),
                html.P("Tingkat Pembatalan", className='kpi-label')
            ], className="kpi-card kpi-orange")
        ], className='kpi-grid')
        
        # Pembatalan per Bulan
        monthly_cancel = data.groupby(['arrival_month_name', 'is_canceled']).size().reset_index(name='count')
        monthly_cancel_pivot = monthly_cancel.pivot(index='arrival_month_name', columns='is_canceled', values='count').fillna(0)
        
        cancel_trend_fig = go.Figure()
        cancel_trend_fig.add_trace(go.Scatter(x=monthly_cancel_pivot.index, y=monthly_cancel_pivot[True],
                                            mode='lines+markers', name='Dibatalkan',
                                            line=dict(color='#e74c3c', width=3)))
        cancel_trend_fig.add_trace(go.Scatter(x=monthly_cancel_pivot.index, y=monthly_cancel_pivot[False],
                                            mode='lines+markers', name='Tidak Dibatalkan',
                                            line=dict(color='#27ae60', width=3)))
        cancel_trend_fig.update_layout(
            title='📅 Tren Pembatalan per Bulan',
            xaxis_title="Bulan", 
            yaxis_title="Jumlah Pemesanan",
            margin=dict(t=50, l=50, r=50, b=50),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        # Pembatalan per Segmen Pasar
        cancel_by_segment = data.groupby(['market_segment', 'is_canceled']).size().reset_index(name='count')
        cancel_pivot = cancel_by_segment.pivot(index='market_segment', columns='is_canceled', values='count').fillna(0)
        cancel_pivot['total'] = cancel_pivot.sum(axis=1)
        cancel_pivot['cancellation_rate'] = (cancel_pivot[True] / cancel_pivot['total'] * 100).fillna(0)
        
        segment_cancel_fig = px.bar(cancel_pivot.reset_index(), x='market_segment', y='cancellation_rate',
                                   title='📊 Tingkat Pembatalan per Segmen Pasar',
                                   color='cancellation_rate',
                                   color_continuous_scale='Reds')
        segment_cancel_fig.update_layout(
            xaxis_title="Segmen Pasar", 
            yaxis_title="Tingkat Pembatalan (%)",
            margin=dict(t=50, l=50, r=50, b=50),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        # Lead Time vs Pembatalan (Scatter Plot)
        scatter_fig = px.scatter(data, x='lead_time', y='is_canceled',
                                title='⏰ Lead Time vs Status Pembatalan',
                                color='is_canceled',
                                opacity=0.6,
                                color_discrete_map={True: '#e74c3c', False: '#27ae60'})
        scatter_fig.update_layout(
            xaxis_title="Lead Time (hari)", 
            yaxis_title="Status Pembatalan",
            margin=dict(t=50, l=50, r=50, b=50),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        return html.Div([
            kpi_cards,
            
            
            html.Div([
                html.Div([
                    dcc.Graph(figure=segment_cancel_fig, id='segment-cancel-chart', className='chart-container')
                ], className='chart-half'),
                
                html.Div([
                    dcc.Graph(figure=cancel_trend_fig, id='cancel-trend-chart', className='chart-container')
                ], className='chart-half')
            ], className='chart-row'),
            html.Div([
                dcc.Graph(figure=scatter_fig, id='leadtime-scatter', className='chart-container')
            ], className='chart-full')
        ])

    def create_segmentation_dashboard(self, data, filters):
        """Dashboard 3: Segmentasi & Profitabilitas + ML Models"""
        
        # Train ADR Model
        # try:
        #     mse, r2, adr_feature_importance = self.train_adr_model(data)
            
        #     adr_ml_card = html.Div([
        #         html.H3(f"{r2:.2f}", className='kpi-value'),
        #         html.P("R² Score ADR Model", className='kpi-label'),
        #         html.P("(Random Forest)", className='kpi-sublabel')
        #     ], className="kpi-card kpi-purple")
        # except:
        #     adr_ml_card = html.Div([
        #         html.H3("N/A", className='kpi-value'),
        #         html.P("ADR Model Error", className='kpi-label')
        #     ], className="kpi-card kpi-gray")
        #     adr_feature_importance = pd.DataFrame()
        
        # Customer Segmentation
        # try:
        #     segmented_data, segment_analysis = self.customer_segmentation(data)
            
        #     seg_ml_card = html.Div([
        #         html.H3("4", className='kpi-value'),
        #         html.P("Customer Segments", className='kpi-label'),
        #         html.P("(K-Means)", className='kpi-sublabel')
        #     ], className="kpi-card kpi-purple")
        # except:
        #     seg_ml_card = html.Div([
        #         html.H3("N/A", className='kpi-value'),
        #         html.P("Segmentation Error", className='kpi-label')
        #     ], className="kpi-card kpi-gray")
        #     segmented_data = data
        #     segment_analysis = pd.DataFrame()
        
        # KPI Cards
        avg_adr = data['adr'].mean()
        total_revenue = data['revenue'].sum()
        
        kpi_cards = html.Div([
            html.Div([
                html.H3(f"${avg_adr:.2f}", className='kpi-value'),
                html.P("ADR Rata-rata", className='kpi-label')
            ], className="kpi-card kpi-green"),
            
            html.Div([
                html.H3(f"${total_revenue:,.0f}", className='kpi-value'),
                html.P("Total Revenue", className='kpi-label')
            ], className="kpi-card kpi-blue"),
            
        ], className='kpi-grid')

        # Negara vs Hotel (Grouped Bar Chart)
        country_hotel = data.groupby(['country', 'hotel']).size().reset_index(name='count')
        top_countries = data['country'].value_counts().head(10).index
        country_hotel_filtered = country_hotel[country_hotel['country'].isin(top_countries)]
        
        country_fig = px.bar(country_hotel_filtered, x='country', y='count', color='hotel',
                            title='🌍 Top 10 Negara Asal Tamu vs Jenis Hotel',
                            barmode='group')
        country_fig.update_layout(
            xaxis_title="Negara", 
            yaxis_title="Jumlah Pemesanan",
            margin=dict(t=50, l=50, r=50, b=50),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        # ADR per Bulan (Line Chart)
        monthly_adr = data.groupby('arrival_month_name')['adr'].mean().reset_index()
        month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                      'July', 'August', 'September', 'October', 'November', 'December']
        monthly_adr['month_order'] = monthly_adr['arrival_month_name'].apply(lambda x: month_order.index(x) if x in month_order else 12)
        monthly_adr = monthly_adr.sort_values('month_order')
        
        adr_fig = px.line(monthly_adr, x='arrival_month_name', y='adr',
                         title='💰 ADR Rata-rata per Bulan',
                         markers=True,
                         color_discrete_sequence=['#f39c12'])
        adr_fig.update_layout(
            xaxis_title="Bulan", 
            yaxis_title="ADR ($)",
            margin=dict(t=50, l=50, r=50, b=50),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )

        # Revenue per Segmen (Treemap)
        revenue_by_segment = data.groupby('market_segment')['revenue'].sum().reset_index()
        revenue_by_segment = revenue_by_segment.sort_values('revenue', ascending=False)
        
        treemap_fig = px.treemap(revenue_by_segment, path=['market_segment'], values='revenue',
                                title='🎯 Distribusi Pendapatan per Segmen Pasar',
                                color='revenue',
                                color_continuous_scale='Blues')
        treemap_fig.update_layout(
            margin=dict(t=50, l=50, r=50, b=50)
        )
        
        # ADR by Season and Hotel
        adr_season = data.groupby(['season', 'hotel'])['adr'].mean().reset_index()
        season_fig = px.bar(adr_season, x='season', y='adr', color='hotel',
                           title='🌤️ ADR berdasarkan Musim dan Jenis Hotel',
                           barmode='group')
        season_fig.update_layout(
            xaxis_title="Musim", 
            yaxis_title="ADR ($)",
            margin=dict(t=50, l=50, r=50, b=50),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        return html.Div([
            kpi_cards,
            # html.Div([
            #     html.Div([
            #         dcc.Graph(figure=segment_fig, id='customer-segmentation-chart', className='chart-container')
            #     ], className='chart-half'),
                
            #     html.Div([
            #         dcc.Graph(figure=adr_feature_fig, id='adr-feature-chart', className='chart-container')
            #     ], className='chart-half')
            # ], className='chart-row'),
            html.Div([
                html.Div([
                    dcc.Graph(figure=country_fig, id='country-hotel-chart', className='chart-container')
                ], className='chart-half'),
                
                html.Div([
                    dcc.Graph(figure=adr_fig, id='adr-monthly-chart', className='chart-container')
                ], className='chart-half')
            ], className='chart-row'),
            html.Div([
                html.Div([
                    dcc.Graph(figure=treemap_fig, id='revenue-treemap', className='chart-container')
                ], className='chart-half'),
                
                html.Div([
                    dcc.Graph(figure=season_fig, id='season-adr-chart', className='chart-container')
                ], className='chart-half')
            ], className='chart-row')
        ])

    def setup_callbacks(self):
        """Setup callbacks untuk interaktivitas (sama seperti sebelumnya)"""
        
        @self.app.callback(
            [Output('dashboard-content', 'children'),
             Output('page-title', 'children'),
             Output('nav-executive', 'className'),
             Output('nav-cancellation', 'className'),
             Output('nav-segmentation', 'className')],
            [Input('nav-executive', 'n_clicks'),
             Input('nav-cancellation', 'n_clicks'),
             Input('nav-segmentation', 'n_clicks'),
             Input('filter-store', 'data')],
            prevent_initial_call=False
        )
        def update_dashboard_content(exec_clicks, cancel_clicks, seg_clicks, filters):
            ctx = callback_context
            
            # Determine active tab
            active_tab = 'executive'  # default
            if ctx.triggered:
                button_id = ctx.triggered[0]['prop_id'].split('.')[0]
                if button_id == 'nav-cancellation':
                    active_tab = 'cancellation'
                elif button_id == 'nav-segmentation':
                    active_tab = 'segmentation'
            
            # Update nav classes
            nav_classes = ['nav-item', 'nav-item', 'nav-item']
            titles = {
                'executive': 'Dashboard Eksekutif',
                'cancellation': 'Analisis Pembatalan',
                'segmentation': 'Segmentasi & Profitabilitas'
            }
            
            if active_tab == 'executive':
                nav_classes[0] = 'nav-item active'
            elif active_tab == 'cancellation':
                nav_classes[1] = 'nav-item active'
            elif active_tab == 'segmentation':
                nav_classes[2] = 'nav-item active'
            
            # Get data and create dashboard
            data = self.get_data(filters)
            
            if active_tab == 'executive':
                content = self.create_executive_dashboard(data, filters)
            elif active_tab == 'cancellation':
                content = self.create_cancellation_dashboard(data, filters)
            elif active_tab == 'segmentation':
                content = self.create_segmentation_dashboard(data, filters)
            else:
                content = html.Div("Dashboard tidak ditemukan")
            
            return content, titles[active_tab], nav_classes[0], nav_classes[1], nav_classes[2]
        
        # ... (callbacks lainnya sama seperti sebelumnya)
        
        # Callback untuk clear filters
        @self.app.callback(
            Output('filter-store', 'data', allow_duplicate=True),
            [Input('clear-filters-btn', 'n_clicks')],
            prevent_initial_call=True
        )
        def clear_filters(n_clicks):
            if n_clicks:
                return {}
            return dash.no_update

        # --- Tambahan callbacks untuk filter on-click ---

        # Executive: market segment bar chart
        @self.app.callback(
            Output('filter-store', 'data', allow_duplicate=True),
            [Input('market-segment-chart', 'clickData')],
            prevent_initial_call=True
        )
        def update_filter_from_market_segment(click_data):
            if click_data:
                return {'market_segment': [click_data['points'][0]['x']]}
            return dash.no_update

        # Executive: hotel pie chart
        @self.app.callback(
            Output('filter-store', 'data', allow_duplicate=True),
            [Input('hotel-pie-chart', 'clickData')],
            prevent_initial_call=True
        )
        def update_filter_from_hotel_pie(click_data):
            if click_data:
                return {'hotel': [click_data['points'][0]['label']]}
            return dash.no_update

        # Executive: line chart (periode)
        @self.app.callback(
            Output('filter-store', 'data', allow_duplicate=True),
            [Input('trend-chart', 'clickData')],
            prevent_initial_call=True
        )
        def update_filter_from_trend(click_data):
            if click_data:
                return {'period': [click_data['points'][0]['x']]}
            return dash.no_update

        # Cancellation: tingkatan pembatalan per segmen
        @self.app.callback(
            Output('filter-store', 'data', allow_duplicate=True),
            [Input('segment-cancel-chart', 'clickData')],
            prevent_initial_call=True
        )
        def update_filter_from_segment_cancel(click_data):
            if click_data:
                return {'market_segment': [click_data['points'][0]['x']]}
            return dash.no_update

        # Segmentation: negara vs hotel
        @self.app.callback(
            Output('filter-store', 'data', allow_duplicate=True),
            [Input('country-hotel-chart', 'clickData')],
            prevent_initial_call=True
        )
        def update_filter_from_country_hotel(click_data):
            if click_data:
                return {'country': [click_data['points'][0]['x']]}
            return dash.no_update

        # Segmentation: ADR per musim
        @self.app.callback(
            Output('filter-store', 'data', allow_duplicate=True),
            [Input('season-adr-chart', 'clickData')],
            prevent_initial_call=True
        )
        def update_filter_from_season_adr(click_data):
            if click_data:
                return {'season': [click_data['points'][0]['x']]}
            return dash.no_update

        # Segmentation: treemap pendapatan
        @self.app.callback(
            Output('filter-store', 'data', allow_duplicate=True),
            [Input('revenue-treemap', 'clickData')],
            prevent_initial_call=True
        )
        def update_filter_from_treemap(click_data):
            if click_data:
                return {'market_segment': [click_data['points'][0]['label']]}
            return dash.no_update
    
    def run(self, debug=True, port=8050):
        """Menjalankan dashboard"""
        self.app.run(debug=debug, port=port)

# Menjalankan Dashboard
if __name__ == '__main__':
    # Inisialisasi dan jalankan dashboard
    dashboard = HotelDashboard()
    
    # Custom CSS (sama + tambahan untuk ML cards)
    dashboard.app.index_string = '''
    <!DOCTYPE html>
    <html>
        <head>
            {%metas%}
            <title>{%title%}</title>
            {%favicon%}
            {%css%}
            <style>
                * {
                    margin: 0;
                    padding: 0;
                    box-sizing: border-box;
                }
                
                body {
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    min-height: 100vh;
                }
                
                .app-container {
                    display: flex;
                    min-height: 100vh;
                }
                
                .sidebar {
                    width: 280px;
                    background: rgba(44, 62, 80, 0.95);
                    backdrop-filter: blur(10px);
                    display: flex;
                    flex-direction: column;
                    box-shadow: 2px 0 10px rgba(0,0,0,0.1);
                    position: fixed;
                    height: 100vh;
                    z-index: 1000;
                }
                
                .nav-item {
                    display: flex;
                    align-items: center;
                    padding: 1rem 1.5rem;
                    color: rgba(255,255,255,0.8);
                    cursor: pointer;
                    transition: all 0.3s ease;
                    border-left: 3px solid transparent;
                    font-size: 0.95rem;
                    font-weight: 500;
                }
                
                .nav-item:hover {
                    background: rgba(52, 152, 219, 0.1);
                    color: white;
                    border-left-color: #3498db;
                }
                
                .nav-item.active {
                    background: rgba(52, 152, 219, 0.2);
                    color: white;
                    border-left-color: #3498db;
                }
                
                .reset-btn {
                    background: linear-gradient(45deg, #e74c3c, #c0392b);
                    color: white;
                    border: none;
                    padding: 0.75rem 1rem;
                    border-radius: 8px;
                    cursor: pointer;
                    font-size: 0.9rem;
                    font-weight: 500;
                    transition: all 0.3s ease;
                    width: 100%;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                }
                
                .reset-btn:hover {
                    transform: translateY(-2px);
                    box-shadow: 0 4px 15px rgba(231, 76, 60, 0.3);
                }
                
                .main-content {
                    flex: 1;
                    margin-left: 280px;
                    background: rgba(255,255,255,0.95);
                    min-height: 100vh;
                }
                
                .main-header {
                    padding: 2rem;
                    background: white;
                    border-bottom: 1px solid #ecf0f1;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
                }
                
                .content-container {
                    padding: 1rem;
                }
                
                .kpi-grid {
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
                    gap: 1rem;
                    margin-bottom: 1rem;
                }
                
                .kpi-card {
                    background: white;
                    padding: 1.5rem;
                    border-radius: 12px;
                    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
                    border-left: 4px solid;
                    transition: transform 0.3s ease;
                }
                
                .kpi-card:hover {
                    transform: translateY(-2px);
                }
                
                .kpi-blue { border-left-color: #3498db; }
                .kpi-red { border-left-color: #e74c3c; }
                .kpi-green { border-left-color: #27ae60; }
                .kpi-orange { border-left-color: #f39c12; }
                .kpi-purple { border-left-color: #9b59b6; }
                .kpi-gray { border-left-color: #95a5a6; }
                
                .kpi-value {
                    font-size: 2rem;
                    font-weight: 700;
                    margin-bottom: 0.5rem;
                    color: #2c3e50;
                }
                
                .kpi-label {
                    color: #7f8c8d;
                    font-size: 0.9rem;
                    font-weight: 500;
                    margin: 0;
                }
                
                .kpi-sublabel {
                    color: #bdc3c7;
                    font-size: 0.8rem;
                    margin: 0;
                    font-style: italic;
                }
                
                .chart-container {
                    background: white;
                    border-radius: 12px;
                    box-shadow: 0 4px 15px rgba(0,0,0,0.08);
                    overflow: hidden;
                }
                
                .chart-full {
                    margin-bottom: 1rem;
                }
                
                .chart-row {
                    display: grid;
                    grid-template-columns: 1fr 1fr;
                    gap: 1rem;
                    margin-bottom: 1rem;
                }
                
                .chart-half {
                    min-height: 400px;
                }
                
                /* Responsive */
                @media (max-width: 768px) {
                    .sidebar {
                        width: 100%;
                        position: relative;
                        height: auto;
                    }
                    
                    .main-content {
                        margin-left: 0;
                    }
                    
                    .chart-row {
                        grid-template-columns: 1fr;
                    }
                    
                    .kpi-grid {
                        grid-template-columns: repeat(2, 1fr);
                    }
                }
                
                @media (max-width: 480px) {
                    .kpi-grid {
                        grid-template-columns: 1fr;
                    }
                }
            </style>
        </head>
        <body>
            {%app_entry%}
            <footer>
                {%config%}
                {%scripts%}
                {%renderer%}
            </footer>
        </body>
    </html>
    '''
    
    print("🚀 Memulai Hotel Analytics Dashboard dengan ML...")
    print("📊 Dashboard tersedia di: http://localhost:8050")
    print("🤖 ML Models: Cancellation Prediction, ADR Prediction, Customer Segmentation")
    print("💡 Tips: Klik pada chart untuk memfilter data dan lihat prediksi ML!")
    
    dashboard.run(debug=True)