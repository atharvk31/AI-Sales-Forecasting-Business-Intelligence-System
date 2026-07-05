import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error
import plotly.express as px
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler

# ---------------- CONFIG ----------------
st.set_page_config(page_title="🚀Sales Forecasting System", layout="wide")

# ---------------- DARK THEME ----------------
st.markdown("""
<style>
.stApp { background: linear-gradient(135deg, #020617, #0f172a); }
[data-testid="stSidebar"] { background:#020617; }
.block-container {
    background:#0f172a;
    padding:2rem;
    border-radius:12px;
}
h1,h2,h3 { color:#e2e8f0 !important; }
.stMetric { background:#020617; padding:10px; border-radius:10px; }
</style>
""", unsafe_allow_html=True)

st.title("🚀 Sales Forecasting System")

# ---------------- LOAD ----------------
df = pd.read_excel("SuperStore Sales DataSet.xlsx")

# ---------------- PREPROCESS ----------------
df['Order Date'] = pd.to_datetime(df['Order Date'])
df['year'] = df['Order Date'].dt.year
df['month'] = df['Order Date'].dt.month
df['day'] = df['Order Date'].dt.day
df['day_of_week'] = df['Order Date'].dt.dayofweek

df_original = df.copy()

# ---------------- ENCODING ----------------
le_r, le_c, le_s = LabelEncoder(), LabelEncoder(), LabelEncoder()
df['Region'] = le_r.fit_transform(df['Region'])
df['Category'] = le_c.fit_transform(df['Category'])
df['Sub-Category'] = le_s.fit_transform(df['Sub-Category'])

# ---------------- MODEL ----------------
X_qty = df[['year','month','day','day_of_week','Region','Category','Sub-Category']]
y_qty = df['Quantity']

X_sales = df[['year','month','day','day_of_week','Region','Category','Sub-Category','Quantity']]
y_sales = df['Sales']

model_qty = XGBRegressor(n_estimators=200)
model_sales = XGBRegressor(n_estimators=200)

model_qty.fit(X_qty, y_qty)
model_sales.fit(X_sales, y_sales)

# ---------------- METRICS ----------------
pred = model_sales.predict(X_sales)
r2 = r2_score(y_sales, pred)
mae = mean_absolute_error(y_sales, pred)

# ---------------- SIDEBAR ----------------
menu = st.sidebar.radio("Navigation", [
    "🏠 Home",
    "🔮 Prediction",
    "📦 Requirements Planner",
    "📊 KPI Dashboard",
    "📉 Model Analysis",
    "📈 Forecast",
    "🌍 Region View"
])

# ---------------- HOME ----------------
if menu == "🏠 Home":

    st.header("💡 AI Sales Forecasting & Decision Support System")

    st.markdown("""
    ### 📌 Project Overview

    The Sales Forecasting System is an advanced data-driven application designed to help businesses make smarter, faster, and more accurate decisions using Machine Learning. 
    In today’s competitive market, understanding future demand and sales trends is extremely important for efficient business planning.

    This system leverages historical sales data to predict future sales and product demand, enabling organizations to optimize their inventory, improve customer satisfaction, 
    and maximize profitability.

    ---

    ### 🎯 Objectives of the Project

    - Predict future sales based on date, region, and product category
    - Estimate product demand (quantity forecasting)
    - Assist in inventory planning and stock management
    - Provide actionable business insights and recommendations
    - Help decision-makers reduce risks and increase revenue

    ---

    ### 🧠 Machine Learning Approach

    This system uses advanced machine learning algorithms like **XGBoost Regressor**, which is known for:
    
    - High accuracy and performance
    - Handling complex relationships in data
    - Robustness with real-world datasets

    The model is trained on features such as:
    
    - Order Date (Year, Month, Day, Day of Week)
    - Region
    - Category and Sub-Category
    - Quantity

    These features help the model understand patterns and trends in sales behavior.

    ---

    ### ⚙️ Key Functionalities

    🔮 **Sales Prediction**  
    Predicts future sales value based on user input.

    📦 **Demand Forecasting**  
    Estimates the expected quantity of products to be sold.

    📊 **KPI Dashboard**  
    Displays important metrics like total sales, average sales, and number of orders.

    📉 **Model Performance Analysis**  
    Evaluates the model using metrics like R² Score and MAE.

    📈 **Sales Forecasting**  
    Provides trend analysis using historical data.

    🌍 **Region Analysis**  
    Shows region-wise performance to identify strong and weak markets.

    📦 **Requirements Planner**  
    Helps users select products and generate order plans based on demand insights.

    ---

    ### 💼 Business Value

    This system acts as a **Decision Support Tool** for businesses by:

    - Reducing overstocking and understocking problems
    - Improving supply chain efficiency
    - Enhancing strategic planning
    - Increasing overall profitability

    ---

    ### 🚀 Technologies Used

    - Python 🐍
    - Streamlit 🎨 (for UI)
    - Pandas 📊 (data processing)
    - Scikit-learn 🤖 (ML utilities)
    - XGBoost ⚡ (prediction model)
    - Plotly 📈 (visualization)

    ---

    ### 🧾 Conclusion

    The Sales Forecasting System demonstrates how Machine Learning can be effectively used to solve real-world business problems. 
    It transforms raw data into meaningful insights, helping organizations make data-driven decisions with confidence.

    👉 This project is a step towards building intelligent, automated, and scalable business solutions.
    """)

# ---------------- PREDICTION ----------------
# ---------------- PREDICTION ----------------
elif menu == "🔮 Prediction":

    st.header("🔮 AI Sales & Demand Predictor")

    st.markdown("💡 *Enter details below to get intelligent sales predictions and business recommendations*")

    # -------- INPUT UI --------
    col1, col2, col3 = st.columns(3)

    date = col1.date_input("📅 Select Order Date")
    region = col2.selectbox("🌍 Select Region", df_original['Region'].unique())
    category = col3.selectbox("📦 Select Category", df_original['Category'].unique())

    sub = st.selectbox("🧾 Select Sub-Category", df_original['Sub-Category'].unique())

    st.markdown("---")

    # -------- PREDICT BUTTON --------
    if st.button("🚀 Generate Prediction"):

        d = pd.to_datetime(date)

        input_df = pd.DataFrame({
            'year':[d.year],
            'month':[d.month],
            'day':[d.day],
            'day_of_week':[d.dayofweek],
            'Region':[le_r.transform([region])[0]],
            'Category':[le_c.transform([category])[0]],
            'Sub-Category':[le_s.transform([sub])[0]]
        })

        # Predictions
        qty = model_qty.predict(input_df)[0]
        input_df['Quantity'] = qty
        sales = model_sales.predict(input_df)[0]

        # -------- RESULT CARDS --------
        st.subheader("📊 Prediction Results")

        c1, c2, c3 = st.columns(3)

        c1.metric("💰 Predicted Sales", f"$ {round(sales,2)}")
        c2.metric("📦 Expected Demand", int(qty))
        c3.metric("💸 Estimated Profit", f"$ {round(sales*0.2,2)}")

        st.markdown("---")

        # -------- SMART INSIGHTS --------
        st.subheader("🧠 AI Business Insights")

        if sales > 800:
            st.success("🔥 Excellent Sales Opportunity! This product has strong market demand.")
        elif sales > 400:
            st.info("📈 Stable Performance. Product has consistent sales.")
        else:
            st.warning("⚠️ Low Sales Expected. Be cautious with inventory.")

        if qty > 10:
            st.success("📦 High Demand Detected! Customers are likely to buy in large quantities.")
        elif qty > 5:
            st.info("📊 Moderate Demand. Maintain balanced stock.")
        else:
            st.warning("📉 Low Demand. Avoid overstocking.")

        st.markdown("---")

        # -------- RECOMMENDATIONS --------
        st.subheader("🤖 Smart Recommendations")

        # Inventory
        recommended_stock = int(qty * 1.3)
        st.info(f"📦 Recommended Inventory Level: {recommended_stock} units")

        # Marketing
        if sales > 800:
            st.success("📢 Run Ads & Promotions → Maximize profit during high demand.")
        else:
            st.warning("📉 Offer Discounts → Boost low sales performance.")

        # Risk
        if sales < 300:
            st.error("🚨 High Risk Product! Avoid bulk investment.")

        # Profit Strategy
        if sales > 700:
            st.success("💰 High Profit Potential → Increase pricing slightly or bundle offers.")
        else:
            st.info("💡 Use combo offers to increase order value.")

        st.markdown("---")

        # -------- FINAL DECISION --------
        st.subheader("🎯 Final Business Decision")

        if sales > 700 and qty > 8:
            st.success("✅ Strong Buy Decision → Invest more in this product 🚀")
        elif sales > 400:
            st.info("⚖️ Balanced Decision → Maintain current strategy")
        else:
            st.warning("❌ Weak Decision → Reduce stock & rethink strategy")

        st.markdown("---")

        # -------- EXTRA INSIGHT --------
        st.subheader("📌 Bonus Insight")

        best_product = df_original.groupby('Sub-Category')['Sales'].sum().idxmax()
        best_region = df_original.groupby('Region')['Sales'].sum().idxmax()

        st.success(f"🏆 Top Selling Product: {best_product}")
        st.success(f"🌍 Best Performing Region: {best_region}")
# ---------------- REQUIREMENTS PLANNER ----------------
elif menu == "📦 Requirements Planner":

    st.header("📦 Requirements & Order Planning")

    selected_category = st.selectbox("Select Category", df_original['Category'].unique())

    top_products = (
        df_original[df_original['Category'] == selected_category]
        .groupby('Sub-Category')['Sales']
        .sum()
        .sort_values(ascending=False)
        .head(5)
    )

    selected_items = {}

    for product in top_products.index:
        col1, col2 = st.columns([2,1])

        with col1:
            check = st.checkbox(product)

        with col2:
            qty = st.number_input(f"Qty", key=product)

        if check:
            selected_items[product] = qty

    if st.button("🛒 Generate Order Plan"):

        if len(selected_items) == 0:
            st.warning("No products selected")
        else:
            order_df = pd.DataFrame(
                list(selected_items.items()),
                columns=["Product","Quantity"]
            )

            st.dataframe(order_df)

            total = order_df["Quantity"].sum()
            st.success(f"📦 Total Units: {total}")

            if total > 50:
                st.success("🔥 Bulk order recommended")
            elif total > 20:
                st.info("📈 Balanced inventory")
            else:
                st.warning("⚠️ Low order")

            # Download CSV
            csv = order_df.to_csv(index=False).encode('utf-8')

            st.download_button(
                "⬇️ Download Order Plan",
                data=csv,
                file_name="order_plan.csv",
                mime="text/csv"
            )

# ---------------- KPI ----------------
# ---------------- KPI DASHBOARD ----------------
elif menu == "📊 KPI Dashboard":

    st.header("📊 Business KPI Dashboard")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("💰 Total Sales", f"$ {round(df['Sales'].sum(),2)}")
    col2.metric("📈 Avg Sales", f"$ {round(df['Sales'].mean(),2)}")
    col3.metric("📦 Total Orders", df.shape[0])
    col4.metric("🛒 Total Quantity", int(df['Quantity'].sum()))

    st.markdown("---")

    # 🔥 Sales by Category
    cat_sales = df_original.groupby('Category')['Sales'].sum().reset_index()
    fig1 = px.bar(cat_sales, x='Category', y='Sales', title="Sales by Category")
    st.plotly_chart(fig1, use_container_width=True)

    # 🔥 Sales Trend
    monthly = df.groupby('month')['Sales'].sum().reset_index()
    fig2 = px.line(monthly, x='month', y='Sales', title="Monthly Sales Trend")
    st.plotly_chart(fig2, use_container_width=True)

    # 🔥 Region Performance
    region = df_original.groupby('Region')['Sales'].sum().reset_index()
    fig3 = px.pie(region, names='Region', values='Sales', title="Region Contribution")
    st.plotly_chart(fig3, use_container_width=True)

    # 🔥 Top Products
    top_products = df_original.groupby('Sub-Category')['Sales'].sum().sort_values(ascending=False).head(5)
    st.subheader("🏆 Top 5 Products")
    st.dataframe(top_products)

        # ================== 🔥 BEST TIME TO SELL ==================
    st.subheader("📅 Best Time to Sell Analysis")

    # Best Month
    best_month = df.groupby('month')['Sales'].mean().idxmax()
    best_month_value = df.groupby('month')['Sales'].mean().max()

    # Best Day
    best_day = df.groupby('day_of_week')['Sales'].mean().idxmax()

    days_map = {
        0:"Monday", 1:"Tuesday", 2:"Wednesday",
        3:"Thursday", 4:"Friday", 5:"Saturday", 6:"Sunday"
    }

    best_day_name = days_map[best_day]

    c1, c2 = st.columns(2)

    c1.metric("🏆 Best Month", f"Month {best_month}")
    c2.metric("📅 Best Day", best_day_name)

    st.success(f"💰 Highest Avg Sales: $ {round(best_month_value,2)}")

    # Trend graph
    monthly_avg = df.groupby('month')['Sales'].mean().reset_index()

    fig4 = px.line(monthly_avg, x='month', y='Sales',
                   title="📈 Avg Monthly Sales Trend",
                   markers=True)
    st.plotly_chart(fig4, use_container_width=True)

    # ---------------- INSIGHTS ----------------
    st.subheader("🧠 AI Insights")

    if best_month in [11,12]:
        st.success("🔥 Peak Season (Festive boost expected)")
    elif best_month in [6,7]:
        st.info("📈 Mid-Year Growth Period")
    else:
        st.warning("📉 Normal Sales Period")

    st.markdown("---")


    # 🔥 Business Insight
    st.subheader("💡 Insights")

    best_cat = cat_sales.sort_values('Sales', ascending=False).iloc[0]['Category']
    best_region = region.sort_values('Sales', ascending=False).iloc[0]['Region']

    st.success(f"🔥 Best Performing Category: {best_cat}")
    st.success(f"🌍 Best Region: {best_region}")

# ---------------- MODEL ANALYSIS ----------------
# ---------------- MODEL ANALYSIS ----------------
# ---------------- MODEL ANALYSIS ----------------
elif menu == "📉 Model Analysis":

    st.header("📉 Model Performance & Analytical Insights")

    st.markdown("💡 *Evaluate model accuracy, understand errors, and improve performance using data-driven insights*")

    # ---------------- KPI METRICS ----------------
    col1, col2, col3 = st.columns(3)

    rmse = np.sqrt(np.mean((y_sales - pred)**2))

    col1.metric("📊 R² Score", round(r2,3))
    col2.metric("📉 MAE", round(mae,2))
    col3.metric("📉 RMSE", round(rmse,2))

    st.markdown("---")

    # ---------------- ACTUAL VS PREDICTED ----------------
    st.subheader("📊 Actual vs Predicted Sales")

    fig1 = px.scatter(x=y_sales, y=pred,
                      labels={'x':'Actual Sales','y':'Predicted Sales'},
                      title="Model Prediction Accuracy")
    st.plotly_chart(fig1, use_container_width=True)

    # ---------------- ERROR DISTRIBUTION ----------------
    st.subheader("📉 Error Distribution")

    errors = y_sales - pred

    fig2 = px.histogram(errors, nbins=50,
                        title="Prediction Error Distribution")
    st.plotly_chart(fig2, use_container_width=True)

    # ---------------- RESIDUAL PLOT ----------------
    st.subheader("📊 Residual Analysis")

    fig3 = px.scatter(x=pred, y=errors,
                      labels={'x':'Predicted','y':'Error'},
                      title="Residual Plot (Error vs Prediction)")
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown("---")

    # ---------------- FEATURE IMPORTANCE ----------------
    st.subheader("🔥 Feature Importance (Key Drivers)")

    try:
        importance = model_sales.feature_importances_
        features = X_sales.columns

        imp_df = pd.DataFrame({
            'Feature': features,
            'Importance': importance
        }).sort_values(by='Importance', ascending=False)

        fig4 = px.bar(imp_df, x='Feature', y='Importance',
                      title="Feature Contribution to Prediction")
        st.plotly_chart(fig4, use_container_width=True)

    except:
        st.warning("⚠️ Feature importance not available")

    st.markdown("---")

    # ---------------- MODEL INTERPRETATION ----------------
    st.subheader("🧠 Model Interpretation")

    if r2 > 0.85:
        st.success("🔥 Excellent Model → Predictions are highly accurate")
    elif r2 > 0.7:
        st.info("📈 Good Model → Can be improved with tuning")
    else:
        st.warning("⚠️ Weak Model → Needs improvement")

    if mae < 100:
        st.success("📉 Low Error → Reliable predictions")
    else:
        st.warning("⚠️ High Error → Predictions may vary")

    st.markdown("---")

   
# ---------------- FORECAST ----------------
# ---------------- FORECAST ----------------
# ---------------- FORECAST (LSTM) ----------------
# ---------------- FORECAST (LSTM WITH DATE INPUT) ----------------
elif menu == "📈 Forecast":

    st.header("🤖 LSTM Sales Forecasting")

    # ---------------- PREP DATA ----------------
    df_time = df.copy()
    df_time = df_time.sort_values('Order Date')

    daily_sales = df_time.groupby('Order Date')['Sales'].sum().reset_index()

    # 👉 USER DATE INPUT
    selected_date = st.date_input("📅 Select Date to Forecast From")

    if selected_date:

        selected_date = pd.to_datetime(selected_date)

        # Filter data till selected date
        filtered_data = daily_sales[daily_sales['Order Date'] <= selected_date]

        if len(filtered_data) < 15:
            st.error("⚠️ Not enough data before selected date")
        else:

            data = filtered_data['Sales'].values.reshape(-1,1)

            # Normalize
            scaler = MinMaxScaler()
            data_scaled = scaler.fit_transform(data)

            # Create sequences
            def create_sequences(data, seq_len=10):
                X, y = [], []
                for i in range(len(data)-seq_len):
                    X.append(data[i:i+seq_len])
                    y.append(data[i+seq_len])
                return np.array(X), np.array(y)

            X, y = create_sequences(data_scaled)

            # ---------------- MODEL ----------------
            model = Sequential()
            model.add(LSTM(50, input_shape=(X.shape[1],1)))
            model.add(Dense(1))

            model.compile(optimizer='adam', loss='mse')

            # Train
            model.fit(X, y, epochs=5, batch_size=16, verbose=0)

            # ---------------- FORECAST ----------------
            st.subheader(f"🔮 Forecast from {selected_date.date()} (Next 30 Days)")

            last_seq = data_scaled[-10:]
            future_preds = []

            for _ in range(30):
                pred = model.predict(last_seq.reshape(1,10,1), verbose=0)[0][0]
                future_preds.append(pred)
                last_seq = np.append(last_seq[1:], [[pred]], axis=0)

            # Inverse transform
            future_preds = scaler.inverse_transform(np.array(future_preds).reshape(-1,1))

            future_dates = pd.date_range(start=selected_date, periods=30)

            forecast_df = pd.DataFrame({
                'Date': future_dates,
                'Forecast': future_preds.flatten()
            })

            # ---------------- GRAPH ----------------
            fig = px.line(forecast_df, x='Date', y='Forecast',
                          title="📈 Future Sales Forecast (LSTM)")
            st.plotly_chart(fig, use_container_width=True)

            # ---------------- INSIGHTS ----------------
            st.subheader("💡 AI Insights")

            if forecast_df['Forecast'].iloc[-1] > forecast_df['Forecast'].iloc[0]:
                st.success("📈 Sales expected to increase after selected date")
            else:
                st.warning("📉 Sales may decline after selected date")

            st.info("""
            📌 Recommendations:
            - Increase stock if upward trend
            - Plan offers if downward trend
            - Monitor selected period closely
            """)
   
# ---------------- REGION ----------------
# ---------------- REGION VIEW ----------------
elif menu == "🌍 Region View":

    st.header("🌍 Regional Sales Intelligence Dashboard")

    st.markdown("💡 *Analyze region-wise performance, identify top markets, and make strategic decisions*")

    # ---------------- KPI ----------------
    col1, col2, col3 = st.columns(3)

    region_sales_df = df_original.groupby('Region')['Sales'].sum().reset_index()

    best_region = region_sales_df.sort_values('Sales', ascending=False).iloc[0]['Region']
    worst_region = region_sales_df.sort_values('Sales').iloc[0]['Region']

    col1.metric("🏆 Best Region", best_region)
    col2.metric("⚠️ Weak Region", worst_region)
    col3.metric("💰 Total Sales", f"$ {round(region_sales_df['Sales'].sum(),2)}")

    st.markdown("---")

    # ---------------- BAR CHART ----------------
    st.subheader("📊 Region-wise Sales Comparison")

    fig1 = px.bar(region_sales_df, x='Region', y='Sales',
                  color='Sales',
                  title="Sales Distribution Across Regions")
    st.plotly_chart(fig1, use_container_width=True)

    # ---------------- PIE CHART ----------------
    st.subheader("🥧 Market Share by Region")

    fig2 = px.pie(region_sales_df, names='Region', values='Sales',
                  title="Region Contribution to Total Sales")
    st.plotly_chart(fig2, use_container_width=True)

    # ---------------- CATEGORY INSIDE REGION ----------------
    st.subheader("📦 Category Performance by Region")

    selected_region = st.selectbox("Select Region", df_original['Region'].unique())

    region_cat = df_original[df_original['Region'] == selected_region] \
                    .groupby('Category')['Sales'].sum().reset_index()

    fig3 = px.bar(region_cat, x='Category', y='Sales',
                  title=f"{selected_region} - Category Performance")
    st.plotly_chart(fig3, use_container_width=True)

    # ---------------- TREND ----------------
    st.subheader("📈 Region Sales Trend Over Time")

    region_time = df_original[df_original['Region'] == selected_region] \
                    .groupby('month')['Sales'].sum().reset_index()

    fig4 = px.line(region_time, x='month', y='Sales',
                   title=f"{selected_region} Monthly Trend")
    st.plotly_chart(fig4, use_container_width=True)

    # ---------------- INSIGHTS ----------------
    st.subheader("🧠 Regional Insights")

    if selected_region == best_region:
        st.success(f"🔥 {selected_region} is your strongest market! Focus more investment here.")
    elif selected_region == worst_region:
        st.warning(f"⚠️ {selected_region} is underperforming. Needs improvement strategy.")
    else:
        st.info(f"📊 {selected_region} has moderate performance.")

    st.markdown("---")

    # ---------------- RECOMMENDATIONS ----------------
    st.subheader("🤖 Strategic Recommendations")

    if selected_region == best_region:
        st.success("🚀 Increase marketing budget in this region")
        st.success("📦 Expand product availability")
        st.success("💰 Focus on premium pricing strategy")

    elif selected_region == worst_region:
        st.warning("📉 Run discounts & offers to boost sales")
        st.warning("📢 Improve marketing campaigns")
        st.warning("🔍 Analyze customer behavior in this region")

    else:
        st.info("⚖️ Maintain balanced strategy")
        st.info("📈 Gradually increase investment")
        st.info("🧪 Test new products in this region")

    st.markdown("---")

    # ---------------- FINAL DECISION ----------------
    st.subheader("🎯 Final Business Strategy")

    if selected_region == best_region:
        st.success("✅ EXPANSION ZONE → Scale aggressively 🚀")
    elif selected_region == worst_region:
        st.error("❌ RISK ZONE → Optimize before investing ⚠️")
    else:
        st.info("⚖️ STABLE ZONE → Monitor & grow steadily")