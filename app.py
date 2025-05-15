import streamlit as st
import plotly.graph_objs as go
import pandas as pd
import datetime
from sklearn.metrics import confusion_matrix
import plotly.figure_factory as ff

from predict import predict_stock
from stock_csv_maker import download_stock_data

# Set up page
st.set_page_config(page_title="📊 MarketMind AI", layout="centered", initial_sidebar_state="expanded")

# Function for pie chart visualization
def plot_pie_chart(data):
    labels = ["Down", "Up"]
    values = [len(data[data == 0]), len(data[data == 1])]
    
    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=0.3)])
    fig.update_layout(title="Prediction Distribution")
    st.plotly_chart(fig, use_container_width=True)

# Header with logo
st.markdown("""
    <style>
        .header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            background-color: #2c2f35;
            padding: 10px 20px;
            color: white;
        }
        .header img {
            height: 50px;
        }
        .header h1 {
            font-size: 24px;
            margin: 0;
        }
    </style>
    <div class="header">
        <img src="https://i.ibb.co/5gwzqB9n/logo2.png" alt="Logo" style="height:50px;">
        <h1>MarketMind AI</h1>
    </div>
""", unsafe_allow_html=True)

# Sidebar navigation
page = st.sidebar.radio("📌 Navigate", ["🏠 Home", "📊 Evaluation & Charts", "ℹ️ About", "❓ How to Use", "👤 About Me"])
today = datetime.date.today()

# -------------------- HOME PAGE --------------------
if page == "🏠 Home":
    st.title("📈 MarketMind AI")
    st.markdown("Predict stock market movements using AI-powered insights. 🚀")

    with st.form("predict_form"):
        st.subheader("🔍 Stock Details")
        ticker = st.text_input("Enter Ticker Symbol (e.g., AAPL)", max_chars=10)
        start_date = st.date_input("Start Date", value=datetime.date(2020, 1, 1), max_value=today)
        end_date = st.date_input("End Date", value=today, max_value=today)
        submitted = st.form_submit_button("🔮 Predict Trend")

    if submitted:
        if not ticker:
            st.error("❌ Please enter a valid stock ticker.")
        elif start_date > end_date:
            st.error("❌ Start date must be before end date.")
        else:
            ticker = ticker.strip().upper()
            start_str = start_date.strftime("%Y-%m-%d")
            end_str = end_date.strftime("%Y-%m-%d")

            with st.spinner("📡 Downloading stock data..."):
                try:
                    file_path = download_stock_data(ticker, start_str, end_str)
                    st.success(f"✅ Data for {ticker} downloaded successfully!")
                except Exception as e:
                    st.error(f"❌ Failed to fetch data: {e}")
                    st.stop()

            with st.spinner("🤖 Running AI prediction..."):
                try:
                    signal, confidence, tomorrow, df = predict_stock(file_path)
                    st.success(f"📢 Tomorrow's Trend: **{signal.upper()}** (Confidence: {confidence:.2f})")
                    st.session_state['df'] = df

                    st.markdown("---")
                    st.subheader("📊 Price Trend")

                    fig_close = go.Figure()
                    fig_close.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close'))
                    fig_close.update_layout(title="Closing Price", template="plotly_white", xaxis_title="Date", yaxis_title="Price")
                    st.plotly_chart(fig_close, use_container_width=True)

                    st.subheader("🕯️ Candlestick Chart")
                    fig_candle = go.Figure(data=[go.Candlestick(
                        x=df.index,
                        open=df['Open'], high=df['High'],
                        low=df['Low'], close=df['Close'],
                        increasing_line_color='lime', decreasing_line_color='red'
                    )])
                    fig_candle.update_layout(title="Candlestick Chart", template="plotly_white")
                    st.plotly_chart(fig_candle, use_container_width=True)
                except Exception as e:
                    st.error(f"❌ Prediction failed: {e}")

# -------------------- EVALUATION PAGE --------------------
elif page == "📊 Evaluation & Charts":
    st.title("📉 Strategy Evaluation")

    if 'df' in st.session_state:
        df = st.session_state['df']

        if 'Target' not in df.columns:
            st.warning("⚠️ Prediction data incomplete (missing 'Target').")
            st.stop()

        st.subheader("📈 Strategy vs Market Returns")
        df['Daily_Return'] = df['Close'].pct_change()
        df['Strategy'] = df['Target'].shift(1) * df['Daily_Return']
        df_cum = df[['Daily_Return', 'Strategy']].cumsum().fillna(0)

        fig_returns = go.Figure()
        fig_returns.add_trace(go.Scatter(x=df.index, y=df_cum['Daily_Return'], name='Market', line=dict(color='white')))
        fig_returns.add_trace(go.Scatter(x=df.index, y=df_cum['Strategy'], name='Strategy', line=dict(color='yellow')))
        fig_returns.update_layout(title="Cumulative Returns", template="plotly_white", xaxis_title="Date", yaxis_title="Return")
        st.plotly_chart(fig_returns, use_container_width=True)

        st.subheader("🧠 Confusion Matrix")
        y_true = df['Target'].dropna()
        y_pred = df['Target'].shift(1).dropna()
        common_index = y_true.index.intersection(y_pred.index)
        y_true = y_true.loc[common_index]
        y_pred = y_pred.loc[common_index]

        cm = confusion_matrix(y_true, y_pred)
        z_text = [[str(y) for y in x] for x in cm]
        
        # Flip axes here: Predicted on y-axis, Actual on x-axis
        fig_cm = ff.create_annotated_heatmap(
            z=cm.T,  # Transpose to flip axes
            x=["Down", "Up"],  # Actual
            y=["Down", "Up"],  # Predicted
            annotation_text=[[str(cell) for cell in row] for row in cm.T],
            colorscale="Blues", showscale=True
        )
        fig_cm.update_layout(title="📊 Confusion Matrix (Flipped)", xaxis_title="Actual", yaxis_title="Predicted")
        st.plotly_chart(fig_cm, use_container_width=True)
    else:
        st.info("ℹ️ Run a prediction first from the Home page to see results.")

# -------------------- ABOUT PAGE --------------------
elif page == "ℹ️ About":
    st.title("📚 About MarketMind AI")
    st.markdown("""
    MarketMind AI is a powerful tool that leverages machine learning algorithms to predict stock market trends. By using historical data, it provides **buy/sell recommendations** to investors and traders. Whether you're new to the stock market or a seasoned investor, MarketMind AI offers valuable insights for your financial strategy.

    ### Key Features:
    - **Stock Trend Prediction**: Get predictions on whether a stock price will go up or down.
    - **Real-Time Data**: Fetch stock data from any date range.
    - **Strategy Evaluation**: Analyze cumulative returns with market comparisons.
    - **Visual Tools**: Candlestick charts, price trends, and confusion matrices.

    Stay ahead of the market with AI-powered predictions! 🚀
    """)

# -------------------- HOW TO USE PAGE --------------------
elif page == "❓ How to Use":
    st.title("📘 How to Use MarketMind AI")
    st.markdown("""
    Follow these simple steps to use **MarketMind AI**:

    1. **Choose a Stock Ticker**:
        - Enter the stock ticker symbol of the company you're interested in (e.g., AAPL for Apple).
    2. **Select the Date Range**:
        - Choose a start date and an end date to fetch historical stock data.
    3. **Get Prediction**:
        - Hit the **"Predict Trend"** button to receive a prediction about whether the stock will go **Up** or **Down** in the near future.
    4. **Evaluate Performance**:
        - On the **"Evaluation & Charts"** page, compare the predicted trends to the actual market returns using graphs and confusion matrices.
    
    For more information, visit the **About** page or contact us at [support@marketmindai.com](mailto:subham2010sh@gmail.com).
    """)

# -------------------- ABOUT ME PAGE --------------------
elif page == "👤 About Me":
    st.title("👨‍💻 About Me")
    st.markdown("""
    Hello, I am **Subham Sharma**, a data enthusiast and machine learning practitioner with a passion for finance. I created **MarketMind AI** as a way to combine my interests in both stock market analysis and AI development.

    With **MarketMind AI**, I aim to help traders, investors, and financial enthusiasts make better-informed decisions by leveraging advanced machine learning techniques.

    If you have any questions, feedback, or would like to connect, feel free to reach out via email or social media:

    📧 **Email**: [subham2010sh@gmail.com](mailto:subham2010sh@gmail.com)

    💼 **LinkedIn**: [Subham Sharma](https://www.linkedin.com/in/subham-sharma-b6b4a727b/)

    I’m always happy to discuss ideas, collaborate, and share knowledge with others in the community! 😊
    """)

# -------------------- CUSTOM STYLING (ANIMATED BACKGROUND) --------------------
st.markdown("""
<style>
body, .stApp {
  background: linear-gradient(-45deg, #2f2f2f, #3a3a3a, #707070);
  background-size: 400% 400%;
  animation: gradientBG 10s ease infinite;
  color: white !important;
  font-family: 'Segoe UI', sans-serif;
}
.stTextInput > div > div > input {
  background-color: #f0f0f0;
  color: black;
  border: 1px solid #888;
  border-radius: 8px;
  padding: 10px;
}
button {
  background-color: #ff4b4b !important;
  color: white !important;
  border-radius: 10px !important;
  padding: 10px 20px !important;
  font-weight: bold;
}
button:hover {
  background-color: #ff1c1c !important;
}
@keyframes gradientBG {
  0% { background-position: 0% 50%; }
  50% { background-position: 100% 50%; }
  100% { background-position: 0% 50%; }
}
</style>
""", unsafe_allow_html=True)
