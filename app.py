import os
import sys
from curl_cffi import requests
import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
import plotly.figure_factory as ff
from datetime import datetime
from sklearn.metrics import confusion_matrix
from predict import predict_stock
from stock_csv_maker import download_stock_data
import plotly.express as px 
# Helper Functions
def valid_stock(stock):
    try:
        return True
    except Exception:
        return False

def tickers_parser(ticker, max_items=1):
    return ticker.strip()

def get_top_holdings(ticker):
    stock = yf.Ticker(ticker)
    try:
        # Attempt to get institutional holders
        holders = stock.institutional_holders
        if holders is not None:
            return holders[['Holder', 'Shares', 'Date Reported']]
        else:
            return "No top holdings data available."
    except Exception as e:
        return f"Error fetching top holdings: {e}"

def get_stock_info(ticker):
    session = requests.Session(impersonate="chrome")
    stock = yf.Ticker(ticker, session=session)
    return stock.info

def load_reviews_from_csv(filename="reviews.csv"):
    if os.path.exists(filename):
        return pd.read_csv(filename)
    else:
        return pd.DataFrame(columns=["Name", "Review", "Rating", "Timestamp"])

def save_review_to_csv(name, review, rating, filename="reviews.csv"):
    new_entry = pd.DataFrame([{
        "Name": name,
        "Review": review,
        "Rating": rating,
        "Timestamp": pd.Timestamp.now()
    }])
    
    # Append to file or create if not exists
    if os.path.exists(filename):
        existing = pd.read_csv(filename)
        updated = pd.concat([new_entry, existing], ignore_index=True)
    else:
        updated = new_entry

    updated.to_csv(filename, index=False)

def plotly_ohlc_chart(df, vol_col=None):
    fig = go.Figure(data=[go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close')])
    if vol_col:
        fig.update_traces(mode='lines+markers')
    return fig

def show_plotly(fig):
    st.plotly_chart(fig, use_container_width=True)

# Main App
def Main():
    st.set_page_config(
    page_title="MarketMind AI - Stock Trend Prediction",  # Page title
    page_icon="https://i.ibb.co/5gwzqB9n/logo2.png",         # Favicon (the image URL)
    layout="centered",
    initial_sidebar_state="expanded"
    )
    st.markdown(
    """
    <h1 style="text-align:center;">
        <img src="https://i.ibb.co/5gwzqB9n/logo2.png" width="40" style="vertical-align:middle; margin-right:10px; border-radius:40%;">
         MarketMind AI - Stock Trend Prediction
    </h1>
    """,
    unsafe_allow_html=True
    )
    
    # Sidebar
    st.sidebar.header("Enter Stock Information")
    ticker = tickers_parser(st.sidebar.text_input("Enter a Stock Ticker", "AAPL"), max_items=1)
    start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2021-01-01"))
    end_date = st.sidebar.date_input("End Date", pd.to_datetime("2022-01-01"))

    with st.sidebar.expander("Configure", expanded=True):
        show_ohlc = st.checkbox("Show OHLC Chart", value=True)
        show_volume = st.checkbox("Show Volume", value=True)
        b_two_col = st.checkbox("Two-Column View", value=True)
        chart_size = st.number_input("Chart Size", value=500, min_value=400, max_value=1500)

    if st.sidebar.button("Show Data") and ticker:
        stock = yf.Ticker(ticker)
        if not valid_stock(stock):
            st.error(f"Cannot find data for `{ticker}`")
            return

        stock_info = get_stock_info(ticker)
        df_all = stock.history(period="max").tail(250)

        col1, col2 = st.columns([2, 1]) if b_two_col else (st.container(), st.container())

        with col1:
            # AI Prediction
            st.subheader("🔮 AI Stock Trend Prediction")
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
                    st.session_state['df'] = df

                    # Highlighted Buy/Sell Signal
                    signal_color = 'green' if signal == 'Buy' else 'red'
                    st.markdown(
                        f"<h2 style='color:{signal_color};text-align:center;'>📢 Tomorrow's Trend: <b>{signal.upper()}</b></h2>",
                        unsafe_allow_html=True
                    )

                    # Charts Section
                    st.markdown("---")
                    st.subheader("📊 Charts")

                    # Closing Price
                    fig_close = go.Figure()
                    fig_close.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Close'))
                    fig_close.update_layout(title="Closing Price", template="plotly_white",
                                            xaxis_title="Date", yaxis_title="Price")
                    st.plotly_chart(fig_close, use_container_width=True)

                    # Candlestick Chart
                    fig_candle = go.Figure(data=[go.Candlestick(
                        x=df.index, open=df['Open'], high=df['High'],
                        low=df['Low'], close=df['Close'],
                        increasing_line_color='lime', decreasing_line_color='red'
                    )])
                    fig_candle.update_layout(title="Candlestick Chart", template="plotly_white")
                    st.plotly_chart(fig_candle, use_container_width=True)

                    # Strategy Evaluation
                    st.subheader("📉 Strategy Evaluation")
                    if 'Target' not in df.columns:
                        st.warning("⚠️ Prediction data incomplete (missing 'Target').")
                        st.stop()

                    df['Daily_Return'] = df['Close'].pct_change()
                    df['Strategy'] = df['Target'].shift(1) * df['Daily_Return']
                    df_cum = df[['Daily_Return', 'Strategy']].cumsum().fillna(0)

                    fig_returns = go.Figure()
                    fig_returns.add_trace(go.Scatter(x=df.index, y=df_cum['Daily_Return'], name='Market', line=dict(color='white')))
                    fig_returns.add_trace(go.Scatter(x=df.index, y=df_cum['Strategy'], name='Strategy', line=dict(color='yellow')))
                    fig_returns.update_layout(title="Cumulative Returns", template="plotly_white", xaxis_title="Date", yaxis_title="Return")
                    st.plotly_chart(fig_returns, use_container_width=True)

                    # Confusion Matrix
                    st.subheader("🧠 Confusion Matrix")
                    y_true = df['Target'].dropna()
                    y_pred = df['Target'].shift(1).dropna()
                    common_index = y_true.index.intersection(y_pred.index)
                    y_true = y_true.loc[common_index]
                    y_pred = y_pred.loc[common_index]

                    cm = confusion_matrix(y_true, y_pred)
                    fig_cm = ff.create_annotated_heatmap(
                        z=cm.T,
                        x=["Down", "Up"],  # Actual
                        y=["Down", "Up"],  # Predicted
                        annotation_text=[[str(cell) for cell in row] for row in cm.T],
                        colorscale="Blues", showscale=True
                    )
                    fig_cm.update_layout(title="📊 Confusion Matrix (Flipped)", xaxis_title="Actual", yaxis_title="Predicted")
                    st.plotly_chart(fig_cm, use_container_width=True)

                except Exception as e:
                    st.error(f"❌ Prediction failed: {e}")

            # Stock Info
            with st.expander(f"📄 Stock Info: {stock_info.get('longName', ticker)}"):
                tab_desc, tab_json, _ = st.tabs(["Description", "JSON", "Top Holdings"])
                tab_json.write(stock_info)
                website = stock_info.get("website", "")
                summary = stock_info.get("longBusinessSummary", "No summary available.")
                tab_desc.markdown(f"[Website]({website})\n\n{summary}")
            
                try:
                    holdings = get_top_holdings(ticker)
                    st.write(holdings)
                except Exception:
                    st.info(f"{ticker} is not a fund type")

            # OHLC Chart
            if show_ohlc:
                with st.expander("📈 OHLC Chart"):
                    fig = plotly_ohlc_chart(df=df_all, vol_col="Volume" if show_volume else None)
                    fig.update_layout(
                        height=chart_size,
                        title=f"{ticker.upper()} - OHLC Chart (Last 250 Days)",
                        template="plotly_dark",
                    )
                    show_plotly(fig)

            # Historical Prices
            with st.expander("📅 Historical Prices"):
                st.dataframe(df_all)

        with col2:
            with st.expander("🏢 Corporate Actions"):
                st.dataframe(stock.actions)

            with st.expander("💰 Dividends"):
                st.dataframe(stock.dividends)

            with st.expander("🔀 Splits"):
                st.dataframe(stock.splits)

            with st.expander("🏦 Major Holders"):
                st.dataframe(stock.major_holders)

            with st.expander("📊 Institutional Holders"):
                st.dataframe(stock.institutional_holders)

            with st.expander("📰 News"):
                news_dict = stock.news
                for n in news_dict:
                    dt = n["content"]["pubDate"]
                    st.write(
                        f'`{dt}` {n["content"]["provider"]["displayName"]}: [{n["content"]["title"]}]({n["content"]["canonicalUrl"]["url"]})'
                    )
    st.markdown("---")
    st.subheader("📘 About MarketMind AI")
    # --- EXTRA SECTIONS AFTER PREDICTION ---
    st.markdown("\n\n\n")
    st.expander("How to Use MarketMind AI", expanded=True).markdown("""
    1. **Enter Stock Ticker**: Type the stock ticker symbol (e.g., AAPL for Apple).
    2. **Select Dates**: Choose the start and end dates for the analysis.
    3. **Configure Options**: Adjust settings like chart type and size.
    4. **View Data**: Click "Show Data" to fetch and display stock data.
    5. **AI Prediction**: The AI will analyze the data and predict the stock trend for tomorrow.
    6. **Charts**: View various charts including closing price, candlestick, and cumulative returns.
    7. **Confusion Matrix**: Understand the prediction accuracy with a confusion matrix.
    8. **Stock Info**: Get detailed information about the stock, including top holdings.
    9. **Corporate Actions**: View dividends, splits, and major holders.
    10. **News**: Stay updated with the latest news related to the stock.
    """)


    st.markdown("---")
    st.header("📘 About MarketMind AI")
    st.markdown("""
    MarketMind AI is a smart dashboard for retail investors and analysts.  
    It uses machine learning to predict stock trends and visualize market performance.

    - Built with: Python, Streamlit, Plotly, and AI models
    - Data source: Yahoo Finance
    - Goal: Make data-driven investing accessible to everyone.
    """)


    # -- Review Form --
    st.markdown("---")
    st.header("⭐ User Reviews")
    # Read reviews from CSV
    review_df = pd.read_csv("reviews.csv")

    # Check if reviews exist, otherwise display a message
    if review_df.empty: 
        st.info("No reviews yet. Be the first to submit one!")
    else:
        review_df['Timestamp'] = pd.to_datetime(review_df['Timestamp'], errors='coerce')
        review_df['Timestamp'] = review_df['Timestamp'].dt.date

        # Loop through the reviews and display them
        st.subheader("📣 User Reviews Summary")

# Create Expander with Tabs
        with st.expander("📊 Review Analytics", expanded=True):
            tab1, tab2, tab3, tab4 = st.tabs(["Review Summary", "Rating Distribution", "Filter Reviews" , "what users are saying"])

            # --- Tab 1: Review Summary ---
            with tab1:
                overall_reviews = f"""
                #### Overall Reviews
                - **Total Reviews**: {len(review_df)}
                - **Average Rating**: {review_df['Rating'].mean():.2f}/5
                - **Latest Review**: {review_df['Timestamp'].max()}
                """
                st.markdown(overall_reviews, unsafe_allow_html=True)

            # --- Tab 2: Rating Distribution (Pie Chart with Soft Colors) ---
            with tab2:
                # Pie Chart for Rating Distribution with Soft Pastel Colors
                rating_counts = review_df['Rating'].value_counts().sort_index()
                fig = px.pie(
                    names=rating_counts.index,
                    values=rating_counts.values,
                    title="Rating Distribution",
                    labels={"Rating": "Stars"},
                    color=rating_counts.index,
                    color_discrete_map={
                        1: 'lightcoral',   # Soft red
                        2: 'lightyellow',  # Soft yellow
                        3: 'lightgreen',   # Soft green
                        4: 'lightblue',    # Soft blue
                        5: 'lightpink'     # Soft pink
                    }
                )
                st.plotly_chart(fig)

            # --- Tab 3: Filter Reviews ---
            with tab3:
                # Filter Reviews Section
                st.subheader("🔍 Filter Reviews")

                # Add a filter for Rating
                selected_rating = st.selectbox("Select Rating", options=[1, 2, 3, 4, 5], index=0)
                filtered_reviews_by_rating = review_df[review_df['Rating'] == selected_rating]
                
                review_df['Timestamp'] = pd.to_datetime(review_df['Timestamp'], errors='coerce')
                review_df['Timestamp'] = review_df['Timestamp'].dt.date

                # Add a filter for Date Range
                min_date = review_df['Timestamp'].min()
                max_date = review_df['Timestamp'].max()

                start_date, end_date = st.date_input(
                    "Select Date Range", [min_date, max_date], min_value=min_date, max_value=max_date
                )

                filtered_reviews_by_date = filtered_reviews_by_rating[
                    (filtered_reviews_by_rating['Timestamp'] >= start_date) &
                    (filtered_reviews_by_rating['Timestamp'] <= end_date)
                ]

                
                # Display filtered reviews
                if not filtered_reviews_by_date.empty:
                    for _, row in filtered_reviews_by_date.iterrows():
                        st.markdown(f"""
                        #### {row['Name']} ⭐ {int(row['Rating'])}/5
                        ##### {row['Review']} 
                        Date : 
                        {row['Timestamp'].strftime('%Y-%m-%d')} 
                        """)
                        st.markdown("---")
                else:
                    st.info("No reviews found for the selected filters.")

        # Sort reviews by timestamp and display the latest 5
            with tab4:
                st.subheader("📣 What Users Are Saying")
                review_df = review_df.sort_values(by='Timestamp', ascending=False)
                review_df = review_df.head(5)  # Display only the latest 5 reviews
                for _, row in review_df.iterrows():
                    st.markdown(f"""
                    - {row['Name']}  ⭐ {int(row['Rating'])}/5   :    
                    |  {row['Review']}
                    """)

    st.markdown("---")
    # --- Review Form ---
    st.subheader("📝 Leave a Review")
    with st.form("review_form", clear_on_submit=True):
        name = st.text_input("Your Name")
        review = st.text_area("Your Review")
        rating = st.slider("Rating (1 = Poor, 5 = Excellent)", 1, 5, 4)
        submitted = st.form_submit_button("Submit Review")

    if submitted:
        save_review_to_csv(name, review, rating)
        st.success("✅ Thank you for your feedback!")

    # --- Display Reviews ---
    st.subheader("📣 Recent Review")
    review_df = load_reviews_from_csv()

    if not review_df.empty:
        review_df.sort_values(by='Timestamp', ascending=True).head(5)
        st.markdown(f""" 
                   {row['Name']}  ⭐ {int(row['Rating'])}/5   :   {row['Review']}
        """)    
    else:
        st.info("No reviews yet. Be the first to submit one!")


    st.markdown("---")
    st.header("❓ FAQ")
    with st.expander("**Que 1 : What is MarketMind AI?**", expanded=False):
        st.markdown("""**Ans:** MarketMind AI is a smart stock market analysis tool that uses machine learning to forecast stock trends. It combines real-time data from Yahoo Finance with AI to help users make informed investment decisions.""")

    with st.expander("**Que 2 : How accurate is it?**", expanded=False):
        st.markdown("""**Ans:** Accuracy depends on the stock and time frame. MarketMind AI shows backtested strategy performance and confusion matrices so users can judge reliability themselves.""")

    with st.expander("**Que 3 : Can I request features?**", expanded=False):
        st.markdown("""**Ans:** Absolutely! We welcome suggestions. Just contact us via email or submit your ideas through the feedback form.""")

    with st.expander("**Que 4 : Is this suitable for beginners?**", expanded=False):
        st.markdown("""**Ans:** Yes. The interface is designed to be beginner-friendly, with no coding required. Simply enter a ticker, pick dates, and click Predict.""")

    with st.expander("**Que 5 : What markets or stocks are supported?**", expanded=False):
        st.markdown("""**Ans:** Any stock available on Yahoo Finance can be used. This includes major U.S. and international stocks. Crypto and forex are not currently supported.""")

    with st.expander("**Que 6 : How often is the data updated?**", expanded=False):
        st.markdown("""**Ans:** MarketMind AI pulls live data from Yahoo Finance. When you run a prediction, the system fetches the most recent market data for that stock.""")

    with st.expander("**Que 7 : Does this provide financial advice?**", expanded=False):
        st.markdown("""**Ans:** No. This tool is for educational and research purposes only. Always do your own research or consult a licensed financial advisor before investing.""")

    with st.expander("**Que 8 : Can I export charts or data?**", expanded=False):
        st.markdown("""**Ans:** You can take screenshots of charts or use Python/Streamlit features to export data. If you'd like an export button added, let us know!""")

    st.markdown("---")
    st.header("👨‍💻 About Me")

    st.markdown("""
    Hello, I am **Subham Sharma**, a data enthusiast and machine learning practitioner with a passion for finance.  
    I created **MarketMind AI** to merge my interest in stock market analysis and AI development.

    With this project, I aim to help **traders, investors, and financial enthusiasts** make better-informed decisions through the power of machine learning.

    If you’d like to connect, collaborate, or just say hi:

    - 📧 **Email**: [Subham Sharma](mailto:subham2010sh@gmail.com)  
    - 💼 **LinkedIn**: [Subham Sharma](https://www.linkedin.com/in/subham-sharma-b6b4a727b/)  
    - 🤝 Always open to discussions, feedback, and learning together!
    """)


    st.markdown("---")
    st.header("📞 Contact Us")
    st.markdown("""
    - 📧 Email: [support@marketmindai.com](mailto:support@marketmindai.com)  
    - 🌐 Website: [https://market-mind.streamlit.app/](https://market-mind.streamlit.app/)  
    - 🏢 Office: Sector 11 , Faridabad 
    """)
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


if __name__ == "__main__":
    Main()
