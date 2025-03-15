import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# --- (All your existing functions: fetch_data, ..., plot_data_plotly, etc. remain the SAME) ---
# Download VADER lexicon
try:
    sid = SentimentIntensityAnalyzer()  # Test if already downloaded
except LookupError:
    nltk.download('vader_lexicon')
    sid = SentimentIntensityAnalyzer()

COIN_GECKO_API_URL = "https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart?vs_currency=usd&days={days}"
COIN_GECKO_COIN_URL = "https://api.coingecko.com/api/v3/coins/{coin_id}"  # New URL for coin info
FNG_API_URL = "https://api.alternative.me/fng/?limit=1"
COINS = {
    "bitcoin": "Bitcoin",
    "ethereum": "Ethereum",
    "dogecoin": "Dogecoin",
    "cardano": "Cardano",
    "solana": "Solana",
    "tether": "Tether",
    "xrp": "XRP",
    "binancecoin": "Binance Coin",
    "usdcoin": "USD Coin",
    "polkadot": "Polkadot",
    "avalanche": "Avalanche",
    "litecoin": "Litecoin",
    "chainlink": "Chainlink",
    "shibainu": "Shiba Inu",
    "polygon": "Polygon",
    "tron": "TRON",
    "uniswap": "Uniswap",
    "wrappedbitcoin": "Wrapped Bitcoin",
    "bitcoincash": "Bitcoin Cash",
    "stellar": "Stellar",
    "cosmos": "Cosmos",
    "monero": "Monero",
    "ethereumclassic": "Ethereum Classic",
    "algorand": "Algorand",
    "vechain": "VeChain",
    "internetcomputer": "Internet Computer",
    "filecoin": "Filecoin",
    "hedera": "Hedera",
    "apecoin": "ApeCoin",
    "tezos": "Tezos",
    "theta": "Theta Network",
    "elrond": "Elrond",
    "eos": "EOS",
    "flow": "Flow",
    "fantom": "Fantom",
    "decentraland": "Decentraland",
    "aave": "Aave",
    "chiliz": "Chiliz",
    "axieinfinity": "Axie Infinity",
    "sandbox": "Sandbox",
    "zcash": "Zcash",
    "maker": "Maker",
    "kusama": "Kusama",
    "neo": "Neo",
    "helium": "Helium",
    "klaytn": "Klaytn",
    "iota": "IOTA",
    "bittorrent": "BitTorrent",
    "gala": "Gala",
    "dash": "Dash",
    "stacks": "Stacks",
    "enjicoin": "Enjin Coin",
    "loopring": "Loopring",
    "pancakeswap": "PancakeSwap",
    "celo": "Celo",
    "harmony": "Harmony",
    "nem": "NEM",
    "convexfinance": "Convex Finance",
    "curvedaotoken": "Curve DAO Token",
    "decred": "Decred",
    "kava": "Kava",
    "arweave": "Arweave",
    "mina": "Mina",
    "gnosis": "Gnosis",
    "qtum": "Qtum",
    "1inch": "1inch",
    "waves": "Waves",
    "ravencoin": "Ravencoin",
    "iotex": "IoTeX",
    "zilliqa": "Zilliqa",
    "oasisnetwork": "Oasis Network",
    "ankr": "Ankr",
    "terra": "Terra",
    "ontology": "Ontology",
    "serum": "Serum",
    "safemoon": "SafeMoon",
    "dydx": "dYdX",
    "ankr": "Ankr",
    "masknetwork": "Mask Network",
    "lisk": "Lisk",
    "horizen": "Horizen",
    "telcoin": "Telcoin",
    "komodo": "Komodo",
    "digibyte": "DigiByte",
    "nervos": "Nervos Network",
    "storj": "Storj",
    "civic": "Civic",
    "nkn": "NKN",
    "dent": "Dent",
    "syscoin": "Syscoin",
    "orchid": "Orchid",
    "fetchai": "Fetch.ai",
    "metal": "Metal",
    "verge": "Verge",
    "ark": "Ark",
    "oceanprotocol": "Ocean Protocol",
    "moonriver": "Moonriver",
    "wax": "WAX",
    "thetafuel": "Theta Fuel",
    "coti": "COTI",
    "originprotocol": "Origin Protocol",
    "aergo": "Aergo",
    "utrust": "Utrust"
}


# --- Data Fetching ---
@st.cache_data(ttl=3600)
import time

def fetch_data(coin_id, days):
    """Fetches historical price data from CoinGecko API."""
    url = COIN_GECKO_API_URL.format(coin_id=coin_id, days=days)
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        df = pd.DataFrame(data['prices'], columns=['Timestamp', 'Price'])
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')
        df.set_index('Timestamp', inplace=True)
        # Calculate Open, High, Low, Close for Candlestick
        df['Open'] = df['Price'].shift(1)
        df['High'] = df['Price'].rolling(window=2).max()
        df['Low'] = df['Price'].rolling(window=2).min()
        df['Close'] = df['Price']
        df.dropna(inplace=True)  # Drop the first row with NaN values
        time.sleep(0.1) # Add a small delay to respect rate limits. (adjust the delay to align with documented rate limit)
        return df
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching data: {e}")
        return None
    except (KeyError, ValueError) as e:
        st.error(f"Error parsing data: {e}")
        return None

def fetch_coin_info(coin_id):
    """Fetches coin information (including market cap) from CoinGecko API."""
    url = COIN_GECKO_COIN_URL.format(coin_id=coin_id)
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        market_cap = data['market_data']['market_cap']['usd']
        time.sleep(0.1) # Adjust value if api rate limit is to high
        return market_cap
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching coin info: {e}")
        return None
    except (KeyError, ValueError) as e:
        st.error(f"Error parsing coin info: {e}")
        return None

@st.cache_data(ttl=3600)
def fetch_fear_greed_index():
    """Fetches Fear & Greed Index from Alternative.me API."""
    try:
        response = requests.get(FNG_API_URL)
        response.raise_for_status()
        data = response.json()
        if data and data['data']:
            index_value = int(data['data'][0]['value'])
            return index_value
        else:
            st.warning("Fear & Greed API returned unexpected data format.")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching Fear & Greed Index: {e}")
        return None
    except (KeyError, ValueError) as e:
        st.error(f"Error parsing Fear & Greed Index: {e}")
        return None

# --- Sentiment Analysis ---
def get_sentiment_score(text):
    """Calculates sentiment score using NLTK VADER."""
    try:
        sid = SentimentIntensityAnalyzer()
        sentiment_scores = sid.polarity_scores(text)
        compound_score = sentiment_scores['compound']
        return compound_score
    except Exception as e:
        st.error(f"Error calculating sentiment: {e}")
        return 0.0

def analyze_sentiment(coin_name):
    """Performs sentiment analysis on cryptocurrency-related news (example)."""
    news_headlines = [
        f"{coin_name} price surges to a new all-time high!",
        f"Analysts predict a major correction for {coin_name} in the coming weeks.",
        f"Regulatory uncertainty continues to plague the {coin_name} market.",
        f"Adoption of {coin_name} by major corporations is on the rise.",
        f"Hacker steals millions of dollars worth of {coin_name} from exchange."
    ]
    sentiment_scores = [get_sentiment_score(headline) for headline in news_headlines]
    average_sentiment = np.mean(sentiment_scores) if sentiment_scores else 0.0
    return average_sentiment

# --- Model Training ---
def train_model(df, model_name, poly_degree=2):
    """Trains the specified model."""
    df['TimeIndex'] = range(len(df))
    X = df[['TimeIndex']]
    y = df['Price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    if model_name == "Linear Regression":
        prediction_model = LinearRegression()
        prediction_model.fit(X_train, y_train)
    elif model_name == "Polynomial Regression":
        poly = PolynomialFeatures(degree=poly_degree)  # Use the specified degree
        X_poly = poly.fit_transform(X_train)
        prediction_model = LinearRegression()
        prediction_model.fit(X_poly, y_train)
        prediction_model.poly = poly
    elif model_name == "Decision Tree":
        prediction_model = DecisionTreeRegressor()
        prediction_model.fit(X_train, y_train)
    else:
        st.error(f"Invalid model name: {model_name}")
        return None
    return prediction_model  # Return the prediction_model

# --- Prediction ---
def predict_future(model, last_time_index, days_to_predict, df, model_name, poly_degree=2):
    """Predicts future prices using the trained model, starting from the last known price."""
    future_dates = []
    future_prices = []

    if len(df) > 1:
        frequency = (df.index[1] - df.index[0])
        last_price = df['Price'].iloc[-1]  # Get the last known price
    else:
        st.warning("Insufficient data to determine frequency. Using default of 1 day.")
        frequency = timedelta(days=1)
        last_price = 0.0  # Default last price

    # Calculate the first predicted point starting from the last known price
    future_time_index = last_time_index + 1
    future_date = df.index[-1] + frequency

    if model_name == "Linear Regression" or model_name == "Decision Tree":
        future_price = model.predict([[future_time_index]])[0]
    elif model_name == "Polynomial Regression":
        X_future = np.array([[future_time_index]])
        poly = PolynomialFeatures(degree=poly_degree)
        X_future_poly = poly.fit_transform(X_future)
        future_price = model.predict(X_future_poly)[0]
    else:
        return [], []

    future_dates.append(future_date)  # Add the first predicted date
    future_prices.append(future_price)  #Add the first predicted prices

    #Calculate rest of the values
    for i in range(2, days_to_predict + 1):  #start index are now 2 , since first value is allready calculated
        future_time_index = last_time_index + i
        future_date = df.index[-1] + (frequency * i)

        if model_name == "Linear Regression" or model_name == "Decision Tree":
            future_price = model.predict([[future_time_index]])[0]
        elif model_name == "Polynomial Regression":
            X_future = np.array([[future_time_index]])
            poly = PolynomialFeatures(degree=poly_degree)
            X_future_poly = poly.fit_transform(X_future)
            future_price = model.predict(X_future_poly)[0]
        else:
            return [], []

        future_dates.append(future_date)
        future_prices.append(future_price)

    return future_dates, future_prices

def calculate_moving_average(df, window):
    """Calculates the moving average of the price data."""
    return df['Price'].rolling(window=window).mean()

def calculate_bollinger_bands(df, window, num_std=2):
    """Calculates the Bollinger Bands."""
    rolling_mean = df['Price'].rolling(window=window).mean()
    rolling_std = df['Price'].rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return upper_band, lower_band

def calculate_rsi(df, window=14):
    """Calculates the Relative Strength Index (RSI)."""
    if len(df) < window:
        st.warning(f"Data size is too small for RSI calculation with window {window}. Returning None.")
        return None
    delta = df['Price'].diff()
    up, down = delta.copy(), delta.copy()
    up[up < 0] = 0
    down[down > 0] = 0
    roll_up1 = up.ewm(span=window, adjust=False).mean()
    roll_down1 = down.abs().ewm(span=window, adjust=False).mean()
    RS = roll_up1 / roll_down1
    RSI = 100.0 - (100.0 / (1.0 + RS))
    return RSI

def calculate_macd(df, fast_period=12, slow_period=26, signal_period=9):
    """Calculates the Moving Average Convergence Divergence (MACD)."""
    if len(df) < slow_period:
        st.warning(f"Data size is too small for MACD calculation with slow period {slow_period}. Returning None.")
        return None, None, None
    fast_ema = df['Price'].ewm(span=fast_period, adjust=False).mean()
    slow_ema = df['Price'].ewm(span=slow_period, adjust=False).mean()
    macd = fast_ema - slow_ema
    signal = macd.ewm(span=signal_period, adjust=False).mean()
    histogram = macd - signal
    return macd, signal, histogram

def plot_residuals(df, model, model_name, poly_degree=2):
    """Plots the residuals of the model."""

    df['TimeIndex'] = range(len(df)) # Ensure the range is same.
    X = df[['TimeIndex']]
    y = df['Price']

    if model_name == "Linear Regression":
        predictions = model.predict(X)
    elif model_name == "Polynomial Regression":
        poly = PolynomialFeatures(degree=poly_degree)
        X_poly = poly.fit_transform(X)
        predictions = model.predict(X_poly)
    elif model_name == "Decision Tree":
         predictions = model.predict(X)
    else:
        st.error("This Model is not supported")
        return None

    residuals = y - predictions # Get residuals

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df.index, residuals)
    ax.set_xlabel('Date')
    ax.set_ylabel('Residuals')
    ax.set_title('Residual Plot')
    ax.grid(True)
    fig.autofmt_xdate()
    return fig


# --- New Technical Indicator Functions ---

def calculate_roc(df, window=14):
    """Calculates the Rate of Change (ROC)."""
    if len(df) < window:
        st.warning(f"Data size is too small for ROC calculation. Returning None.")
        return None
    roc = ((df['Price'] - df['Price'].shift(window)) / df['Price'].shift(window)) * 100
    return roc

def calculate_stochastic_oscillator(df, k_window=14, d_window=3):
    """Calculates the Stochastic Oscillator (%K and %D)."""
    if len(df) < k_window:
        st.warning(f"Data size is too small for Stochastic Oscillator calculation. Returning None.")
        return None, None

    low_min = df['Low'].rolling(window=k_window).min()
    high_max = df['High'].rolling(window=k_window).max()
    k = 100 * ((df['Close'] - low_min) / (high_max - low_min))
    d = k.rolling(window=d_window).mean()
    return k, d

def calculate_williams_r(df, window=14):
    """Calculates Williams %R."""
    if len(df) < window:
        st.warning(f"Data size is too small for Williams %R calculation. Returning None.")
        return None
    highest_high = df['High'].rolling(window=window).max()
    lowest_low = df['Low'].rolling(window=window).min()
    r = (highest_high - df['Close']) / (highest_high - lowest_low) * -100
    return r

def calculate_ema(df, window):
    """Calculates the Exponential Moving Average (EMA)."""
    return df['Price'].ewm(span=window, adjust=False).mean()

def calculate_ichimoku_cloud(df, tenkan_window=9, kijun_window=26, senkou_span_b_window=52):
    """Calculates the Ichimoku Cloud components."""
    if len(df) < senkou_span_b_window:  # Need at least 52 periods
        st.warning("Data size is too small for Ichimoku Cloud calculation. Returning None.")
        return None, None, None, None, None

    high_9 = df['High'].rolling(window=tenkan_window).max()
    low_9 = df['Low'].rolling(window=tenkan_window).min()
    tenkan_sen = (high_9 + low_9) / 2

    high_26 = df['High'].rolling(window=kijun_window).max()
    low_26 = df['Low'].rolling(window=kijun_window).min()
    kijun_sen = (high_26 + low_26) / 2

    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun_window)

    high_52 = df['High'].rolling(window=senkou_span_b_window).max()
    low_52 = df['Low'].rolling(window=senkou_span_b_window).min()
    senkou_span_b = ((high_52 + low_52) / 2).shift(kijun_window)

    chikou_span = df['Close'].shift(-kijun_window)  # Shift *backwards*

    return tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, chikou_span


def calculate_atr(df, window=14):
    """Calculates the Average True Range (ATR)."""
    if len(df) < window:
        st.warning(f"Data size is too small for ATR calculation. Returning None.")
        return None

    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()

    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)
    atr = true_range.rolling(window=window).mean()
    return atr

def calculate_obv(df):
    """Calculates On-Balance Volume (OBV)."""
    obv = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    return obv

def calculate_cmf(df, window=20):
    """Calculates the Chaikin Money Flow (CMF)."""
    if len(df) < window:
        st.warning(f"Data size is too small for CMF calculation.  Returning None.")
        return None

    mf_multiplier = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
    mf_volume = mf_multiplier * df['Volume']
    cmf = mf_volume.rolling(window=window).sum() / df['Volume'].rolling(window=window).sum()
    return cmf


# --- Visualization ---
def plot_data_plotly(df, future_dates, future_prices, chart_type,
                    show_moving_average=False, moving_average_window=50,
                    show_bollinger_bands=False, bollinger_window=20, num_std=2,
                    show_rsi=False, rsi_window=14,
                    show_macd=False,
                    show_roc=False, roc_window=14,
                    show_stoch=False, k_window=14, d_window=3,
                    show_willr=False, willr_window=14,
                    show_ema=False, ema_window=20,
                    show_ichimoku=False, tenkan_window=9, kijun_window=26, senkou_span_b_window=52,
                    show_atr=False, atr_window=14,
                    show_obv=False,
                    show_cmf=False, cmf_window=20):
    """Plots historical data and predictions using Plotly."""

    fig = go.Figure()
    if df is None or df.empty: #Adding Check
        st.error("Unable to create Chart with empty data")
        return fig

    # --- Base Chart (always present) ---
    if chart_type == "Line":
        fig.add_trace(go.Scatter(x=df.index, y=df['Price'], mode='lines', name='Historical Price'))
        if future_dates and future_prices: # Check if future dates and prices are not empty
            fig.add_trace(go.Scatter(x=future_dates, y=future_prices, mode='lines', name='Predicted Price', line=dict(dash='dash')))
    elif chart_type == "Candlestick":
        fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Price'], name='Historical Price'))
    elif chart_type == "Bar":
        fig.add_trace(go.Bar(x=df.index, y=df['Price'], name='Historical Price'))
    elif chart_type == "Histogram":
        fig = go.Figure(data=[go.Histogram(x=df['Price'])])
        fig.update_layout(xaxis_title="Price", yaxis_title="Frequency")

    # --- Technical Indicators (added based on user selection) ---

    if show_moving_average and len(df) >= moving_average_window:
        moving_average = calculate_moving_average(df, moving_average_window)
        fig.add_trace(go.Scatter(x=df.index, y=moving_average, mode='lines', name=f'{moving_average_window}-Day MA'))

    if show_bollinger_bands and len(df) >= bollinger_window:
        upper_band, lower_band = calculate_bollinger_bands(df, bollinger_window, num_std)
        fig.add_trace(go.Scatter(x=df.index, y=upper_band, mode='lines', name=f'Bollinger Upper Band ({num_std} std)', line=dict(color='rgba(255,0,0,0.3)')))
        fig.add_trace(go.Scatter(x=df.index, y=lower_band, mode='lines', name=f'Bollinger Lower Band ({num_std} std)', line=dict(color='rgba(0,0,255,0.3)'), fill='tonexty'))

    if show_rsi:
        rsi = calculate_rsi(df, rsi_window)
        if rsi is not None:
            fig.add_trace(go.Scatter(x=df.index, y=rsi, mode='lines', name=f'RSI ({rsi_window})', yaxis="y2"))

    if show_macd:
        macd, signal, histogram = calculate_macd(df)
        if macd is not None:  # Check if MACD was calculated
            fig.add_trace(go.Scatter(x=df.index, y=macd, mode='lines', name='MACD', yaxis="y3"))
            fig.add_trace(go.Scatter(x=df.index, y=signal, mode='lines', name='Signal Line', yaxis="y3"))

    if show_roc:
        roc = calculate_roc(df, roc_window)
        if roc is not None:
          fig.add_trace(go.Scatter(x=df.index, y=roc, mode='lines', name=f'ROC ({roc_window})', yaxis='y4'))
    if show_stoch:
        k, d = calculate_stochastic_oscillator(df, k_window, d_window)
        if k is not None:
            fig.add_trace(go.Scatter(x=df.index, y=k, mode='lines', name=f'%K ({k_window})', yaxis='y5'))
            fig.add_trace(go.Scatter(x=df.index, y=d, mode='lines', name=f'%D ({d_window})', yaxis='y5'))

    if show_willr:
        willr = calculate_williams_r(df, willr_window)
        if willr is not None:
           fig.add_trace(go.Scatter(x=df.index, y=willr, mode='lines', name=f'Williams %R ({willr_window})', yaxis='y6'))

    if show_ema and len(df) >= ema_window:
       ema = calculate_ema(df,ema_window)
       fig.add_trace(go.Scatter(x=df.index, y=ema, mode='lines', name=f'{ema_window}-Day EMA'))

    if show_ichimoku:
        tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, chikou_span = calculate_ichimoku_cloud(df)
        if tenkan_sen is not None: # Check to prevent error
            fig.add_trace(go.Scatter(x=df.index, y=tenkan_sen, mode='lines', name='Tenkan-sen'))
            fig.add_trace(go.Scatter(x=df.index, y=kijun_sen, mode='lines', name='Kijun-sen'))
            fig.add_trace(go.Scatter(x=df.index, y=senkou_span_a, mode='lines', name='Senkou Span A'))
            fig.add_trace(go.Scatter(x=df.index, y=senkou_span_b, mode='lines', name='Senkou Span B', fill='tonexty'))
            fig.add_trace(go.Scatter(x=df.index, y=chikou_span, mode='lines', name='Chikou Span'))

    if show_atr:
        atr = calculate_atr(df, atr_window)
        if atr is not None:
          fig.add_trace(go.Scatter(x=df.index, y=atr, mode='lines', name=f'ATR ({atr_window})', yaxis='y7'))
    if show_obv:
       # Create fake data for obv, and cmf, as we don't have volume
       df['Volume'] = np.random.randint(1000, 5000, size=len(df))  # Random volume
       obv = calculate_obv(df)
       fig.add_trace(go.Scatter(x=df.index, y=obv, mode='lines', name=f'OBV', yaxis='y8'))
    if show_cmf:
       df['Volume'] = np.random.randint(1000, 5000, size=len(df))  # Random volume for example
       cmf = calculate_cmf(df, cmf_window)
       if cmf is not None:
          fig.add_trace(go.Scatter(x=df.index, y=cmf, mode='lines', name=f'CMF ({cmf_window})', yaxis='y9'))

    # --- Layout Configuration ---
    fig.update_layout(
        title=f'Cryptocurrency Price ({chart_type} Chart)',
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        hovermode="x unified",
        yaxis=dict(title="Price (USD)"),
        yaxis2=dict(title="RSI", overlaying="y", side="right", range=[0, 100]),  # RSI overlay
        yaxis3=dict(title="MACD", overlaying="y", side="right", showgrid=False, zeroline=False), # MACD
        yaxis4=dict(title="ROC", overlaying="y", side="right", showgrid=False, zeroline=False),  # ROC
        yaxis5=dict(title="Stochastic", overlaying="y", side="right", showgrid=False, zeroline=False, range=[0, 100]),  # Stochastic
        yaxis6=dict(title="Williams %R", overlaying="y", side="right", showgrid=False, zeroline=False, range=[-100, 0]),  # Williams %R
        yaxis7=dict(title="ATR", overlaying="y", side="right", showgrid=False, zeroline=False),  # ATR
        yaxis8=dict(title="OBV", overlaying="y", side="right", showgrid=False, zeroline=False), # OBV
        yaxis9=dict(title="CMF", overlaying="y", side="right", showgrid=False, zeroline=False), # CMF
        xaxis_rangeslider_visible=False # Add this line to remove range slider
    )

    return fig

def plot_fear_greed_pie_chart(index_value):
    """Plots Fear & Greed Index."""
    if index_value is None: return None
    levels = {
        "Extreme Fear": (index_value, '#8B0000'),  # Darker Red
        "Fear": (index_value, '#FF4500'),        # Orange Red
        "Neutral": (index_value, '#F0E68C'),     # Khaki
        "Greed": (index_value, '#9ACD32'),        # Yellow Green
        "Extreme Greed": (index_value, '#006400')   # Dark Green
    }
    for level, (val, color) in levels.items():
        if val <= 25 and level == "Extreme Fear": break
        if val <= 45 and level == "Fear": break
        if val <= 55 and level == "Neutral": break
        if val <= 75 and level == "Greed": break
        if val > 75 and level == "Extreme Greed": break

    labels = [level, '']
    sizes = [index_value, 100 - index_value]
    colors = [color, '#f0f0f0'] # Lighter neutral color

    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90, textprops={'color': "black"})
    ax.axis('equal')
    return fig
def display_sentiment_analysis(average_sentiment):
    """Displays sentiment analysis."""
    if average_sentiment is None:
        st.write("Sentiment analysis unavailable.")
        return
    sentiments = {
       average_sentiment > 0.2 : ("Overall positive sentiment.", "success"),
       average_sentiment < -0.2: ("Overall negative sentiment.", "error"),
    }
    st.write("### Sentiment Analysis")
    st.write(f"Average Sentiment Score: {average_sentiment:.2f}")
    for condition, value in sentiments.items():
      if condition:
        if value[1] == "success":
            st.success(value[0])
        elif value[1] == "error":
            st.error(value[0])
        return
    st.info("Overall neutral sentiment.")

def display_historical_data(df):
    """Displays historical data."""
    st.write("### Historical Data")
    st.dataframe(df)

def generate_summary(df, future_dates, future_prices, selected_coin, model_name, market_cap):
    """Generates a summary of data and predictions."""

    if df is None or df.empty:
        return "No historical data available to generate a summary."

    last_price = df['Price'].iloc[-1]
    predicted_price_str = f"${future_prices[0]:.2f}" if future_prices else "N/A"
    trend = "an upward" if future_prices and future_prices[0] > last_price else "a downward" if future_prices and future_prices[0] < last_price else "unclear"
    market_cap_str = f"${market_cap:,.2f}" if market_cap is not None else "N/A"

    summary = (
        f"Here's a detailed summary for {selected_coin}:<br><br>"
        f"The current price is ${last_price:.2f}.  Based on the {model_name} model, "
        f"the prediction suggests {trend} trend with a predicted price of {predicted_price_str}.<br><br>"
        f"The market capitalization is approximately: {market_cap_str}."
    )
    return summary

# --- STREAMLIT UI (Corrected Layout and Styling) ---

# Custom CSS - Simplified
st.markdown(
    """
    <style>
    body {
        color: #333;
        font-family: sans-serif;
    }
    .stApp {
        background: linear-gradient(to bottom, #e6f7ff, #b3d9ff);
        background-size: cover;
    }
    .st-bx, .st-bw, .st-cb, .st-cj, .st-cl, .st-cm, .st-cn, .st-cp, .st-cq,
    .st-cr, .st-cs, .st-ct, .st-cu, .st-cv, .st-cw, .st-cx, .st-cy, .st-cz,
    .st-da, .st-db, .st-dc, .st-dd, .st-de, .st-df, .st-dg, .st-dh, .st-di,
    .st-dj, .st-dk, .st-dl, .st-dm {
        background-color: rgba(255, 255, 255, 0.6) !important;
        border-radius: 8px !important;
        padding: 0px !important;
        margin-bottom: 0px !important;
    }
    .stSelectbox > div > div > div {
        padding: 5px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("Cryptocurrency Price Prediction and Analysis")

# --- Layout with Columns ---
col2, col1 = st.columns([1, 3])  # Explanations and main content

with col1:  # MAIN CONTENT (Right Side)
    selected_coin = st.selectbox("Select Cryptocurrency", options=COINS.values())
    coin_id = [k for k, v in COINS.items() if v == selected_coin][0]

    # --- Sidebar (Settings) - Inside col1, still a sidebar ---
    with st.sidebar:
        st.header("Settings")
        with st.expander("Prediction Settings", expanded=True):
            prediction_horizon = st.selectbox("Prediction Horizon", ["1 Day", "1 Week", "1 Month", "1 Year"])
            historical_range = st.selectbox("Historical Data Range", ["1 Month", "3 Months", "6 Months", "1 Year", "All"])
            model_selection = st.selectbox("Model", ["Linear Regression", "Polynomial Regression", "Decision Tree"])

        chart_type = st.selectbox("Chart Type", ["Line", "Candlestick", "Bar", "Histogram"])

        if model_selection == "Polynomial Regression":
            poly_degree = st.slider("Polynomial Degree", min_value=2, max_value=5, value=2, step=1)
        else:
            poly_degree = 2

        st.subheader("Technical Indicators")
        with st.expander("Trend Indicators", expanded=False):
            show_moving_average = st.checkbox("Show Moving Average", value=False)
            moving_average_window = st.slider("MA Window", 10, 200, 50, 5, disabled=not show_moving_average) if show_moving_average else 50

            show_ema = st.checkbox("Show EMA", value=False)
            ema_window = st.slider("EMA Window", 10, 200, 20, 5, disabled=not show_ema) if show_ema else 20

            show_ichimoku = st.checkbox("Show Ichimoku Cloud", value=False)
            tenkan_window, kijun_window, senkou_span_b_window = 9, 26, 52
            if show_ichimoku:
                tenkan_window = st.slider("Tenkan-sen Window", 5, 20, 9, 1, disabled=not show_ichimoku)
                kijun_window = st.slider("Kijun-sen Window", 10, 52, 26, 1, disabled=not show_ichimoku)
                senkou_span_b_window = st.slider("Senkou Span B Window", 26, 104, 52, 1, disabled=not show_ichimoku)

        with st.expander("Momentum Indicators", expanded=False):
            show_rsi = st.checkbox("Show RSI", value=False)
            rsi_window = st.slider("RSI Window", 7, 30, 14, 1, disabled=not show_rsi) if show_rsi else 14
            show_macd = st.checkbox("Show MACD", value=False)
            show_roc = st.checkbox("Show ROC", value=False)
            roc_window = st.slider("ROC Window", 5, 30, 14, 1, disabled=not show_roc) if show_roc else 14
            show_stoch = st.checkbox("Show Stochastic Oscillator", value=False)
            k_window, d_window = 14, 3
            if show_stoch:
                k_window = st.slider("Stochastic %K Window", 5, 30, 14, 1, disabled=not show_stoch)
                d_window = st.slider("Stochastic %D Window", 1, 10, 3, 1, disabled=not show_stoch)
            show_willr = st.checkbox("Show Williams %R", value=False)
            willr_window = st.slider("Williams %R Window", 5, 30, 14, 1, disabled=not show_willr) if show_willr else 14

        with st.expander("Volatility Indicators", expanded=False):
            show_bollinger_bands = st.checkbox("Show Bollinger Bands", value=False)
            bollinger_window, num_std = 20, 2
            if show_bollinger_bands:
                bollinger_window = st.slider("Bollinger Bands Window", 10, 100, 20, 5, disabled=not show_bollinger_bands)
                num_std = st.slider("Number of Standard Deviations", 1, 3, 2, 1, disabled=not show_bollinger_bands)
            show_atr = st.checkbox("Show ATR", value=False)
            atr_window = st.slider("ATR Window", 5, 30, 14, 1, disabled=not show_atr) if show_atr else 14

        with st.expander("Volume Indicators", expanded=False):
            show_obv = st.checkbox("Show OBV", value=False)
            show_cmf = st.checkbox("Show CMF", value=False)
            cmf_window = st.slider("CMF Window", 5, 50, 20, 1, disabled=not show_cmf) if show_cmf else 20

        show_residuals = st.checkbox("Show Residuals Plot", value=False)

        # Convert prediction horizon to days
        days_to_predict = {"1 Day": 1, "1 Week": 7, "1 Month": 30, "1 Year": 365}[prediction_horizon]
        historical_days = {"1 Month": 30, "3 Months": 90, "6 Months": 180, "1 Year": 365, "All": 730}[historical_range]

    df = fetch_data(coin_id, historical_days)
    market_cap = fetch_coin_info(coin_id)

    if df is not None and not df.empty and len(df) > 1:
        prediction_model = train_model(df, model_selection, poly_degree)
        if prediction_model is not None:
            last_time_index = df['TimeIndex'].iloc[-1]
            future_dates, future_prices = predict_future(prediction_model, last_time_index, days_to_predict, df, model_selection, poly_degree)

            fig = plot_data_plotly(df, future_dates, future_prices, chart_type,
                                    show_moving_average, moving_average_window,
                                    show_bollinger_bands, bollinger_window, num_std,
                                    show_rsi, rsi_window,
                                    show_macd,
                                    show_roc, roc_window,
                                    show_stoch, k_window, d_window,
                                    show_willr, willr_window,
                                    show_ema, ema_window,
                                    show_ichimoku, tenkan_window, kijun_window, senkou_span_b_window,
                                    show_atr, atr_window,
                                    show_obv,
                                    show_cmf, cmf_window)
            st.plotly_chart(fig, use_container_width=True)

            with st.container():
                st.subheader("Summary")
                summary_text = generate_summary(df, future_dates, future_prices, selected_coin, model_selection, market_cap) if market_cap else generate_summary(df, future_dates, future_prices, selected_coin, model_selection, 0)
                st.markdown(summary_text, unsafe_allow_html=True)

            with st.expander("Show Historical Data"):
                st.dataframe(df)

            if show_residuals:
                st.subheader("Residuals Plot")
                residuals_fig = plot_residuals(df, prediction_model, model_selection, poly_degree)
                if residuals_fig is not None:
                    st.pyplot(residuals_fig)
                else:
                    st.warning("Could not generate residuals plot.")
        else:
            st.warning("Model training failed. Check the model selection and data.")
    else:
        st.warning("Could not fetch price data, data is empty, or data has insufficient length. Check the coin selection and try again.")

    st.subheader("Fear & Greed Index")
    fng_index = fetch_fear_greed_index()
    if fng_index is not None:
        st.pyplot(plot_fear_greed_pie_chart(fng_index))

    st.subheader("Sentiment Analysis")
    average_sentiment = analyze_sentiment(selected_coin)
    display_sentiment_analysis(average_sentiment)
# --- INDICATOR EXPLANATIONS (Correctly Placed in col2) ---

with col2:  # LEFT-SIDE PANEL
    st.header("Indicator Explanations")

    if show_moving_average:
        with st.expander("Moving Average (MA)"): # Added expander
           st.markdown("""
                The Moving Average smooths price data to identify trend direction and potential support/resistance.

                **How to read it:**
                *   **Uptrend:** Price above the moving average.
                *   **Downtrend:** Price below the moving average.
                *   **Crossovers:** Shorter-term MA crossing above longer-term MA (bullish); and vice-versa.
                """)

    if show_ema:
        with st.expander("Exponential Moving Average (EMA)"): # Added expander
           st.markdown("""
                The Exponential Moving Average (EMA) is similar to the simple moving average (SMA), but it gives more weight to recent prices.

                **How to read it:** Same principles as the MA (uptrend, downtrend, crossovers).
                """)

    if show_bollinger_bands:
        with st.expander("Bollinger Bands"): # Added expander
           st.markdown("""
                Bollinger Bands are volatility bands placed above and below a moving average.  Volatility is based on the standard deviation, which changes as volatility increases and decreases.

                **How to read it:**
                * **Squeezes:**  When the bands are close together, it suggests low volatility and a potential breakout.
                * **Touches:** Prices tend to return to the middle of the bands.  Touches of the upper band *can* indicate overbought conditions, and touches of the lower band *can* indicate oversold conditions (but this is not always a reliable signal by itself!).
                * **Breakouts:** A move outside the bands suggests a strong trend.
                """)

    if show_rsi:
       with st.expander("Relative Strength Index (RSI)"): # Added expander
          st.markdown("""
                The RSI is a momentum oscillator that measures the speed and change of price movements.  It oscillates between 0 and 100.

                **How to read it:**
                * **Overbought:**  Traditionally, RSI above 70 is considered overbought (and a potential sell signal).
                * **Oversold:** RSI below 30 is considered oversold (and a potential buy signal).
                * **Divergences:**  If the price makes a new high but the RSI makes a lower high, this is *bearish divergence*.  If the price makes a new low, but the RSI makes a higher low, this is *bullish divergence*.
                """)

    if show_macd:
        with st.expander("Moving Average Convergence Divergence (MACD)"): # Added expander
           st.markdown("""
               The MACD is a trend-following momentum indicator that shows the relationship between two moving averages of a security’s price.

               **How to read it:**
                * **Crossovers:**  When the MACD line crosses above the signal line, it's a bullish signal.  When it crosses below, it's bearish.
                * **Histogram:**  The histogram represents the difference between the MACD and signal lines.  Increasing histogram bars can indicate strengthening momentum.
                * **Divergences:**  Similar to RSI, divergences between price and MACD can signal trend reversals.
                """)

    if show_roc:
        with st.expander("Rate of Change (ROC)"): # Added expander
           st.markdown("""
                The ROC is a momentum oscillator that measures the percentage change in price over a given period.

               **How to read it:**
                * **Positive ROC:** Indicates upward momentum.
                * **Negative ROC:** Indicates downward momentum.
                * **Crossovers of the zero line:** Can be used as buy/sell signals (but are often lagging).
                * **Extreme readings:** High ROC values can indicate overbought conditions, and low values oversold.
                """)

    if show_stoch:
      with st.expander("Stochastic Oscillator"): # Added expander
        st.markdown("""
                The Stochastic Oscillator is a momentum indicator that compares a security's closing price to its price range over a given period.

                **How to read it:**
                * **%K and %D:** The %K line is the main line, and %D is a moving average of %K.
                * **Overbought:** Readings above 80 are generally considered overbought.
                * **Oversold:** Readings below 20 are generally considered oversold.
                * **Crossovers:** %K crossing above %D is a bullish signal; %K crossing below %D is bearish.
                """)

    if show_willr:
      with st.expander("Williams %R"): # Added expander
        st.markdown("""
                Williams %R is a momentum indicator that measures overbought and oversold levels.  It's very similar to the Stochastic Oscillator, but it's scaled differently (-100 to 0).

                **How to read it:**
                * **Overbought:** Readings above -20 are generally considered overbought.
                * **Oversold:** Readings below -80 are generally considered oversold.
                """)
    if show_ichimoku:
       with st.expander("Ichimoku Cloud"): # Added expander
          st.markdown("""
              The Ichimoku Cloud is a comprehensive indicator that defines support and resistance, identifies trend direction, gauges momentum, and provides trading signals.

               **Components:**
               *   **Tenkan-sen (Conversion Line):** Short-term moving average.
               *   **Kijun-sen (Base Line):** Medium-term moving average.
               *   **Senkou Span A (Leading Span A):** Average of Tenkan-sen and Kijun-sen, plotted ahead.
               *   **Senkou Span B (Leading Span B):** Average of highest high/lowest low over a longer period, plotted ahead. The space between Span A and Span B is the "cloud".
               *   **Chikou Span (Lagging Span):** Closing price, plotted *behind* current price.

                **How to read it:**
                *   **Trend:** Price above cloud = uptrend; below cloud = downtrend; inside cloud = consolidating.
               * **Support/Resistance:** Cloud edges act as support/resistance.
                *   **Crossovers:** Tenkan-sen crossing above Kijun-sen (bullish).
                """)
    if show_atr:
      with st.expander("Average True Range (ATR)"): # Added expander
        st.markdown("""
               The ATR is a volatility indicator that measures the average range between the high and low of a price over a given period.

               **How to read it:**
               *   **High ATR:** High volatility.
               *   **Low ATR:** Low volatility.
               *   **Use for stop-loss orders:** Place a stop-loss a multiple of the ATR below/above entry.
                """)

    if show_obv:
       with st.expander("On Balance Volume (OBV)"):
          st.markdown("""
                OBV is a volume-based indicator measuring buying/selling pressure.

                **How to read it:**
                *  **Uptrend:** Price and OBV both increasing.
                * **Downtrend:** Price and OBV both decreasing.
                *   **Divergences:** Bullish divergence (OBV increasing, price decreasing) suggests potential reversal.
                """)
    if show_cmf:
       with st.expander("Chaikin Money Flow (CMF)"):
          st.markdown("""
               CMF is a volume-weighted average of accumulation/distribution over a period.

               **How to read it:**
                *  **Positive Values:** Accumulation (buying pressure).
                *   **Negative Values:** Distribution (selling pressure).
                """)
    if show_residuals:
        with st.expander("Residuals Plot"):
           st.markdown("""
               A residuals plot shows the difference between observed and predicted values.

               **How to read it:**
                *  **Randomly Scattered:** Indicates a good linear model fit.
                *  **Patterns:** Suggest a linear model is not a good fit.
                """)
