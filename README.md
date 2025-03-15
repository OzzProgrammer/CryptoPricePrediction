## Crypto Price Prediction and Technical Analysis App

This Streamlit app provides a comprehensive cryptocurrency price prediction and technical analysis platform. It empowers users to visualize historical price data, explore various technical indicators, and generate future price predictions using machine learning models.

**Important Disclaimer: This app is for EDUCATIONAL PURPOSES ONLY and should NOT be considered financial advice. Trading cryptocurrencies involves significant risk, and you could lose money. Consult with a qualified financial advisor before making any investment decisions.**

**Key Features:**

*   **Data Visualization:** Interactive charts (Line, Candlestick, Bar, Histogram) for visualizing historical cryptocurrency price data fetched from the CoinGecko API.
*   **Technical Indicators:** A wide range of technical indicators (Moving Average, EMA, Bollinger Bands, RSI, MACD, ROC, Stochastic Oscillator, Williams %R, Ichimoku Cloud, ATR, OBV, CMF) to analyze price trends, momentum, volatility, and volume.
*   **Price Prediction:** Future price predictions using Linear Regression, Polynomial Regression, and Decision Tree models. Users can adjust the prediction horizon and historical data range.
*   **Sentiment Analysis:** Integration of sentiment analysis based on example news headlines to gauge market sentiment.
*   **Fear & Greed Index:** Display of the Fear & Greed Index from Alternative.me to provide a broad market sentiment overview.
*   **Model Evaluation:** Residuals plot for model evaluation.
*   **User-Friendly Interface:** A clean and intuitive Streamlit interface with customizable settings via sidebar.
*   **Informative Explanations:** Detailed descriptions of each technical indicator and its interpretation.

**Intended Audience:**

This app is designed for cryptocurrency enthusiasts, traders, and investors who seek to:

*   Gain insights into historical price movements.
*   Apply technical analysis techniques.
*   Explore potential future price trends.
*   **Understand the concepts of technical analysis and price prediction models. Do not use this app for actual investment decisions.**

**Technologies Used:**

*   **Python:** The core programming language.
*   **Streamlit:** For building the interactive web application.
*   **Requests:** For fetching data from the CoinGecko and Alternative.me APIs.
*   **Pandas:** For data manipulation and analysis.
*   **NumPy:** For numerical computations.
*   **Plotly:** For creating interactive charts and visualizations.
*   **NLTK (VADER):** For sentiment analysis.
*   **Scikit-learn (sklearn):** For machine learning models (Linear Regression, Polynomial Regression, Decision Tree).
*   **Matplotlib:** For Residuals plot figure.

**How to Use:**

1.  Select a cryptocurrency from the dropdown menu.
2.  Adjust the prediction settings (horizon, historical data range, model) in the sidebar.
3.  Choose a chart type and select the desired technical indicators.
4.  Review the generated chart, summary, and additional information to inform your learning about technical analysis.
5.  **Remember, the results are for educational purposes only.**

This project serves as a valuable tool for understanding and predicting cryptocurrency price movements by combining data visualization, technical analysis, and machine learning techniques. **However, it is crucial to understand that past performance is not indicative of future results, and the predictions generated by this app should not be used as the sole basis for making financial decisions.**

Demo : https://ozzprogrammer-cryptopriceprediction-crypto-predictor11-p2vxog.streamlit.app/
