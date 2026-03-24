import pandas as pd
import yfinance as yf
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import streamlit as st 
import os

st.set_page_config(page_title="Stock Tracker", page_icon="📈", layout="wide")


if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "username" not in st.session_state:
    st.session_state.username = ""

def check_login():
    if st.session_state.logged_in:
        return True

    st.title("Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        app_username = os.environ.get("APP_USERNAME")
        app_password = os.environ.get("APP_PASSWORD")

        if username == app_username and password == app_password:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.rerun()
        else:
            st.error("Invalid username or password")

    return False

if not check_login():
    st.stop()

with st.sidebar:
    st.write(f"Logged in as: {st.session_state.username}")
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.rerun()



def compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    close = pd.to_numeric(close, errors="coerce")
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=window, min_periods=window).mean()
    avg_loss = loss.rolling(window=window, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, float("nan"))
    rsi = 100 - (100 / (1 + rs))
    return pd.to_numeric(rsi, errors="coerce")

def build_features(data: pd.DataFrame) -> pd.DataFrame:
    df = data.copy()
    df["Return_1d"] = df["Close"].pct_change()
    df["Return_5d"] = df["Close"].pct_change(5)
    df["Return_10d"] = df["Close"].pct_change(10)
    df["MA20"] = df["Close"].rolling(20).mean()
    df["MA50"] = df["Close"].rolling(50).mean()
    df["MA200"] = df["Close"].rolling(200).mean()
    df["Momentum_5"] = df["Close"] / df["Close"].shift(5) - 1
    df["Momentum_20"] = df["Close"] / df["Close"].shift(20) - 1
    df["Volatility_10"] = df["Return_1d"].rolling(10).std()
    df["Volatility_30"] = df["Return_1d"].rolling(30).std()
    df["Trend_MA20"] = df["Close"] / df["MA20"] - 1
    df["Trend_MA50"] = df["Close"] / df["MA50"] - 1
    rolling_mean_20 = df["Close"].rolling(20).mean()
    rolling_std_20 = df["Close"].rolling(20).std()
    df["Z_Score_20"] = (df["Close"] - rolling_mean_20) / rolling_std_20.replace(0, float("nan"))
    df["Volume_Change"] = df["Volume"].pct_change()
    df["Volume_MA_Ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()
    df["RSI_14"] = compute_rsi(df["Close"], 14)
    df["Return_Lag1"] = df["Return_1d"].shift(1)
    df["Return_Lag2"] = df["Return_1d"].shift(2)
    df["Return_Lag3"] = df["Return_1d"].shift(3)
    future_return_5d = df["Close"].shift(-5) / df["Close"] -1
    df["Target"] = (future_return_5d > 0.01).astype(int)
    return df

@st.cache_data(ttl=3600, show_spinner=False)
def download_ticker(ticker: str) -> pd.DataFrame:
    data = yf.download(ticker, period="3y", auto_adjust=False, progress=False)
    if data.empty:
        return pd.DataFrame()
    data = data.reset_index()
    data.columns = [c[0] if isinstance(c, tuple) else c for c in data.columns]
    return data

def process_ticker(ticker: str):
    data = download_ticker(ticker)
    if data.empty:
        return None

    required = {"Date", "Open", "High", "Low", "Close", "Volume"}
    if not required.issubset(data.columns):
        return None

    data["Ticker"] = ticker
    data = build_features(data)

    feature_cols = [
        "Return_1d", "Return_5d", "Return_10d",
        "Momentum_5", "Momentum_20",
        "Volatility_10", "Volatility_30",
        "Trend_MA20", "Trend_MA50",
        "Z_Score_20",
        "Volume_Change", "Volume_MA_Ratio",
        "RSI_14",
        "Return_Lag1", "Return_Lag2", "Return_Lag3",
    ]

    for col in feature_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data["Target"] = pd.to_numeric(data["Target"], errors="coerce")

    model_data = data.dropna(subset=feature_cols + ["Target"]).copy()
    if len(model_data) < 180:
        return None

    X = model_data[feature_cols]
    y = model_data["Target"]

    split_idx = int(len(model_data) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    if len(X_train) < 100 or len(X_test) < 20:
        return None

    best_model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        eval_metric="logloss",
        n_jobs=-1,
    )
    best_model.fit(X_train, y_train)

    test_preds = best_model.predict(X_test)
    accuracy_pct = accuracy_score(y_test, test_preds) * 100
    precision_pct = precision_score(y_test, test_preds, zero_division=0) * 100
    recall_pct = recall_score(y_test, test_preds, zero_division=0) * 100
    f1_pct = f1_score(y_test, test_preds, zero_division=0) * 100

    latest_features = X.tail(1)
    latest_price = float(model_data["Close"].iloc[-1])
    prob_up = float(best_model.predict_proba(latest_features)[0][1]) * 100

    latest_return_1d = float(model_data["Return_1d"].iloc[-1])
    latest_volatility_30 = float(model_data["Volatility_30"].iloc[-1])
    latest_rsi = float(model_data["RSI_14"].iloc[-1])
    ma50 = float(model_data["MA50"].iloc[-1])
    ma200 = float(model_data["MA200"].iloc[-1])

    feature_importance = pd.Series(best_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    top_features = ", ".join(feature_importance.head(3).index.tolist())

    return {
        "Ticker": ticker,
        "Latest Price ($)": latest_price,
        "MA50 ($)": ma50,
        "MA200 ($)": ma200,
        "Daily Return": latest_return_1d,
        "Volatility": latest_volatility_30,
        "Predicted Prob Up (%)": prob_up,
        "Accuracy (%)": accuracy_pct,
        "Precision (%)": precision_pct,
        "Recall (%)": recall_pct,
        "F1 Score (%)": f1_pct,
        "RSI_14": latest_rsi,
        "Top Features": top_features,
    }

def score_row(row: pd.Series):
    vals = [
        row["Predicted Prob Up (%)"],
        row["Accuracy (%)"],
        row["Precision (%)"],
        row["F1 Score (%)"],
        row["RSI_14"],
    ]
    if any(pd.isna(v) for v in vals):
        return None
    rsi = row["RSI_14"]
    bonus = 10 if rsi < 30 else 5 if rsi < 40 else -10 if rsi > 70 else -5 if rsi > 60 else 0
    return round(
        0.40 * row["Predicted Prob Up (%)"]
        + 0.20 * row["Accuracy (%)"]
        + 0.25 * row["Precision (%)"]
        + 0.15 * row["F1 Score (%)"]
        + bonus,
        1,
    )

st.title("📈 Stock Tracker")
st.caption("Disclaimer: These are not guarenteed, only modelled best guess predictions")

with st.sidebar:
    st.header("Inputs")
    default_tickers = "MSFT, GOOGL, TSCO.L, SHEL.L"
    tickers_text = st.text_area("Tickers (comma-separated)", value=default_tickers, height=140)
    run_clicked = st.button("Run model", type="primary")
    st.markdown("Examples: `AAPL`, `MSFT`, `SHEL.L`, `TSCO.L`")

if run_clicked:
    tickers = [t.strip().upper() for t in tickers_text.split(",") if t.strip()]
    if not tickers:
        st.error("Please enter at least one ticker.")
    else:
        results = []
        errors = []
        progress = st.progress(0)

        for i, ticker in enumerate(tickers, start=1):
            summary = process_ticker(ticker)
            if summary is None:
                errors.append(ticker)
            else:
                results.append(summary)
            progress.progress(i / len(tickers))

        if not results:
            st.warning("No usable results were generated.")
        else:
            df = pd.DataFrame(results)
            df["Score"] = df.apply(score_row, axis=1)
            df["Rating"] = df["Score"].apply(
                lambda s: ""
                if pd.isna(s)
                else "Strong Buy"
                if s >= 70
                else "Watchlist"
                if s >= 60
                else "Neutral"
                if s >= 50
                else "Avoid / Consider Shorting"
            )
            df = df.sort_values(["Score", "Predicted Prob Up (%)"], ascending=False)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Tickers modelled", len(df))
            c2.metric("Avg prob. up", f"{df['Predicted Prob Up (%)'].mean():.1f}%")
            c3.metric("Avg accuracy", f"{df['Accuracy (%)'].mean():.1f}%")
            c4.metric("Strong buys", int((df["Rating"] == "Strong Buy").sum()))

            st.subheader("Ranked results")
            st.dataframe(df, use_container_width=True)

            st.subheader("Top ideas")
            top = df.head(10)[["Ticker", "Predicted Prob Up (%)", "Accuracy (%)", "Score", "Rating"]]
            st.dataframe(top, use_container_width=True)

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button("Download results as CSV", csv, "stock_tracker_results.csv", "text/csv")

        if errors:
            st.info("Skipped tickers: " + ", ".join(errors))
else:
    st.info("Enter tickers in the sidebar and click Run model.")
