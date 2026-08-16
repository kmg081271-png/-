import os
import requests
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier

# 분석할 종목 리스트 (필요에 따라 변경 가능)
TICKERS = [""SOXX"]

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

def compute_technical_indicators(df):
    """이평선, RSI, MACD 등 기술적 보조지표 계산"""
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA60'] = df['Close'].rolling(window=60).mean()
    df['Dist_MA20'] = (df['Close'] - df['MA20']) / df['MA20']
    
    # RSI (14)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-9)
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD (12, 26, 9)
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = macd - signal
    
    # 거래량 및 변동폭
    df['Vol_Change'] = df['Volume'].pct_change()
    df['Daily_Range'] = (df['High'] - df['Low']) / df['Close']
    df['Return_1D'] = df['Close'].pct_change()

    # 다음 날 상승(1) / 하락(0)
    df['Target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)
    return df.dropna()

def predict_next_day_prob(ticker):
    """과거 5년치 데이터 기반 다음 날 상승 확률 계산"""
    try:
        df = yf.download(ticker, period="5y", interval="1d", progress=False)
        if len(df) < 100:
            return None
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
            
        data = compute_technical_indicators(df.copy())
        feature_cols = ['Dist_MA20', 'RSI', 'MACD_Hist', 'Vol_Change', 'Daily_Range', 'Return_1D']
        
        X = data[feature_cols].iloc[:-1]
        y = data['Target'].iloc[:-1]
        
        model = RandomForestClassifier(n_estimators=150, max_depth=5, random_state=42)
        model.fit(X, y)
        
        latest_features = data[feature_cols].iloc[[-1]]
        prob_up = float(model.predict_proba(latest_features)[0][1] * 100)
        prob_down = 100.0 - prob_up
        
        latest_close = float(df['Close'].iloc[-1])
        rsi_val = float(data['RSI'].iloc[-1])
        
        return {
            "ticker": ticker,
            "close": latest_close,
            "prob_up": prob_up,
            "prob_down": prob_down,
            "rsi": rsi_val
        }
    except Exception as e:
        print(f"Error analyzing {ticker}: {e}")
        return None

def send_telegram_msg(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "Markdown"}
    requests.post(url, json=payload)

if __name__ == "__main__":
    report = ["📊 *미 증시 마감 후 내일 장 확률 분석 리포트* 📊\n"]
    
    for ticker in TICKERS:
        res = predict_next_day_prob(ticker)
        if res:
            signal = "🟢 상승 우세" if res['prob_up'] >= 55 else "🔴 하락 우세" if res['prob_down'] >= 55 else "⚪ 횡보/혼조"
            report.append(
                f"*{res['ticker']}* (종가: ${res['close']:.2f})\n"
                f"• 다음 장 상승 확률: *{res['prob_up']:.1f}%* / 하락: *{res['prob_down']:.1f}%*\n"
                f"• 단기 신호: {signal} (RSI: {res['rsi']:.1f})\n"
            )
            
    final_message = "\n".join(report)
    send_telegram_msg(final_message)
