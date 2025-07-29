# 1. 필요한 라이브러리 불러오기
import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import curve_fit
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# 2. 비트코인 데이터 다운로드 (최근 90일)
end_date = datetime.today()
start_date = end_date - timedelta(days=90)
# 최근 90일 간 비트코인 가격 데이터를 다운로드한다.
# yfinance 최신 버전에서는 컬럼이 MultiIndex 형태이므로 첫 번째 레벨만 사용한다.
btc = yf.download('BTC-USD', start=start_date, end=end_date, interval='1d')
btc = btc.droplevel(1, axis=1)
btc.dropna(inplace=True)

# 3. 맥스웰-볼츠만 분포 함수 정의
def mb_distribution(energy, beta, A):
    return A * np.exp(-beta * energy)

# 4. 각 일자별 MB 분포 피팅하여 베타(beta) 추정
mb_fits = []

for idx, row in btc.iterrows():
    prices = np.array([row['Open'], row['High'], row['Low'], row['Close']])
    volumes = np.full(4, row['Volume'] / 4)  # OHLC에 균등 분배
    energy = prices - prices.min()  # 최소값 기준 에너지 정규화

    try:
        popt, _ = curve_fit(mb_distribution, energy, volumes, p0=(0.01, row['Volume']))
        beta, A = popt
    except:
        beta, A = np.nan, np.nan

    mb_fits.append({'Date': idx, 'Beta': beta, 'A': A})

# 5. 결과 병합
mb_df = pd.DataFrame(mb_fits).set_index('Date')
btc = btc.merge(mb_df, left_index=True, right_index=True)

# 6. 베타 변화율 기반 신호 생성
btc['Beta_Change'] = btc['Beta'].diff()
btc['Signal'] = 0
btc.loc[btc['Beta_Change'] < 0, 'Signal'] = 1   # 매수 신호 (시장 온도 상승)
btc.loc[btc['Beta_Change'] > 0, 'Signal'] = -1  # 매도 신호 (시장 냉각)

# 예측 신호의 정확도 측정: 다음 날 종가 변동 방향과 비교
btc['Next_Close_Change'] = btc['Close'].diff().shift(-1)
btc['Actual'] = 0
btc.loc[btc['Next_Close_Change'] > 0, 'Actual'] = 1
btc.loc[btc['Next_Close_Change'] < 0, 'Actual'] = -1
accuracy = (btc['Signal'] == btc['Actual']).mean()
print(f'Signal accuracy: {accuracy:.2%}')

# 7. 결과 시각화
plt.figure(figsize=(14, 6))
plt.plot(btc.index, btc['Close'], label='BTC Close Price', color='black')
plt.plot(btc.index, btc['Beta'], label='Beta (Market Temperature)', color='purple')

# 신호 표시
buy_signals = btc[btc['Signal'] == 1]
sell_signals = btc[btc['Signal'] == -1]
plt.scatter(buy_signals.index, buy_signals['Close'], marker='^', color='green', label='Buy Signal', s=100)
plt.scatter(sell_signals.index, sell_signals['Close'], marker='v', color='red', label='Sell Signal', s=100)

plt.title('Bitcoin Price Prediction with Maxwell-Boltzmann Model')
plt.xlabel('Date')
plt.ylabel('Price / Beta')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('btc_beta_plot.png')

# 8. 마지막 30일 데이터 출력
print(btc.tail(30)[['Close', 'Beta', 'Beta_Change', 'Signal']])
