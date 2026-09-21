# Recurrent Networks for Taiwan Stock Forecasting and Trading

A 2024 student project that trains recurrent neural networks (GRU, LSTM, ConvLSTM, CNN+RNN and residual variants) to predict the next day's closing price of Taiwan stocks from price, technical-indicator and institutional-flow features, and turns the forecasts into a daily long/short trading rule that is backtested on 2023.

Write-up: [Deep Learning-Driven CTA Investment Decisions: Stock Price Prediction and Portfolio Management](https://medium.com/@eric07310115/deep-learning-driven-cta-investment-decisions-stock-price-prediction-and-portfolio-management-9b676da6d2f7) (Medium). Follow-up across the whole market: [tw-stock-rnn-selection](https://github.com/Ericchuchu/tw-stock-rnn-selection).

**Status.** The repository is kept as a record of the project. A later review found that the backtests overstate performance, mainly because the stop-loss rule uses same-day information. The [retrospective](#retrospective-what-the-backtests-actually-show) quantifies this; the figures in the HTML reports should not be read as achievable returns.

## Data

| Item | Detail |
|---|---|
| Market | Taiwan stocks and ETFs, daily bars; 27 tickers in `data/` (for example 2303 UMC, 2603 Evergreen Marine, 3037 Unimicron, 3105 WIN Semiconductors, 00632R, 00637L) |
| Period | 2016-01-04 to 2022-12-30 for training and validation (chronological 80/20 split), calendar 2023 as the test year |
| Sources | OHLCV from the `twstock` package; institutional buy/sell volumes from the FinMind API (`get_data.py`) |
| Features (20) | Volume, open, high, low, close, change; Keltner Channel (EMA20 ± 1.5 ATR20); Bollinger Bands (20, 2); 5-day moving average; 10-day momentum; buy and sell volumes of foreign investors, investment trusts, dealers (proprietary) and dealers (hedging) |
| Input / target | The previous 5 days × 20 features, min-max scaled → next day's scaled close |

## Models

All models are in `model/`; one model is trained per ticker.

| Class | Architecture |
|---|---|
| `GRUModel` | GRU 256 → GRU 128 → dense 16 → 1, dropout 0.3 |
| `GRUModel2` | GRU 256 → 128 → 64 → 1, dropout 0.1 |
| `LSTMModel` | LSTM 256 → 128 → dense 16 → 1, dropout 0.3 |
| `ResGRU` | Hand-written GRU cells (128, 64, 32) with a residual projection on the first layer |
| `ResLSTM` | Hand-written LSTM cells (128, 64), optional attention branch |
| `ConvLSTM` | LSTM cells with 1-D convolutional gates on a 3-channel input (prices, indicators, institutional flows) |
| `CNN_RNN` | Three parallel dilated-convolution branches, each followed by a bidirectional GRU; outputs summed and passed to a final GRU and linear layer |

Training (`run_neural_network*.py`): MSE loss, Adam with weight decay 0.005, learning rate 4e-3 (5e-4 for the convolutional models), batch size 30, up to 300 epochs, best-validation checkpoint kept.

## Trading rule and backtest

- **Signal.** Go long when the predicted close for day *t* is at least `threshold` above the close of day *t−1*; go short in the mirror case. The threshold per ticker (0 to 5%) was chosen with an Optuna NSGA-II search (`trade/trade_GRU_test_hyperparameter.ipynb`).
- **Exit.** Opposite signal, or a stop at the 5-day low (long) or 5-day high (short).
- **Engine.** vectorbt `Portfolio.from_signals`, orders filled at the open, all-in sizing, long and short, no commission, tax or slippage.
- **Window.** 2023-02-17 to 2023-12-28 (213 bars). One HTML report per model and ticker is stored in `trade/`.

## Retrospective: what the backtests actually show

Reviewing the code in 2026 turned up four problems. They are documented here instead of being silently fixed, because the stored reports and the article are based on them.

1. **Same-bar look-ahead in the stop rule.** Orders are filled at `Open[t]`, but the stop is tested with `Close[t]` against a 5-day low/high that includes day *t* (`create_signals.py`, and the signal cell of every notebook in `trade/`). The rule therefore avoids entering, and exits at the open, on exactly the days that go on to close badly.
2. **No trading costs.** `fees=0`, although a Taiwan round trip costs about 0.585% (0.1425% commission per side plus 0.3% tax on sales) and the strategy makes roughly 15 to 75 trades per ticker in ten months.
3. **Threshold tuned on the test window.** The Optuna search maximizes Sharpe ratio and return on the same 2023 data that is reported.
4. **Forecasts do not beat persistence.** For all 31 prediction files in `predicted_result/`, the model's MSE is 1.7 to 56 times that of the naive forecast "tomorrow's close = today's close", and mean directional accuracy is 50.7%.

`analysis/lookahead_check.py` rebuilds the notebook logic in plain Python, reproduces the reported returns of all eleven GRU reports, and re-runs them with the stop tested only on information available before the open. Total return for 2023-02-17 to 2023-12-28, Sharpe ratio (252-day) in brackets:

| Ticker | As reported | Lagged stop | Lagged stop + costs |
|---|---|---|---|
| 1605 | 100.9% (3.17) | 58.4% (2.05) | 38.0% (1.48) |
| 2303 | 82.5% (3.66) | 28.0% (1.49) | −11.3% (−0.57) |
| 2409 | 43.5% (2.04) | −16.7% (−0.67) | −35.1% (−1.77) |
| 2603 | 167.9% (2.55) | 132.8% (2.24) | 111.3% (2.02) |
| 2610 | 68.9% (2.71) | 17.3% (0.81) | 0.9% (0.18) |
| 2891 | 46.9% (3.36) | 6.6% (0.59) | −25.8% (−2.33) |
| 3037 | 233.2% (4.37) | 73.7% (2.08) | 55.5% (1.70) |
| 3105 | 126.6% (2.89) | −15.2% (−0.43) | −33.4% (−1.28) |
| 3481 | 47.6% (2.27) | −36.1% (−1.92) | −50.5% (−3.04) |
| 00632R | 14.0% (1.51) | 5.3% (0.63) | −11.5% (−1.25) |
| 00637L | 59.0% (2.18) | 21.4% (1.03) | −4.6% (−0.09) |
| **Median** | **68.9%** | **17.3%** | **−11.3%** |

The Sharpe ratios printed in the HTML reports are about 20% higher than the "as reported" column because vectorbt annualizes with 365 periods per year. The lagged-stop figures still use thresholds tuned on the same window, so they remain optimistic.

Lessons carried into later work: decide with information up to *t−1* and execute at *t*; include costs from the first backtest; keep a final period untouched by any tuning; compare every forecast with a naive baseline before building a strategy on it.

## Repository layout

| Path | Purpose |
|---|---|
| `get_data.py` | Download prices (`twstock`) and institutional flows (FinMind) |
| `data_preprocess.py`, `data_preprocess_3channels.py` | Indicators (TA-Lib), scaling, 5-day windows; the 3-channel version feeds ConvLSTM and CNN+RNN |
| `model/` | Model definitions |
| `train_test_valid.py`, `run_neural_network*.py` | Training loops and entry points (`_convlstm`, `_cnnrnn`, `_selected_stock` variants) |
| `test_model*.py` | Write 2023 predictions to `predicted_result/` |
| `trade/` | Backtest notebooks, exported HTML reports, Optuna trial tables (`trials/`) |
| `combined_portfolio.ipynb` | Averages the per-ticker ResGRU backtests |
| `create_signals.py` | Signal generation on new data with a trained model |
| `analysis/lookahead_check.py` | Dependency-free re-run of the backtests, see the retrospective |
| `2023籌碼選股/`, `selected_stock/` | Ticker screening by trading volume and institutional holdings |
| `data/`, `model_state/`, `predicted_result/` | Input data, trained weights, stored predictions |

## Running

```bash
pip install -r requirements.txt    # TA-Lib also needs its C library installed
python analysis/lookahead_check.py # runs as is, standard library only
```

The training and backtest scripts were written on Windows and contain absolute paths (`C:\Users\...`) and backslash separators; adjust the paths at the top of each script before running. `data_preprocess.py` loads ticker 3105 at import time, so edit the file name there to train another ticker. The source file of the `GatedMultiModel` checkpoints in `model_state/` is not included.

## License

MIT, see [LICENSE](LICENSE). Not investment advice.
