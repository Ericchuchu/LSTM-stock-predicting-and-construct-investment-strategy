"""Re-run the 2023 backtests with and without the same-bar stop rule.

The trade notebooks place orders at Open[t] but test the stop-loss with Close[t]
and a 5-day low/high that includes bar t, i.e. with information that is not
available at the open. This script rebuilds the notebook logic in plain Python
(no pandas / vectorbt needed) and compares four variants per model and ticker:

  as_is          notebook rule, no costs (reproduces the HTML reports in trade/)
  lagged         stop tested on Close[t-1] and the 5-day low/high up to t-1
  lagged_costs   lagged + Taiwan costs (0.1425% per side, 0.3% tax on sales)
  as_is_costs    notebook rule + Taiwan costs

Usage:  python analysis/lookahead_check.py
"""
import csv
import math
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TW_BUY = 0.001425
TW_SELL = 0.001425 + 0.003

# (model, ticker, entry threshold used in the corresponding trade notebook)
CASES = [
    ('GRUModel', '1605', 0), ('GRUModel', '2303', 0.005), ('GRUModel', '2409', 0.015),
    ('GRUModel', '2603', 0.035), ('GRUModel', '2610', 0), ('GRUModel', '2891', 0),
    ('GRUModel', '3037', 0.05), ('GRUModel', '3105', 0.03), ('GRUModel', '3481', 0.01),
    ('GRUModel', '00632R', 0.01), ('GRUModel', '00637L', 0.03),
]


def load(stock, model):
    path = os.path.join(ROOT, 'data', f'{stock} 2023.csv')
    rows = list(csv.DictReader(open(path, encoding='utf-8', errors='ignore')))
    O = [float(r['Open']) for r in rows]
    H = [float(r['High']) for r in rows]
    L = [float(r['Low']) for r in rows]
    C = [float(r['Close']) for r in rows]
    n = len(rows)
    minlow = [None] * n
    maxhigh = [None] * n
    for i in range(4, n):
        minlow[i] = min(L[i - 4:i + 1])
        maxhigh[i] = max(H[i - 4:i + 1])
    path = os.path.join(ROOT, 'predicted_result', f'{model} {stock} 2023 predicted prices.csv')
    pred = list(csv.DictReader(open(path, encoding='utf-8')))
    p = [float(r['predicted_price']) for r in pred]
    t = [float(r['true_price']) for r in pred]
    sl = slice(25, n - 1)  # same alignment as the notebooks
    O, C, minlow, maxhigh = O[sl], C[sl], minlow[sl], maxhigh[sl]
    assert len(O) == len(p), (stock, len(O), len(p))
    return O, C, minlow, maxhigh, p, t


def conditions(p, t, thr):
    n = len(p)
    long_c = [0] * n
    short_c = [0] * n
    for i in range(1, n):
        chg = (p[i] - t[i - 1]) / t[i - 1]
        long_c[i] = int(t[i - 1] <= p[i] and chg >= thr)
        short_c[i] = int(t[i - 1] > p[i] and abs(chg) >= thr)
    return long_c, short_c


def signals(C, minlow, maxhigh, long_c, short_c, lagged):
    n = len(C)
    s_long = [0] * n
    s_short = [0] * n
    pos, stop = 0, 0
    for i in range(n):
        if pos == 0 and long_c[i] == 1:
            s_long[i], pos = 1, 1
            stop = minlow[i - 1] if lagged else minlow[i]
            if lagged:
                continue
        if pos == 1:
            ref = C[i - 1] if lagged else C[i]
            if ref <= stop or short_c[i] == 1:
                s_long[i], stop, pos = -1, 0, 0
    pos, stop = 0, 0
    for i in range(n):
        if pos == 0 and short_c[i] == 1:
            s_short[i], pos = -1, -1
            stop = maxhigh[i - 1] if lagged else maxhigh[i]
            if lagged:
                continue
        if pos == -1:
            ref = C[i - 1] if lagged else C[i]
            if ref >= stop or long_c[i] == 1:
                s_short[i], stop, pos = 1, 0, 0
    return ([x == 1 for x in s_long], [x == -1 for x in s_long],
            [x == -1 for x in s_short], [x == 1 for x in s_short])


def simulate(P, entries_long, exits_long, entries_short, exits_short, fee_buy=0.0, fee_sell=0.0):
    cash, shares, values = 100.0, 0.0, []
    for i, px in enumerate(P):
        def close_position():
            nonlocal cash, shares
            if shares > 0:
                cash += shares * px * (1 - fee_sell)
            elif shares < 0:
                cash -= -shares * px * (1 + fee_buy)
            shares = 0.0
        if entries_long[i] and shares <= 0:
            close_position()
            q = cash / (px * (1 + fee_buy))
            cash -= q * px * (1 + fee_buy)
            shares = q
        elif entries_short[i] and shares >= 0:
            close_position()
            q = cash / px / (1 + fee_sell)
            cash += q * px * (1 - fee_sell)
            shares = -q
        elif exits_long[i] and shares > 0:
            close_position()
        elif exits_short[i] and shares < 0:
            close_position()
        values.append(cash + shares * px)
    rets = [0.0] + [values[i] / values[i - 1] - 1 for i in range(1, len(values))]
    mean = sum(rets) / len(rets)
    sd = math.sqrt(sum((r - mean) ** 2 for r in rets) / (len(rets) - 1))
    sharpe = mean / sd * math.sqrt(252) if sd else float('nan')
    return (values[-1] / 100 - 1) * 100, sharpe


def main():
    print('| Model | Ticker | As reported | Lagged stop | Lagged stop + costs | As reported + costs |')
    print('|---|---|---|---|---|---|')
    for model, stock, thr in CASES:
        O, C, mn, mx, p, t = load(stock, model)
        lc, sc = conditions(p, t, thr)
        as_is = signals(C, mn, mx, lc, sc, lagged=False)
        lagged = signals(C, mn, mx, lc, sc, lagged=True)
        cells = [
            simulate(O, *as_is),
            simulate(O, *lagged),
            simulate(O, *lagged, TW_BUY, TW_SELL),
            simulate(O, *as_is, TW_BUY, TW_SELL),
        ]
        body = ' | '.join(f'{tr:.1f}% ({sh:.2f})' for tr, sh in cells)
        print(f'| {model} | {stock} | {body} |')
    print('\nCells show total return 2023-02-17 to 2023-12-28 and, in brackets, the Sharpe ratio annualized with 252 days.')


if __name__ == '__main__':
    main()
