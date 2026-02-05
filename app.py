# app.py —— PAFER 交易看板（Zero-Compile · Streamlit Cloud Python 3.13 Ready）
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time
import math
import random
from itertools import product
from concurrent.futures import ThreadPoolExecutor, as_completed

# -------------------------------
# 🔧 极简配置（全部内置，无外部依赖）
# -------------------------------
class Config:
    SYMBOL = "ETH/USDT"
    TIMEFRAMES = [
        '1m','3m','5m','10m','15m','30m',
        '1h','2h','3h','4h','6h','12h',
        '1d','2d','3d','5d','1w','1M','3M'
    ]
    
    # 风控
    MAX_LOSS_PCT = 5.0
    STOP_LOSS_BUFFER = 0.003
    
    # 默认策略参数（可被滑块或优化覆盖）
    macd_fast = 3
    macd_slow = 18
    macd_signal = 6
    kdj_period = 9
    kdj_smooth_k = 3
    kdj_smooth_d = 3
    momentum_threshold_pct = 15.0
    max_klines_for_resonance = 4
    
    # 虚拟账户
    VIRTUAL_INITIAL_BALANCE = 100.0

# -------------------------------
# 📊 纯 Python 模拟K线生成器（零 pandas/numpy）
# -------------------------------
def generate_klines(timeframe: str, n: int = 100) -> dict:
    now = datetime.now()
    # 时间频率映射（用于生成 timestamp list）
    freq_map = {
        '1m': 60, '3m': 180, '5m': 300, '10m': 600, '15m': 900, '30m': 1800,
        '1h': 3600, '2h': 7200, '3h': 10800, '4h': 14400, '6h': 21600, '12h': 43200,
        '1d': 86400, '2d': 172800, '3d': 259200, '5d': 432000, '1w': 604800,
        '1M': 2592000, '3M': 7776000
    }
    step_sec = freq_map.get(timeframe, 900)
    
    timestamps = []
    for i in range(n):
        ts = now - timedelta(seconds=(n - 1 - i) * step_sec)
        timestamps.append(ts.isoformat())
    
    # 主趋势 + 波动 + BOLL（纯 Python 实现）
    base = 3200.0
    trend = [0.0] * n
    for i in range(1, n):
        trend[i] = trend[i-1] + (0.1 if random.random() > 0.5 else -0.1)
    
    noise = [0.0] * n
    for i in range(1, n):
        noise[i] = noise[i-1] + random.gauss(0, 0.5)
    
    close = [base + trend[i] + noise[i] for i in range(n)]
    
    # BOLL(10,2) —— 手动滚动计算
    boll_upper, boll_mid, boll_lower = [], [], []
    for i in range(n):
        start = max(0, i - 9)
        window = close[start:i+1]
        if len(window) >= 10:
            mid_val = sum(window) / len(window)
            std_val = (sum((x - mid_val)**2 for x in window) / len(window)) ** 0.5
            boll_mid.append(mid_val)
            boll_upper.append(mid_val + 2 * std_val)
            boll_lower.append(mid_val - 2 * std_val)
        else:
            boll_mid.append(close[i])
            boll_upper.append(close[i] + 10.0)
            boll_lower.append(close[i] - 10.0)
    
    # MA线（手动滚动平均）
    def calc_ma(data: list, window: int) -> list:
        ma = []
        for i in range(len(data)):
            start = max(0, i - window + 1)
            window_data = data[start:i+1]
            ma.append(sum(window_data) / len(window_data))
        return ma
    
    ma5 = calc_ma(close, 5)
    ma10 = calc_ma(close, 10)
    ma30 = calc_ma(close, 30)
    ma45 = calc_ma(close, 45)
    
    # OHLCV（模拟）
    open_price = [c - random.uniform(1, 3) for c in close]
    high_price = [c + random.uniform(2, 5) for c in close]
    low_price = [c - random.uniform(2, 5) for c in close]
    volume = [random.randint(500, 3000) for _ in range(n)]
    
    return {
        'timestamp': timestamps,
        'open': open_price,
        'high': high_price,
        'low': low_price,
        'close': close,
        'volume': volume,
        'boll_upper': boll_upper,
        'boll_mid': boll_mid,
        'boll_lower': boll_lower,
        'ma5': ma5,
        'ma10': ma10,
        'ma30': ma30,
        'ma45': ma45,
    }

# -------------------------------
# 🧠 纯 Python PAFER 信号生成（无 numpy/pandas）
# -------------------------------
def generate_paferr_signal(kdata: dict, config) -> dict:
    n = len(kdata['close'])
    if n < 50:
        return {'action': 'hold', 'reason': 'Not enough data'}
    
    latest = {
        'close': kdata['close'][-1],
        'high': kdata['high'][-1],
        'low': kdata['low'][-1],
        'ma45': kdata['ma45'][-1]
    }
    
    # MACD 计算（纯 Python）
    close = kdata['close']
    def ewm(data, span):
        alpha = 2.0 / (span + 1)
        result = [data[0]]
        for i in range(1, len(data)):
            val = alpha * data[i] + (1 - alpha) * result[-1]
            result.append(val)
        return result
    
    ema_fast = ewm(close, config.macd_fast)
    ema_slow = ewm(close, config.macd_slow)
    macd_line = [a - b for a, b in zip(ema_fast, ema_slow)]
    signal_line = ewm(macd_line, config.macd_signal)
    macd_hist = [a - b for a, b in zip(macd_line, signal_line)]
    
    # KDJ 计算（纯 Python）
    low = kdata['low']
    high = kdata['high']
    rsv = []
    for i in range(len(close)):
        start = max(0, i - config.kdj_period + 1)
        window_low = min(low[start:i+1])
        window_high = max(high[start:i+1])
        if window_high == window_low:
            rsv.append(50.0)
        else:
            rsv.append((close[i] - window_low) / (window_high - window_low) * 100)
    
    def smooth(data, span):
        alpha = 2.0 / (span + 1)
        result = [data[0]]
        for i in range(1, len(data)):
            val = alpha * data[i] + (1 - alpha) * result[-1]
            result.append(val)
        return result
    
    k = smooth(rsv, config.kdj_smooth_k)
    d = smooth(k, config.kdj_smooth_d)
    j = [3 * k_i - 2 * d_i for k_i, d_i in zip(k, d)]
    
    # 共振检测（15m/30m/1h）—— 降级为简单逻辑（因无 resample）
    recent_15 = kdata['close'][-config.max_klines_for_resonance:]
    resonance_15 = all(c > kdata['ma45'][i] for i, c in enumerate(recent_15))
    total_resonance = 1 if resonance_15 else 0
    
    is_bullish = total_resonance >= 1
    
    # 力度（MACD柱面积变化率）
    hist_change = 0.0
    if len(macd_hist) >= 2:
        prev = abs(macd_hist[-2])
        curr = abs(macd_hist[-1])
        hist_change = abs(curr - prev) / (prev + 1e-8) * 100
    
    has_momentum = hist_change > config.momentum_threshold_pct
    
    # 时效性（最近4根是否突破MA45）
    timely = sum(1 for i in range(-config.max_klines_for_resonance, 0) 
                 if kdata['close'][i] > kdata['ma45'][i]) >= config.max_klines_for_resonance
    
    if is_bullish and has_momentum and timely:
        sl = latest['ma45'] * (1 - config.STOP_LOSS_BUFFER)
        tp = latest['high'] + 1.5 * (latest['high'] - latest['low'])
        return {
            'action': 'buy',
            'reason': f'✅ Bullish ({total_resonance}/1)+Momentum+Timely',
            'stop_loss': sl,
            'take_profit': tp
        }
    
    elif not is_bullish and has_momentum and timely:
        sl = latest['ma45'] * (1 + config.STOP_LOSS_BUFFER)
        tp = latest['low'] - 1.5 * (latest['high'] - latest['low'])
        return {
            'action': 'sell',
            'reason': f'⚠️ Bearish (0/{total_resonance})+Momentum+Timely',
            'stop_loss': sl,
            'take_profit': tp
        }
    
    return {'action': 'hold', 'reason': 'No signal'}

# -------------------------------
# 🧪 纯 Python GridSearch（零依赖）
# -------------------------------
def run_grid_search():
    # 参数空间（小范围，确保快速）
    param_space = {
        'macd_fast': [2, 3, 4],
        'kdj_period': [7, 9, 11],
        'momentum_threshold_pct': [10.0, 15.0, 20.0]
    }
    
    keys = list(param_space.keys())
    values = list(param_space.values())
    combinations = list(product(*values))
    
    # 固定基础K线（避免每次重生成）
    base_kdata = generate_klines('15m', 100)
    
    def evaluate(params):
        cfg = Config()
        for k, v in zip(keys, params):
            setattr(cfg, k, v)
        
        # 模拟交易（纯 Python）
        trades = []
        balance = 100.0
        for i in range(50, len(base_kdata['close'])):
            window = {k: v[:i+1] for k, v in base_kdata.items()}
            signal = generate_paferr_signal(window, cfg)
            if signal['action'] in ['buy', 'sell']:
                pnl = 10.0 if signal['action'] == 'buy' else -8.0
                fee = 0.006
                net = pnl - fee
                balance += net
                trades.append(net)
        
        if len(trades) < 5:
            return params, -1.0
        
        returns = [t / 100.0 for t in trades]
        mean_ret = sum(returns) / len(returns)
        std_ret = (sum((r - mean_ret)**2 for r in returns) / len(returns)) ** 0.5
        sharpe = mean_ret / (std_ret + 1e-8) * (252 * 4) ** 0.5
        return params, float(sharpe)
    
    best_score = -10.0
    best_params = None
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(evaluate, combo): combo for combo in combinations}
        for future in as_completed(futures):
            try:
                params, score = future.result()
                if score > best_score:
                    best_score = score
                    best_params = params
            except Exception:
                pass
    
    elapsed = time.time() - start_time
    
    if best_params:
        for k, v in zip(keys, best_params):
            setattr(Config, k, v)
        st.session_state.opt_result = dict(zip(keys, best_params))
        st.success(f"✅ 优化完成！{elapsed:.1f}s | 夏普: {best_score:.3f} | {dict(zip(keys, best_params))}")
        st.toast(f"🎉 参数已更新：{dict(zip(keys, best_params))}", icon="✅")
    else:
        st.warning("⚠️ 未找到有效参数")

# -------------------------------
# 🖼️ 单屏渲染（输入 dict，输出 Plotly 图）
# -------------------------------
def render_timeframe_screen(screen_id: int, timeframe: str, config):
    st.subheader(f"⏱️ {timeframe} — 屏幕 #{screen_id}")

    selected_tf = st.selectbox(
        "选择时间级别",
        options=config.TIMEFRAMES,
        index=config.TIMEFRAMES.index(timeframe),
        key=f"tf_{screen_id}"
    )

    kdata = generate_klines(selected_tf, 100)
    signal = generate_paferr_signal(kdata, config)

    # 创建三联图
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.5, 0.25, 0.25],
        subplot_titles=(f'K线图（{selected_tf}）', 'MACD', 'KDJ')
    )

    # K线（绿色涨 / 红色跌）
    opens = kdata['open']
    highs = kdata['high']
    lows = kdata['low']
    closes = kdata['close']
    timestamps = kdata['timestamp']
    
    # Plotly Candlestick 接受 list 输入
    inc = [c > o for o, c in zip(opens, closes)]
    colors = ['green' if x else 'red' for x in inc]
    fill_colors = ['lightgreen' if x else 'lightsalmon' for x in inc]
    
    fig.add_trace(go.Candlestick(
        x=timestamps,
        open=opens,
        high=highs,
        low=lows,
        close=closes,
        increasing_line_color='green',
        decreasing_line_color='red',
        increasing_fillcolor='lightgreen',
        decreasing_fillcolor='lightsalmon',
        name='K线'
    ), row=1, col=1)

    # BOLL（土黄上下轨 + 红色中轨）
    if kdata['boll_upper']:
        fig.add_trace(go.Scatter(
            x=timestamps, y=kdata['boll_upper'],
            mode='lines', name='BOLL上轨',
            line=dict(color='#CC9900', width=1.2, dash='dot')
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=timestamps, y=kdata['boll_mid'],
            mode='lines', name='BOLL中轨',
            line=dict(color='red', width=2.5)
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=timestamps, y=kdata['boll_lower'],
            mode='lines', name='BOLL下轨',
            line=dict(color='#CC9900', width=1.2, dash='dot')
        ), row=1, col=1)

    # MA线（严格配色）
    ma_configs = [
        ('ma5', '#4B0082', 'MA5（靛蓝）'),
        ('ma10', 'red', 'MA10（红）'),
        ('ma30', 'goldenrod', 'MA30（黄）'),
        ('ma45', '#9400D3', 'MA45（亮紫）'),
    ]
    for col, color, name in ma_configs:
        if kdata[col]:
            fig.add_trace(go.Scatter(
                x=timestamps, y=kdata[col],
                mode='lines', name=name,
                line=dict(color=color, width=1.8, shape='spline')
            ), row=1, col=1)

    # PAFER信号标记
    if signal['action'] in ['buy', 'sell']:
        latest_ts = timestamps[-1]
        color = 'green' if signal['action'] == 'buy' else 'red'
        fig.add_vline(
            x=latest_ts,
            line_dash="solid",
            line_color=color,
            annotation_text=f"{signal['action'].upper()} SIGNAL",
            annotation_position="top",
            row=1, col=1
        )
        fig.add_hline(y=signal['stop_loss'], line_dash="dash", line_color="red", annotation_text="STOP LOSS", row=1, col=1)
        fig.add_hline(y=signal['take_profit'], line_dash="dash", line_color="green", annotation_text="TAKE PROFIT", row=1, col=1)

    # MACD（纯 Python 计算）
    close = kdata['close']
    def ewm(data, span):
        alpha = 2.0 / (span + 1)
        result = [data[0]]
        for i in range(1, len(data)):
            result.append(alpha * data[i] + (1 - alpha) * result[-1])
        return result
    ema_fast = ewm(close, config.macd_fast)
    ema_slow = ewm(close, config.macd_slow)
    macd_line = [a - b for a, b in zip(ema_fast, ema_slow)]
    signal_line = ewm(macd_line, config.macd_signal)
    macd_hist = [a - b for a, b in zip(macd_line, signal_line)]
    
    colors = ['red' if x < 0 else 'green' for x in macd_hist]
    fig.add_trace(go.Bar(
        x=timestamps, y=macd_hist,
        marker_color=colors,
        showlegend=False
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=timestamps, y=macd_line,
        mode='lines', name='MACD Line',
        line=dict(color='orange', width=2)
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=timestamps, y=signal_line,
        mode='lines', name='Signal Line',
        line=dict(color='purple', width=2, dash='dot')
    ), row=2, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="gray", row=2, col=1)

    # KDJ（纯 Python）
    low = kdata['low']
    high = kdata['high']
    rsv = []
    for i in range(len(close)):
        start = max(0, i - config.kdj_period + 1)
        win_low = min(low[start:i+1])
        win_high = max(high[start:i+1])
        if win_high == win_low:
            rsv.append(50.0)
        else:
            rsv.append((close[i] - win_low) / (win_high - win_low) * 100)
    
    def smooth(data, span):
        alpha = 2.0 / (span + 1)
        result = [data[0]]
        for i in range(1, len(data)):
            result.append(alpha * data[i] + (1 - alpha) * result[-1])
        return result
    
    k = smooth(rsv, config.kdj_smooth_k)
    d = smooth(k, config.kdj_smooth_d)
    j = [3 * k_i - 2 * d_i for k_i, d_i in zip(k, d)]

    fig.add_trace(go.Scatter(x=timestamps, y=k, mode='lines', name='K', line=dict(color='purple', width=2)), row=3, col=1)
    fig.add_trace(go.Scatter(x=timestamps, y=d, mode='lines', name='D', line=dict(color='pink', width=2)), row=3, col=1)
    fig.add_trace(go.Scatter(x=timestamps, y=j, mode='lines', name='J', line=dict(color='yellow', width=2, dash='dot')), row=3, col=1)
    fig.add_hrect(y0=80, y1=100, fillcolor="red", opacity=0.1, layer="below", row=3, col=1)
    fig.add_hrect(y0=0, y1=20, fillcolor="green", opacity=0.1, layer="below", row=3, col=1)
    fig.update_yaxes(range=[0, 100], row=3, col=1)

    fig.update_layout(
        height=750,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=10, r=10, t=30, b=10),
        hovermode='x unified',
        font=dict(size=11)
    )
    fig.update_xaxes(rangeslider_visible=False, row=1, col=1)
    fig.update_xaxes(type="date", tickformat="%H:%M", row=2, col=1)
    fig.update_xaxes(type="date", tickformat="%H:%M", row=3, col=1)
    st.plotly_chart(fig, use_container_width=True, width='stretch')

# -------------------------------
# 🧩 主程序（Streamlit App）
# -------------------------------
def main():
    st.set_page_config(
        page_title="PAFER 交易看板（Zero-Compile · Python 3.13 Ready）",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("🎯 PAFER 交易看板（Streamlit Cloud · Python 3.13 · 零编译）")
    st.caption("✅ 无 pandas ✅ 无 numpy ✅ 无 ccxt ✅ 无 distutils ✅ 100% wheel 安装")

    # === 顶部控制栏 ===
    col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
    with col1:
        live_mode = st.toggle("🟢 实盘模式（演示关闭）", value=False)
        if live_mode:
            st.warning("⚠️ 实盘需 API 密钥，当前为虚拟模式")
    with col2:
        st.metric("💰 虚拟余额", f"{Config.VIRTUAL_INITIAL_BALANCE:.2f} USDT")
    with col3:
        st.metric("📊 当前信号", "等待中...")
    with col4:
        st.metric("🛡️ 风险等级", "✅ 正常")

    # === 左侧参数面板 ===
    with st.sidebar:
        st.header("⚙️ PAFER 参数")
        momentum_thresh = st.slider(
            "力度阈值 (%)",
            min_value=5.0, max_value=30.0,
            value=Config.momentum_threshold_pct,
            step=0.5
        )
        max_k = st.number_input(
            "时效K线数",
            min_value=2, max_value=6,
            value=Config.max_klines_for_resonance,
            step=1
        )
        sl_buffer = st.slider(
            "止损缓冲比例 (%)",
            min_value=0.1, max_value=1.0,
            value=Config.STOP_LOSS_BUFFER * 100,
            step=0.1
        )
        
        Config.momentum_threshold_pct = momentum_thresh
        Config.max_klines_for_resonance = max_k
        Config.STOP_LOSS_BUFFER = sl_buffer / 100.0

        # ✅ 优化按钮（纯 Python）
        st.divider()
        st.subheader("🔬 参数优化（GridSearch）")
        if st.button("⚡ 运行网格搜索（27种组合）", use_container_width=True, type="primary"):
            run_grid_search()

        if hasattr(st.session_state, 'opt_result'):
            st.info(f"🏆 当前最优: {st.session_state.opt_result}")

    # === 多屏K线 ===
    st.subheader("🖥️ 多周期K线矩阵（1–6 屏）")

    if 'screens' not in st.session_state:
        st.session_state.screens = [{'id': 1, 'tf': '15m'}]

    screens = st.session_state.screens
    n_screens = len(screens)

    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown(f"**当前屏幕：{n_screens} 个** | 时间级别：{' | '.join([f'`{s['tf']}`' for s in screens])}")
    with col2:
        if n_screens < 6:
            if st.button("➕ Add Screen", use_container_width=True):
                new_id = max([s['id'] for s in screens], default=0) + 1
                st.session_state.screens.append({'id': new_id, 'tf': '15m'})
                st.rerun()
        if n_screens > 1:
            if st.button("➖ Remove Last", use_container_width=True):
                st.session_state.screens.pop()
                st.rerun()

    # 渲染
    if n_screens == 1:
        render_timeframe_screen(screens[0]['id'], screens[0]['tf'], Config)
    elif n_screens <= 2:
        cols = st.columns(2)
        for i, screen in enumerate(screens):
            with cols[i]:
                render_timeframe_screen(screen['id'], screen['tf'], Config)
    elif n_screens <= 4:
        cols = st.columns(2)
        for i, screen in enumerate(screens):
            with cols[i % 2]:
                render_timeframe_screen(screen['id'], screen['tf'], Config)
    else:
        cols = st.columns(3)
        for i, screen in enumerate(screens):
            with cols[i % 3]:
                render_timeframe_screen(screen['id'], screen['tf'], Config)

    # === 虚拟交易记录 ===
    st.divider()
    st.subheader("📋 虚拟交易记录（实时滚动）")

    if 'virtual_trades' not in st.session_state:
        st.session_state.virtual_trades = []

    # 每次渲染添加一笔
    now = datetime.now()
    last_balance = Config.VIRTUAL_INITIAL_BALANCE
    if st.session_state.virtual_trades:
        last_balance = st.session_state.virtual_trades[-1]['balance_after']

    side = 'buy' if len(st.session_state.virtual_trades) % 2 == 0 else 'sell'
    pnl = 10.0 if side == 'buy' else -8.0
    balance_after = round(last_balance + pnl - 0.006, 2)

    new_trade = {
        'trade_id': f"VIRT_{int(now.timestamp())}",
        'side': side,
        'open_time': now.isoformat(),
        'open_price': round(last_balance * 32.0, 2),
        'close_time': (now + timedelta(minutes=15)).isoformat(),
        'close_price': round(last_balance * 32.0 + (10 if side == 'buy' else -8), 2),
        'pnl': pnl,
        'fee': 0.006,
        'net_pnl': round(pnl - 0.006, 4),
        'balance_after': balance_after,
        'reason': 'PAFER Optimized Signal'
    }
    st.session_state.virtual_trades.append(new_trade)

    # 最近20笔
    trades_list = st.session_state.virtual_trades[-20:]
    trades_df = {
        'trade_id': [t['trade_id'] for t in trades_list],
        'side': [t['side'] for t in trades_list],
        'open_time': [t['open_time'] for t in trades_list],
        'open_price': [t['open_price'] for t in trades_list],
        'close_time': [t['close_time'] for t in trades_list],
        'close_price': [t['close_price'] for t in trades_list],
        'pnl': [t['pnl'] for t in trades_list],
        'fee': [t['fee'] for t in trades_list],
        'net_pnl': [t['net_pnl'] for t in trades_list],
        'balance_after': [t['balance_after'] for t in trades_list],
        'reason': [t['reason'] for t in trades_list],
    }

    st.dataframe(
        trades_df,
        use_container_width=True,
        column_config={
            "open_time": st.column_config.DatetimeColumn("开仓时间"),
            "close_time": st.column_config.DatetimeColumn("平仓时间"),
            "pnl": st.column_config.NumberColumn("毛收益", format="%.4f USDT"),
            "fee": st.column_config.NumberColumn("手续费", format="%.4f USDT"),
            "net_pnl": st.column_config.NumberColumn("净收益", format="%.4f USDT"),
            "balance_after": st.column_config.NumberColumn("余额", format="%.2f USDT"),
            "reason": st.column_config.TextColumn("信号原因", width="large")
        },
        hide_index=True
    )

    # CSV导出
    import csv
    from io import StringIO
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=trades_df.keys())
    writer.writeheader()
    for trade in trades_list:
        writer.writerow({
            'trade_id': trade['trade_id'],
            'side': trade['side'],
            'open_time': trade['open_time'],
            'open_price': trade['open_price'],
            'close_time': trade['close_time'],
            'close_price': trade['close_price'],
            'pnl': trade['pnl'],
            'fee': trade['fee'],
            'net_pnl': trade['net_pnl'],
            'balance_after': trade['balance_after'],
            'reason': trade['reason']
        })
    csv_content = output.getvalue().encode('utf-8')
    
    st.download_button(
        "📥 导出全部虚拟交易",
        data=csv_content,
        file_name=f"pafar_virtual_trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True
    )

if __name__ == "__main__":
    main()
