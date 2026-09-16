import os
import sys
from flask import Flask, render_template, request, jsonify

from stant import APP_DESCRIPTION, APP_NAME
from stant.market_data import build_chart_payload, get_stock_data
from stant.strategies import run_strategy_analysis, STRATEGY_NAMES
from stant.backtest import backtest_strategy
from stant.multi_analysis import run_multi_stock_screener
from stant.portfolio import build_optimized_portfolio

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', app_name=APP_NAME, app_description=APP_DESCRIPTION, strategies=STRATEGY_NAMES)

@app.route('/api/stock')
def api_stock():
    ticker = request.args.get('ticker', '').strip()
    interval = request.args.get('interval', '1d').strip()
    strategy_key = request.args.get('strategy', 'ensemble').strip()
    start = request.args.get('start', '').strip()
    end = request.args.get('end', '').strip()
    
    if not ticker:
        return jsonify({'error': 'Ticker symbol is required'}), 400

    if start and end and start > end:
        return jsonify({'error': 'Start date must be earlier than or equal to end date'}), 400
        
    try:
        stock_data = get_stock_data(ticker, interval, start, end)
        
        if stock_data.frame.empty:
            return jsonify({
                'error': (
                    f"No data returned for ticker '{ticker.upper()}'. Please verify the ticker symbol "
                    f"and verify that your selected interval ('{interval}') falls within Yahoo Finance's limits "
                    f"(e.g., 1m is restricted to last 7 days, 1h is restricted to last 730 days). "
                    f"Tried Yahoo Finance symbols: {', '.join(stock_data.tried_symbols)}."
                )
            }), 400

        payload = build_chart_payload(stock_data, interval)

        if not payload["chart_data"]:
            return jsonify({
                'error': (
                    f"No plottable OHLC data was returned for ticker '{ticker.upper()}' with interval '{interval}'. "
                    "Try a different date range or interval."
                )
            }), 400

        # Enrich payload with trading strategies, indicators, signals, and trade setups
        strategy_analysis = run_strategy_analysis(stock_data.frame, strategy_key=strategy_key, interval=interval)
        payload.update(strategy_analysis)

        return jsonify(payload)
        
    except Exception as e:
        print(f"Error fetching data: {e}", file=sys.stderr)
        return jsonify({'error': str(e)}), 500

@app.route('/api/signals')
def api_signals():
    ticker = request.args.get('ticker', '').strip()
    interval = request.args.get('interval', '1d').strip()
    strategy_key = request.args.get('strategy', 'ensemble').strip()
    
    if not ticker:
        return jsonify({'error': 'Ticker symbol is required'}), 400

    try:
        stock_data = get_stock_data(ticker, interval)
        if stock_data.frame.empty:
            return jsonify({'error': f"No data returned for ticker '{ticker}'"}), 400

        analysis = run_strategy_analysis(stock_data.frame, strategy_key=strategy_key, interval=interval)
        analysis["ticker"] = stock_data.resolved_symbol
        return jsonify(analysis)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/backtest')
def api_backtest():
    ticker = request.args.get('ticker', '').strip()
    interval = request.args.get('interval', '1d').strip()
    strategy_key = request.args.get('strategy', 'ensemble').strip()
    
    if not ticker:
        return jsonify({'error': 'Ticker symbol is required'}), 400

    try:
        stock_data = get_stock_data(ticker, interval)
        if stock_data.frame.empty:
            return jsonify({'error': f"No data returned for ticker '{ticker}'"}), 400

        bt_res = backtest_strategy(stock_data.frame, strategy_key=strategy_key, interval=interval)
        bt_res["ticker"] = stock_data.resolved_symbol
        return jsonify(bt_res)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/screener')
def api_screener():
    raw_tickers = request.args.get('tickers', '').strip()
    interval = request.args.get('interval', '1d').strip()
    strategy_key = request.args.get('strategy', 'ensemble').strip()

    ticker_list = [t.strip() for t in raw_tickers.split(',') if t.strip()] if raw_tickers else None

    try:
        screener_res = run_multi_stock_screener(tickers=ticker_list, interval=interval, strategy_key=strategy_key)
        return jsonify(screener_res)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/portfolio', methods=['GET', 'POST'])
def api_portfolio():
    if request.method == 'POST':
        body = request.get_json() or {}
        raw_tickers = body.get('tickers', '')
        model_type = body.get('model_type', 'max_sharpe')
        capital = float(body.get('capital', 10000.0))
        risk_tolerance = body.get('risk_tolerance', 'moderate')
        goal = body.get('goal', 'growth')
    else:
        raw_tickers = request.args.get('tickers', '')
        model_type = request.args.get('model_type', 'max_sharpe')
        capital = float(request.args.get('capital', 10000.0))
        risk_tolerance = request.args.get('risk_tolerance', 'moderate')
        goal = request.args.get('goal', 'growth')

    ticker_list = [t.strip() for t in raw_tickers.split(',') if t.strip()] if isinstance(raw_tickers, str) and raw_tickers else raw_tickers

    try:
        res = build_optimized_portfolio(
            tickers=ticker_list,
            model_type=model_type,
            capital=capital,
            risk_tolerance=risk_tolerance,
            goal=goal,
        )
        return jsonify(res)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', '0') == '1'
    app.run(host='0.0.0.0', port=port, debug=debug)


