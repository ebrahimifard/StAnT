import sys
from flask import Flask, render_template, request, jsonify

from stant import APP_DESCRIPTION, APP_NAME
from stant.market_data import build_chart_payload, get_stock_data

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', app_name=APP_NAME, app_description=APP_DESCRIPTION)

@app.route('/api/stock')
def api_stock():
    ticker = request.args.get('ticker', '').strip()
    interval = request.args.get('interval', '1d').strip()
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

        return jsonify(payload)
        
    except Exception as e:
        print(f"Error fetching data: {e}", file=sys.stderr)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Default to port 5000
    app.run(debug=True, port=5000)
