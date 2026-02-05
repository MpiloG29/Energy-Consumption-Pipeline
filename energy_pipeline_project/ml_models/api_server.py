# api_server.py - lightweight HTTP server for model predictions (no Flask)
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
import joblib
import traceback

MODEL_PATH = "ml_models/energy_predictor.pkl"

def load_model(path):
    try:
        m = joblib.load(path)
        return m
    except Exception:
        with open('ml_models/api_server.log', 'a') as f:
            f.write('Failed to load model:\n')
            traceback.print_exc(file=f)
        raise

model = load_model(MODEL_PATH)

class PredictHandler(BaseHTTPRequestHandler):
    def _set_headers(self, code=200):
        self.send_response(code)
        self.send_header('Content-type', 'application/json')
        self.end_headers()

    def do_POST(self):
        if self.path != '/predict':
            self._set_headers(404)
            self.wfile.write(json.dumps({'error': 'not found'}).encode())
            return
        length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(length)
        try:
            data = json.loads(body)
            hour = data.get('hour')
            day = data.get('day')
            pred = model.predict([[hour, day]])
            resp = {'predicted_consumption': float(pred[0])}
            self._set_headers(200)
            self.wfile.write(json.dumps(resp).encode())
        except Exception as e:
            self._set_headers(400)
            self.wfile.write(json.dumps({'error': str(e)}).encode())

def run(server_class=HTTPServer, handler_class=PredictHandler, port=5000):
    server_address = ('', port)
    httpd = server_class(server_address, handler_class)
    msg = f'Starting model API on port {port}...'
    print(msg)
    with open('ml_models/api_server.log', 'a') as f:
        f.write(msg + '\n')
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    httpd.server_close()

if __name__ == '__main__':
    run()
