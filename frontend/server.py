"""Local frontend for the Passport_Verify n8n workflow.

Serves index.html and proxies uploads to the n8n webhook so the webhook's
"API" header key stays server-side and the browser never hits CORS.

Config via environment variables:
  N8N_WEBHOOK_URL  full webhook URL (required)
  N8N_API_KEY      value for the webhook's "API" header
  PORT             optional, default 8080

Run:  python server.py   then open http://localhost:8080
"""
import json
import os
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

WEBHOOK_URL = os.environ.get('N8N_WEBHOOK_URL', '')
PORT = int(os.environ.get('PORT', '8080'))
INDEX = Path(__file__).parent / 'public' / 'index.html'
KEY_FILE = Path(__file__).with_name('.n8n_api_key')
MAX_BODY = 25 * 1024 * 1024


def api_key():
    # Read per request so the key can be added without restarting the server.
    if os.environ.get('N8N_API_KEY'):
        return os.environ['N8N_API_KEY']
    return KEY_FILE.read_text(encoding='utf-8').strip() if KEY_FILE.is_file() else ''


class Handler(BaseHTTPRequestHandler):
    def _send(self, code, body, ctype='application/json'):
        data = body if isinstance(body, bytes) else body.encode('utf-8')
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path in ('/', '/index.html'):
            self._send(200, INDEX.read_bytes(), 'text/html; charset=utf-8')
        elif self.path == '/api/config':
            self._send(200, json.dumps({'webhookUrl': None}))  # None = use this server's /api/verify proxy
        elif self.path == '/config':
            self._send(200, json.dumps({'webhook': bool(WEBHOOK_URL), 'api_key_set': bool(api_key())}))
        else:
            self._send(404, json.dumps({'error': 'not found'}))

    def do_POST(self):
        if self.path != '/api/verify':
            self._send(404, json.dumps({'error': 'not found'}))
            return
        if not WEBHOOK_URL:
            self._send(500, json.dumps({'error': 'Set N8N_WEBHOOK_URL before starting the server'}))
            return
        key = api_key()
        length = int(self.headers.get('Content-Length') or 0)
        if length <= 0 or length > MAX_BODY:
            self._send(413, json.dumps({'error': 'Empty or too-large request'}))
            return
        body = self.rfile.read(length)
        headers = {'Content-Type': 'application/json'}
        if key:  # webhook header auth is optional
            headers['API'] = key
        req = urllib.request.Request(WEBHOOK_URL, data=body, method='POST', headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=900) as resp:
                self._send(resp.status, resp.read())
        except urllib.error.HTTPError as e:
            self._send(e.code, e.read() or json.dumps({'error': str(e)}).encode())
        except Exception as e:  # network/timeout
            self._send(502, json.dumps({'error': f'Could not reach n8n: {e}'}))

    def log_message(self, fmt, *args):
        print('[frontend]', fmt % args)


if __name__ == '__main__':
    print(f'Passport verification UI on http://localhost:{PORT}')
    if not api_key():
        print(f'No webhook API key set — calling without the "API" header (add {KEY_FILE.name} if auth is enabled).')
    ThreadingHTTPServer(('127.0.0.1', PORT), Handler).serve_forever()
