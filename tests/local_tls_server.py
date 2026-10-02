import http.server
import os
import pathlib
import ssl
import subprocess
import sys
import tempfile
import threading


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"ok")


with tempfile.TemporaryDirectory() as temporary:
    directory = pathlib.Path(temporary)
    cert = directory / "root.pem"
    key = directory / "key.pem"
    subprocess.run([
        "openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "1",
        "-keyout", str(key), "-out", str(cert), "-subj", "/CN=localhost",
        "-addext", "subjectAltName=DNS:localhost",
        "-addext", "basicConstraints=critical,CA:TRUE",
    ], check=True, capture_output=True)
    cert_directory = directory / "certs"
    cert_directory.mkdir()
    digest = subprocess.check_output([
        "openssl", "x509", "-in", str(cert), "-hash", "-noout"
    ], text=True).strip()
    (cert_directory / f"{digest}.0").write_bytes(cert.read_bytes())
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(cert, key)
    server.socket = context.wrap_socket(server.socket, server_side=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    environment = {k: v for k, v in os.environ.items()
                   if k not in {"SSL_CERT_FILE", "SSL_CERT_DIR"}}
    url = f"https://localhost:{server.server_port}"
    cases = [
        (url, "trusted", {"SSL_CERT_FILE": str(cert)}),
        (url, "trusted", {"SSL_CERT_DIR": str(cert_directory)}),
        (url.replace("localhost", "127.0.0.1"), "untrusted", {"SSL_CERT_FILE": str(cert)}),
        (url, "invalid", {"SSL_CERT_FILE": str(directory / "missing.pem")}),
        (url, "system", {}),
    ]
    try:
        for endpoint, mode, overrides in cases:
            subprocess.run([sys.argv[1], endpoint, mode],
                           env=environment | overrides, check=True, timeout=45)
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
