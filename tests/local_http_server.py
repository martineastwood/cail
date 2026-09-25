import http.server
import json
import subprocess
import sys
import threading
import time


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_args):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/chat/error":
            self.send_json(429, {"error": {"message": "slow down", "code": "rate_limited", "type": "rate_limit"}},
                           {"X-Request-Id": "req_local"})
        elif self.path == "/chat/hold":
            self.send_stream([
                'data: {"choices":[{"index":0,"delta":{"content":"Hi"}}]}\n\n',
            ], hold=True)
        elif self.path == "/chat/truncated":
            self.send_stream(['data: {"choices":[{"index":0,"delta":{"content":"partial"}}]}\n\n'])
        elif self.path == "/chat/structured":
            if body.get("response_format", {}).get("json_schema", {}).get("strict") is not True:
                self.send_error(400)
                return
            self.send_json(200, {"choices": [{"index": 0, "finish_reason": "stop",
                                             "message": {"content": '{"answer":"yes"}'}}]})
        elif self.path == "/chat" and body.get("stream"):
            self.send_stream([
                'data: {"choices":[{"index":0,"delta":{"content":"Hi"}}]}\n\n',
                'data: {"choices":[{"index":0,"finish_reason":"stop"}]}\n\n',
                'data: {"choices":[],"usage":{"prompt_tokens":3,"completion_tokens":1}}\n\n',
                'data: [DONE]\n\n',
            ])
        elif self.path == "/chat":
            self.send_json(200, {"choices": [{"index": 0, "finish_reason": "stop",
                                             "message": {"content": "Hello"}}]})
        elif self.path == "/responses" and body.get("stream"):
            self.send_stream([
                'event: response.output_text.delta\ndata: {"delta":"Hi"}\n\n',
                'event: response.completed\ndata: {"response":{"status":"completed","output":[{"type":"message","content":[{"type":"output_text","text":"Hi"}]}]}}\n\n',
            ])
        elif self.path == "/truncated/responses":
            self.send_stream(['event: response.output_text.delta\ndata: {"delta":"partial"}\n\n'])
        elif self.path == "/anthropic/error/messages":
            self.send_json(429, {"error": {"type": "rate_limit_error", "message": "slow down"}},
                           {"request-id": "anthropic_req_local"})
        elif self.path == "/anthropic/truncated/messages":
            self.send_stream(['event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"partial"}}\n\n'])
        elif self.path == "/anthropic/hold/messages":
            self.send_stream([
                'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hi"}}\n\n',
            ], hold=True)
        elif self.path == "/anthropic/structured/messages":
            schema = body.get("output_config", {}).get("format", {}).get("schema", {})
            if schema.get("type") == "object" and schema.get("properties", {}).get("answer", {}).get("type") == "string":
                self.send_json(200, {"stop_reason": "end_turn", "content": [
                    {"type": "text", "text": '{"answer":"yes"}'}]})
            else:
                self.send_json(400, {"error": {"type": "invalid_request_error", "message": "missing output schema"}})
        elif self.path == "/anthropic/validate/messages":
            blocks = [block for message in body.get("messages", []) for block in message.get("content", [])]
            valid = (self.headers.get("Authorization") == "Bearer test-key" and
                     self.headers.get("anthropic-version") == "2023-06-01" and
                     body.get("max_tokens") == 256 and body.get("system") == "Be concise." and
                     body.get("tools", [{}])[0].get("input_schema", {}).get("type") == "object" and
                     any(block.get("type") == "image" and
                         block.get("source", {}).get("data") == "QQBC" for block in blocks) and
                     any(block.get("type") == "tool_use" and
                         block.get("input") == {"q": "x"} for block in blocks) and
                     any(block.get("type") == "tool_result" and
                         block.get("tool_use_id") == "toolu_1" for block in blocks))
            if valid:
                self.send_json(200, {"stop_reason": "end_turn", "content": [{"type": "text", "text": "Done"}]})
            else:
                self.send_json(400, {"error": {"type": "invalid_request_error", "message": "bad request mapping"}})
        elif self.path == "/anthropic/messages" and body.get("stream"):
            self.send_stream([
                'event: message_start\ndata: {"type":"message_start","message":{"content":[],"usage":{"input_tokens":5,"cache_read_input_tokens":2}}}\n\n',
                'event: content_block_start\ndata: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\n\n',
                'event: content_block_delta\ndata: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hi"}}\n\n',
                'event: content_block_stop\ndata: {"type":"content_block_stop","index":0}\n\n',
                'event: content_block_start\ndata: {"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"toolu_1","name":"search","input":{}}}\n\n',
                'event: content_block_delta\ndata: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\\"q\\":\\"x\\"}"}}\n\n',
                'event: content_block_stop\ndata: {"type":"content_block_stop","index":1}\n\n',
                'event: message_delta\ndata: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":4}}\n\n',
                'event: message_stop\ndata: {"type":"message_stop"}\n\n',
            ])
        elif self.path == "/anthropic/messages":
            self.send_json(200, {"stop_reason": "tool_use", "content": [
                {"type": "text", "text": "Searching"},
                {"type": "thinking", "thinking": "Need facts"},
                {"type": "tool_use", "id": "toolu_1", "name": "search", "input": {"q": "x"}}],
                "usage": {"input_tokens": 5, "output_tokens": 4,
                          "cache_read_input_tokens": 2, "cache_creation_input_tokens": 1}})
        elif self.path == "/gemini/models/test-model:generateContent":
            if self.headers.get("x-goog-api-key") != "test-key":
                self.send_error(401)
            elif body.get("generationConfig"):
                schema = body["generationConfig"].get("responseJsonSchema", {})
                if schema.get("properties", {}).get("answer", {}).get("type") != "string":
                    self.send_error(400)
                else:
                    self.send_json(200, {"candidates": [{"finishReason": "STOP", "content": {
                        "parts": [{"text": '{"answer":"yes"}'}]}}]})
            elif body.get("tools"):
                self.send_json(200, {"candidates": [{"finishReason": "STOP", "content": {
                    "parts": [{"functionCall": {"name": "search", "args": {"q": "x"}, "id": "call_1"},
                               "thoughtSignature": "signature"}]}}],
                    "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 3}})
            else:
                self.send_json(200, {"candidates": [{"finishReason": "STOP", "content": {
                    "parts": [{"text": "Hello"}]}}]})
        elif self.path == "/gemini/validate/models/test-model:generateContent":
            parts = [part for content in body.get("contents", []) for part in content.get("parts", [])]
            valid = (body.get("systemInstruction", {}).get("parts", [{}])[0].get("text") == "Be concise." and
                     any(part.get("inlineData", {}).get("data") == "QQBC" for part in parts) and
                     any(part.get("functionCall", {}).get("name") == "search" and
                         part.get("functionCall", {}).get("id") == "call_1" and
                         part.get("thoughtSignature") == "signature" for part in parts) and
                     any(part.get("functionResponse", {}).get("name") == "search" and
                         part.get("functionResponse", {}).get("id") == "call_1" and
                         part.get("functionResponse", {}).get("response") == {"answer": "yes"} for part in parts))
            if valid:
                self.send_json(200, {"candidates": [{"finishReason": "STOP", "content": {
                    "parts": [{"text": "Done"}]}}]})
            else:
                self.send_json(400, {"error": {"message": "bad mapping", "status": "INVALID_ARGUMENT"}})
        elif self.path == "/gemini/models/test-model:streamGenerateContent?alt=sse":
            self.send_stream([
                'data: {"candidates":[{"content":{"parts":[{"text":"Hi"}]}}]}\n\n',
                'data: {"candidates":[{"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":1}}\n\n',
            ])
        elif self.path == "/responses":
            self.send_json(200, {"status": "completed", "output": [
                {"type": "message", "content": [{"type": "output_text", "text": "Hello"}]}]})
        else:
            self.send_error(404)

    def send_json(self, status, value, headers=None):
        payload = json.dumps(value).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        for name, content in (headers or {}).items():
            self.send_header(name, content)
        self.end_headers()
        self.wfile.write(payload)

    def send_stream(self, events, hold=False):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "close")
        self.end_headers()
        for event in events:
            try:
                self.wfile.write(event.encode())
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                return
        if hold:
            time.sleep(5)
        self.close_connection = True


class Server(http.server.ThreadingHTTPServer):
    daemon_threads = True


with Server(("127.0.0.1", 0), Handler) as server:
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        subprocess.run([sys.argv[1], f"http://127.0.0.1:{server.server_port}"], check=True, timeout=10)
    finally:
        server.shutdown()
        thread.join()
