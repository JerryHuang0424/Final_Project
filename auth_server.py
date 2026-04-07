"""
HTTP Authentication Server with JSON Storage

This module provides a complete authentication system with:
- User registration with unique usernames
- Password hashing for security
- Password recovery via security questions
- JSON file storage for user data
- HTTP API endpoints for web integration
"""

import json
import hashlib
import secrets
import os
from datetime import datetime, timedelta
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse
import threading


class AuthSystem:
    """Authentication system with JSON file storage"""

    def __init__(self, users_file="users.json"):
        self.users_file = users_file
        self.users = self._load_users()

    def _load_users(self) -> dict:
        """Load users from JSON file"""
        if os.path.exists(self.users_file):
            try:
                with open(self.users_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                return {"users": {}}
        return {"users": {}}

    def _save_users(self):
        """Save users to JSON file"""
        with open(self.users_file, 'w', encoding='utf-8') as f:
            json.dump(self.users, f, indent=2, ensure_ascii=False)

    def hash_password(self, password: str) -> str:
        """Hash password using SHA-256"""
        return hashlib.sha256(password.encode('utf-8')).hexdigest()

    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify password against hash"""
        return self.hash_password(password) == hashed

    def register_user(self, username: str, password: str, security_question: str, security_answer: str) -> dict:
        """Register new user with unique username"""

        # Validate inputs
        if not username or not password:
            return {"success": False, "error": "Username and password are required"}

        if len(username) < 3:
            return {"success": False, "error": "Username must be at least 3 characters"}

        if len(password) < 6:
            return {"success": False, "error": "Password must be at least 6 characters"}

        # Check if username already exists
        if username in self.users.get("users", {}):
            return {"success": False, "error": "Username already exists"}

        # Create user record
        self.users.setdefault("users", {})[username] = {
            "password_hash": self.hash_password(password),
            "security_question": security_question,
            "security_answer_hash": self.hash_password(security_answer.lower().strip()),
            "created_at": datetime.now().isoformat(),
            "last_login": None,
            "recovery_token": None,
            "recovery_token_expiry": None
        }

        self._save_users()
        return {"success": True, "message": "User registered successfully"}

    def login_user(self, username: str, password: str) -> dict:
        """Authenticate user and return session"""

        user = self.users.get("users", {}).get(username)
        if not user:
            return {"success": False, "error": "Invalid username or password"}

        if not self.verify_password(password, user["password_hash"]):
            return {"success": False, "error": "Invalid username or password"}

        # Update last login
        user["last_login"] = datetime.now().isoformat()
        self._save_users()

        # Generate session token
        session_token = secrets.token_urlsafe(32)

        return {
            "success": True,
            "message": "Login successful",
            "username": username,
            "session_token": session_token
        }

    def verify_security_answer(self, username: str, answer: str) -> bool:
        """Verify security answer for password recovery"""
        user = self.users.get("users", {}).get(username)
        if not user:
            return False

        return self.verify_password(answer.lower().strip(), user["security_answer_hash"])

    def recover_password(self, username: str, security_answer: str, new_password: str) -> dict:
        """Reset password using security answer"""

        user = self.users.get("users", {}).get(username)
        if not user:
            return {"success": False, "error": "User not found"}

        # Verify security answer
        if not self.verify_security_answer(username, security_answer):
            return {"success": False, "error": "Incorrect security answer"}

        # Update password
        user["password_hash"] = self.hash_password(new_password)
        user["recovery_token"] = None
        user["recovery_token_expiry"] = None

        self._save_users()
        return {"success": True, "message": "Password reset successfully"}

    def get_security_question(self, username: str) -> dict:
        """Get security question for password recovery"""
        user = self.users.get("users", {}).get(username)
        if not user:
            return {"success": False, "error": "User not found"}

        return {"success": True, "security_question": user["security_question"]}


class AuthHTTPHandler(BaseHTTPRequestHandler):
    """HTTP request handler for authentication endpoints"""

    def __init__(self, *args, auth_system=None, **kwargs):
        self.auth_system = auth_system or AuthSystem()
        super().__init__(*args, **kwargs)

    def _send_response(self, status_code, data):
        """Send JSON response"""
        self.send_response(status_code)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

        response = json.dumps(data, ensure_ascii=False)
        self.wfile.write(response.encode('utf-8'))

    def _parse_post_data(self):
        """Parse POST data from request"""
        content_length = int(self.headers.get('Content-Length', 0))
        if content_length == 0:
            return {}

        post_data = self.rfile.read(content_length).decode('utf-8')
        return parse_qs(post_data)

    def do_OPTIONS(self):
        """Handle CORS preflight requests"""
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_GET(self):
        """Handle GET requests"""
        parsed_path = urlparse(self.path)

        if parsed_path.path == '/status':
            self._send_response(200, {"status": "Server is running"})
        else:
            self._send_response(404, {"error": "Endpoint not found"})

    def do_POST(self):
        """Handle POST requests"""
        parsed_path = urlparse(self.path)
        post_data = self._parse_post_data()

        # Extract values from post_data (which returns lists)
        def get_value(key):
            return post_data.get(key, [''])[0] if key in post_data else ''

        try:
            if parsed_path.path == '/register':
                # User registration
                result = self.auth_system.register_user(
                    username=get_value('username'),
                    password=get_value('password'),
                    security_question=get_value('security_question'),
                    security_answer=get_value('security_answer')
                )
                self._send_response(200 if result["success"] else 400, result)

            elif parsed_path.path == '/login':
                # User login
                result = self.auth_system.login_user(
                    username=get_value('username'),
                    password=get_value('password')
                )
                self._send_response(200 if result["success"] else 401, result)

            elif parsed_path.path == '/recover':
                # Password recovery
                result = self.auth_system.recover_password(
                    username=get_value('username'),
                    security_answer=get_value('security_answer'),
                    new_password=get_value('new_password')
                )
                self._send_response(200 if result["success"] else 400, result)

            elif parsed_path.path == '/security-question':
                # Get security question
                result = self.auth_system.get_security_question(
                    username=get_value('username')
                )
                self._send_response(200 if result["success"] else 404, result)

            else:
                self._send_response(404, {"error": "Endpoint not found"})

        except Exception as e:
            self._send_response(500, {"error": f"Internal server error: {str(e)}"})


def start_auth_server(port=8000):
    """Start the authentication HTTP server"""
    server = HTTPServer(('localhost', port), AuthHTTPHandler)
    print(f"Authentication server running on http://localhost:{port}")
    print("Available endpoints:")
    print("  POST /register - Register new user")
    print("  POST /login - User login")
    print("  POST /recover - Password recovery")
    print("  POST /security-question - Get security question")
    print("  GET /status - Server status")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        server.shutdown()


def test_auth_system():
    """Test the authentication system"""
    auth = AuthSystem("test_users.json")

    # Test registration
    print("Testing user registration...")
    result = auth.register_user(
        username="testuser",
        password="password123",
        security_question="What is your favorite color?",
        security_answer="blue"
    )
    print(f"Registration result: {result}")

    # Test login
    print("\nTesting user login...")
    result = auth.login_user("testuser", "password123")
    print(f"Login result: {result}")

    # Test password recovery
    print("\nTesting password recovery...")
    security_question = auth.get_security_question("testuser")
    print(f"Security question: {security_question}")

    result = auth.recover_password("testuser", "blue", "newpassword456")
    print(f"Password recovery result: {result}")

    # Test login with new password
    result = auth.login_user("testuser", "newpassword456")
    print(f"Login with new password: {result}")

    # Clean up test file
    if os.path.exists("test_users.json"):
        os.remove("test_users.json")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "server":
        port = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
        start_auth_server(port)
    else:
        test_auth_system()