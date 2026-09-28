"""
Unit and integration tests for User Authentication, Registration, and Session management.
"""

import unittest
from src.app import create_app
from src.config import Config
from src.database import db, User, get_user_by_username, get_user_by_email


class TestAuthSystem(unittest.TestCase):
    """Tests the login, registration, password hashing, and session management system."""

    def setUp(self):
        class TestConfig(Config):
            TESTING = True
            SQLALCHEMY_DATABASE_URI = "sqlite:///:memory:"
            WTF_CSRF_ENABLED = False

        self.app = create_app(TestConfig)
        self.client = self.app.test_client()

        with self.app.app_context():
            db.create_all()
            from src.database import seed_default_users
            seed_default_users()

    def tearDown(self):
        with self.app.app_context():
            db.session.remove()
            db.drop_all()

    def test_seed_demo_users_exist(self):
        """Verifies default demo clinician and researcher accounts are properly seeded."""
        with self.app.app_context():
            doc = get_user_by_username("demo_doctor")
            self.assertIsNotNone(doc)
            self.assertTrue(doc.check_password("password123"))
            self.assertFalse(doc.check_password("wrongpassword"))
            self.assertEqual(doc.full_name, "Dr. Sarah Chen, MD")

            res = get_user_by_username("researcher")
            self.assertIsNotNone(res)
            self.assertTrue(res.check_password("password123"))

    def test_login_page_renders(self):
        """Should serve the modern medical login page with clean form fields and healthcare branding."""
        resp = self.client.get("/login")
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"Welcome Back!", resp.data)
        self.assertIn(b"Login", resp.data)
        self.assertIn(b"medicalLoginForm", resp.data)
        self.assertIn(b"Email Address or Username", resp.data)

    def test_register_page_renders(self):
        """Should serve the registration page."""
        resp = self.client.get("/register")
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"Create Clinical / Research Account", resp.data)
        self.assertIn(b"Dermatologist", resp.data)

    def test_login_successful_via_json(self):
        """Should authenticate valid credentials and return user metadata."""
        resp = self.client.post("/api/auth/login", json={
            "username": "demo_doctor",
            "password": "password123"
        })
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertTrue(data["success"])
        self.assertEqual(data["user"]["username"], "demo_doctor")
        self.assertEqual(data["user"]["full_name"], "Dr. Sarah Chen, MD")

    def test_login_successful_via_form(self):
        """Should handle form-based POST and redirect to dashboard."""
        resp = self.client.post("/api/auth/login", data={
            "username": "researcher",
            "password": "password123"
        }, follow_redirects=False)
        self.assertEqual(resp.status_code, 302)
        self.assertIn("/", resp.headers.get("Location", ""))

    def test_login_invalid_password(self):
        """Should reject incorrect password with 401 status."""
        resp = self.client.post("/api/auth/login", json={
            "username": "demo_doctor",
            "password": "incorrect_password"
        })
        self.assertEqual(resp.status_code, 401)
        data = resp.get_json()
        self.assertFalse(data["success"])
        self.assertIn("Invalid credentials", data["error"])

    def test_login_nonexistent_user(self):
        """Should reject nonexistent user with 401 status."""
        resp = self.client.post("/api/auth/login", json={
            "username": "unknown_user_999",
            "password": "password123"
        })
        self.assertEqual(resp.status_code, 401)
        data = resp.get_json()
        self.assertFalse(data["success"])

    def test_registration_and_immediate_session(self):
        """Should create a new user account and log them in."""
        resp = self.client.post("/api/auth/register", json={
            "full_name": "Dr. Marcus Brody",
            "username": "mbrody",
            "email": "mbrody@hospital.org",
            "password": "securepassword123",
            "confirm_password": "securepassword123",
            "role": "Dermatologist"
        })
        self.assertEqual(resp.status_code, 201)
        data = resp.get_json()
        self.assertTrue(data["success"])
        self.assertEqual(data["user"]["username"], "mbrody")

        with self.app.app_context():
            created = get_user_by_username("mbrody")
            self.assertIsNotNone(created)
            self.assertTrue(created.check_password("securepassword123"))

    def test_registration_duplicate_username(self):
        """Should reject registration if username already exists."""
        resp = self.client.post("/api/auth/register", json={
            "full_name": "Duplicate Sarah",
            "username": "demo_doctor",
            "email": "another@clinic.med",
            "password": "password123",
            "confirm_password": "password123",
            "role": "Dermatologist"
        })
        self.assertEqual(resp.status_code, 409)
        data = resp.get_json()
        self.assertFalse(data["success"])
        self.assertIn("already taken", data["error"])

    def test_logout_clears_session(self):
        """Should clear user session and redirect to login page."""
        # Log in first
        self.client.post("/api/auth/login", json={
            "username": "demo_doctor",
            "password": "password123"
        })

        # Verify logged in
        me_resp = self.client.get("/api/auth/me")
        self.assertTrue(me_resp.get_json()["authenticated"])

        # Logout
        logout_resp = self.client.get("/logout")
        self.assertEqual(logout_resp.status_code, 302)

        # Verify session cleared
        me_after = self.client.get("/api/auth/me")
        self.assertFalse(me_after.get_json()["authenticated"])

    def test_me_endpoint_unauthenticated(self):
        """Should return authenticated: false for guest visitors."""
        resp = self.client.get("/api/auth/me")
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertFalse(data["authenticated"])
        self.assertIsNone(data["user"])


if __name__ == "__main__":
    unittest.main()
