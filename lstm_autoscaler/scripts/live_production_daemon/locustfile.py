"""
Locust Load Testing Script

This script simulates concurrent user traffic against the microservices cluster.
It replays user traces from the boutique_workload_large.json file to mimic
realistic user behavior and generate telemetry data for the autoscaler.
"""

from locust import HttpUser, task, constant
import os
import sys
import json
import time

workload_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'raw', 'boutique_workload_large.json')

# TIME_SCALE compresses/expands the trace timeline.
# 1.0 = original pacing, 0.5 = 2x faster, 0.25 = 4x faster (more RPS at same users).
TIME_SCALE = 0.5

# =============================================================================
# Section 1: Workload Loading & Preprocessing
# =============================================================================
with open(workload_path, "r", encoding="utf-8") as f:
    WORKLOAD = json.load(f)

# Handle both list and dict formats in the JSON file
if isinstance(WORKLOAD, dict):
    WORKLOAD = list(WORKLOAD.values())

# Sort all requests by their relative timestamp
WORKLOAD = sorted(WORKLOAD, key=lambda x: x["time"])

# Extract unique user IDs for round-robin assignment
AVAILABLE_USERS = sorted(list({r.get("user") for r in WORKLOAD if r.get("user")}))
TOTAL_USERS = len(AVAILABLE_USERS)


# =============================================================================
# Section 2: Virtual User Class
# =============================================================================
class BoutiqueUser(HttpUser):
    wait_time = constant(0)

    # -------------------------------------------------------------------------
    # Section 2.1: User Initialization (on_start)
    # -------------------------------------------------------------------------
    def on_start(self):
        self.start_time = time.time()

        # Thread-safe round-robin user assignment using environment-level counter
        if not hasattr(self.environment, "user_counter"):
            self.environment.user_counter = 0

        if TOTAL_USERS > 0:
            idx = self.environment.user_counter % TOTAL_USERS
            self.user_id = AVAILABLE_USERS[idx]
        else:
            self.user_id = None

        self.environment.user_counter += 1

        # Filter the global workload to only this user's requests
        if self.user_id is None:
            self.my_workload = []
        else:
            self.my_workload = [r for r in WORKLOAD if r.get("user") == self.user_id]

        # Pointer to track the current position in the workload
        self.index = 0
        print(f"User {self.user_id} loaded {len(self.my_workload)} requests")

    # -------------------------------------------------------------------------
    # Section 2.2: Checkout Data Preparation
    # -------------------------------------------------------------------------
    def prepare_checkout_data(self, data):
        checkout_data = {
            "email": "mahmoud@example.com",
            "street_address": "1600 drasat  FGSR",
            "zip_code": "94043",
            "city": "mansoura",
            "state": "MN",
            "country": "Egypt",
            "credit_card_number": "4111111111111111",
            "credit_card_expiration_month": "12",
            "credit_card_expiration_year": "2030",
            "credit_card_cvv": "123",
        }
        if data:
            checkout_data.update(data)
        return checkout_data

    # -------------------------------------------------------------------------
    # Section 2.3: Workload Replay Task (Main Loop)
    # -------------------------------------------------------------------------
    @task
    def replay_workload(self):

        # Reset when workload is exhausted to loop indefinitely
        if self.index >= len(self.my_workload):
            self.index = 0
            self.start_time = time.time()
            return

        now = (time.time() - self.start_time) / TIME_SCALE  # Apply time compression

        while self.index < len(self.my_workload):
            req = self.my_workload[self.index]

            # Wait until the scheduled time for this request has arrived
            if req["time"] > now:
                break

            method = (req.get("method") or "GET").upper()
            url = req.get("url")

            # Skip malformed entries with no URL
            if not url:
                self.index += 1
                continue

            # Normalize checkout URL (remove trailing slash to avoid redirect)
            if "checkout" in url and url.endswith("/"):
                url = url[:-1]

            data = req.get("data", {}) or {}

            try:
                if method == "GET":
                    self.client.get(url, name=url)

                elif method == "POST":
                    # Checkout requires a fully structured form payload
                    if url == "/cart/checkout":
                        payload = self.prepare_checkout_data(data)
                        self.client.post(url, data=payload, name=url)
                    else:
                        self.client.post(url, data=data, name=url)

                else:
                    self.client.request(method, url, data=data, name=url)

            except Exception as e:
                print(f"Request failed: {method} {url} - {e}")

            self.index += 1

            # Throttle between consecutive requests from this virtual user.
            # Increased from 10ms to 100ms to reduce overload on the target service.
            time.sleep(0.1)