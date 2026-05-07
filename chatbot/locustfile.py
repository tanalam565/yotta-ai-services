from locust import HttpUser, task, between
import random
import logging

class ChatbotUser(HttpUser):
    wait_time = between(1, 3)  # Wait 1-3 seconds between requests
    
    def on_start(self):
        """Called when a user starts"""
        self.session_id = f"load-test-{random.randint(1000, 9999)}"
        self.api_key = "your-secret-api-key-here"
    
    @task(3)  # Weight 3 - happens 3x more than health check
    def send_chat_message(self):
        """Simulate a chat message"""
        queries = [
            "What are the move-out procedures?",
            "Tell me about the security deposit policy",
            "What is the pet policy?",
            "How do I submit a maintenance request?",
            "What are the lease renewal terms?",
        ]

        try:
            with self.client.post(
                "/api/chat",
                json={
                    "message": random.choice(queries),
                    "session_id": self.session_id
                },
                headers={"X-API-Key": self.api_key},
                catch_response=True,
                timeout=90
            ) as response:
                if response.status_code == 200:
                    response.success()
                elif response.status_code == 429:
                    response.failure(f"Rate limited (429) - reduce user count or increase RATE_LIMIT_CHAT")
                else:
                    response.failure(f"HTTP {response.status_code}: {response.text[:200]}")
        except Exception as e:
            logging.error("Chat task exception: %s", e)
    
    @task(1)  # Weight 1 - happens less frequently
    def check_health(self):
        """Health check"""
        self.client.get("/api/health")