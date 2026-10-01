import random

import pandas as pd
from locust import HttpUser, task, between


DATA_PATH = (
    "data/processed/twitch/"
    "twitch_test_ranking.csv.gz"
)

df = pd.read_csv(
    DATA_PATH,
    usecols=["user_id"],
)

USER_IDS = (
    df["user_id"]
    .drop_duplicates()
    .astype(int)
    .tolist()
)

HOT_USERS = USER_IDS[:20]
LARGE_USER_POOL = USER_IDS[:5000]


class RankingUser(HttpUser):

    wait_time = between(0.05, 0.2)

    def send_request(self, user_id):
        with self.client.get(
            f"/recommend/{user_id}",
            params={
                "top_k": 10,
                "candidate_k": 200,
            },
            name="/recommend",
            catch_response=True,
        ) as response:

            if response.status_code != 200:
                response.failure(
                    f"HTTP {response.status_code}"
                )
                return

            try:
                body = response.json()
            except Exception:
                response.failure(
                    "Invalid JSON response"
                )
                return

            if body.get("cache_hit"):
                response.request_meta[
                    "name"
                ] = "/recommend/cache-hit"
            else:
                response.request_meta[
                    "name"
                ] = "/recommend/cache-miss"

            response.success()

    @task(8)
    def hot_user(self):
        self.send_request(
            random.choice(HOT_USERS)
        )

    @task(2)
    def broad_user(self):
        self.send_request(
            random.choice(LARGE_USER_POOL)
        )
