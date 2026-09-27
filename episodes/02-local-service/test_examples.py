"""No model download. Tests routing logic and HTTP wiring with an explicit fake."""
import unittest
from unittest.mock import patch

from quickstart import choose_action


def answers(confidence, risk):
    return {"department": {"choice": "billing", "confidence": confidence},
            "threatens_cancellation": {"noul": risk}}


class PolicyTests(unittest.TestCase):
    def test_low_confidence_wins_over_high_risk(self):
        self.assertEqual(choose_action(answers(.07, 1)), "human_review")

    def test_confident_high_risk(self):
        self.assertEqual(choose_action(answers(.9, .9)), "priority_review")

    def test_ordinary_route(self):
        self.assertEqual(choose_action(answers(.9, .2)), "route_to_billing")

    def test_invalid_probability(self):
        self.assertEqual(choose_action(answers(float('nan'), .2)), "human_review")

    def test_threshold_edges(self):
        self.assertEqual(choose_action(answers(.85, .80)), "route_to_billing")
        self.assertEqual(choose_action(answers(.849, .80)), "human_review")


class FakeAgent:
    def predict(self, text, questions):
        return {"answers": answers(.07, 1)}


class ServiceTests(unittest.TestCase):
    def setUp(self):
        import server
        from fastapi.testclient import TestClient
        self.server = server
        self.patch = patch.object(server, "load_agent", return_value=FakeAgent())
        self.patch.start()
        self.client = TestClient(server.app)
        self.client.__enter__()

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.patch.stop()

    def test_response(self):
        self.assertEqual(self.client.get("/readyz").status_code, 200)
        response = self.client.post("/triage", json={"text": "重复扣款"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["action"], "human_review")

    def test_busy_rejects_instead_of_unbounded_wait(self):
        self.server.app.state.gate.acquire()
        try:
            response = self.client.post("/triage", json={"text": "重复扣款"})
            self.assertEqual(response.status_code, 503)
            self.assertEqual(response.headers["Retry-After"], "1")
        finally:
            self.server.app.state.gate.release()

    def test_input_limits(self):
        for text in ("", "x" * 1001):
            self.assertEqual(self.client.post("/triage", json={"text": text}).status_code, 422)


if __name__ == "__main__":
    unittest.main()
