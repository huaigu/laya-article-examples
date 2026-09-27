"""Chinese ticket example; no fabricated inference output or speed promise."""
import json

MODEL_REPO = "convaiinnovations/laya"
MODEL_REVISION = "55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851"
TICKET = "今天重复扣款了，请核实并退款；如果一直没有处理，我会取消订阅。"
QUESTIONS = {
    "department": {
        "type": "choice",
        "instructions": "这张工单应该分给哪个部门？",
        "criteria": {
            "billing": "账单、重复扣款、退款",
            "technical": "故障、报错、接口问题",
            "general": "其他咨询",
        },
    },
    "threatens_cancellation": {
        "type": "noul",
        "instructions": "客户是否明确表示要取消订阅或解除合同？",
    },
    "urgency": {
        "type": "score",
        "instructions": "这张工单的紧急程度如何？",
        "criteria": ["普通咨询", "需要尽快跟进", "明确的当日截止时间或严重损失"],
    },
}

# Demonstration policy, NOT empirically calibrated business thresholds.
REVIEW_BELOW = 0.85
ESCALATE_RISK_ABOVE = 0.80


def select_device():
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_agent():
    import laya
    return laya.load(
        MODEL_REPO, subfolder="multilingual",
        revision=MODEL_REVISION, device=select_device(),
    )


def choose_action(answers):
    """Priority: low confidence -> review; risk -> escalation; else route."""
    department = answers["department"]
    confidence = float(department["confidence"])
    risk = float(answers["threatens_cancellation"]["noul"])
    if not (0 <= confidence <= 1 and 0 <= risk <= 1):
        return "human_review"
    if confidence < REVIEW_BELOW:
        return "human_review"
    if risk > ESCALATE_RISK_ABOVE:
        return "priority_review"
    return "route_to_" + department["choice"]


def main():
    agent = load_agent()
    result = agent.predict(TICKET, QUESTIONS)
    print(json.dumps({"answers": result["answers"],
                      "action": choose_action(result["answers"])},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
