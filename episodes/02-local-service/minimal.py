import laya

agent = laya.load(
    "convaiinnovations/laya",
    subfolder="multilingual",
    revision="55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851",
)

questions = {
    "department": {
        "type": "choice",
        "instructions": "这张工单应该分给哪个部门？",
        "criteria": {
            "billing": "账单、重复扣款、退款",
            "technical": "故障、报错、接口问题",
            "general": "其他咨询",
        },
    }
}

result = agent.predict("今天重复扣款了，请核实并退款。", questions)
print(result["answers"]["department"])
