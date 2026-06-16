from typing import Dict


def format_hallucination_input(row: Dict) -> str:
    return (
        "Câu hỏi:\n"
        f"{row.get('question', '')}\n\n"
        "Ngữ cảnh:\n"
        f"{row.get('context', '')}\n\n"
        "Câu trả lời cần kiểm chứng:\n"
        f"{row.get('answer', '')}"
    )
