from typing import Dict


def format_hallucination_input(row: Dict) -> str:
    """Format examples so the answer is preserved when long context is truncated."""
    return (
        "Câu hỏi:\n"
        f"{row.get('question', '')}\n\n"
        "Câu trả lời cần kiểm chứng:\n"
        f"{row.get('answer', '')}\n\n"
        "Ngữ cảnh:\n"
        f"{row.get('context', '')}"
    )
