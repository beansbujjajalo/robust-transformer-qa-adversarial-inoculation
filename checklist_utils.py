from dataclasses import dataclass
from typing import List, Dict
import torch
from model_utils import load_finetuned_model

# NOTE: Lightweight CheckList implementation (Ribeiro et al., 2020).
# For this specific project, we define a small static suite of 6 examples 
# to verify basic linguistic capabilities, rather than importing the full library.
CHECKLIST_SUITE = [
    {
        "category": "negation",
        "context": "Alice visited Paris in 2019, but she did not visit London that year. Her brother Bob visited London in 2020 instead.",
        "question": "Which city did Alice visit in 2019?",
        "answer": "Paris"
    },
    {
        "category": "negation",
        "context": "The company did not hire John for the manager role; instead, they hired Sarah. John remained in his previous position.",
        "question": "Who was hired for the manager role?",
        "answer": "Sarah"
    },
    {
        "category": "distractor",
        "context": "Michael Jordan was a famous basketball player for the Chicago Bulls. A different Michael, Michael Smith, once played basketball for a local high school.",
        "question": "Which team did Michael Jordan play for?",
        "answer": "Chicago Bulls"
    },
    {
        "category": "distractor",
        "context": "The Great Wall of China is a historic fortification in China. A replica wall was built in a theme park in another country.",
        "question": "In which country is the Great Wall located?",
        "answer": "China"
    },
    {
        "category": "entity_swap",
        "context": "In 1999, the festival took place in Berlin. In 2001, the festival moved to Madrid, where it attracted many visitors.",
        "question": "In which city did the festival take place in 1999?",
        "answer": "Berlin"
    },
    {
        "category": "entity_swap",
        "context": "Dr. Smith treated patients in Boston, while Dr. Lee worked in Seattle. Both doctors specialized in cardiology.",
        "question": "Which city did Dr. Smith work in?",
        "answer": "Boston"
    }
]

@dataclass
class CheckListExample:
    category: str
    context: str
    question: str
    answer: str

def build_checklist_tests() -> List[CheckListExample]:
    return [CheckListExample(**ex) for ex in CHECKLIST_SUITE]

def _get_answer_from_logits(start_logits, end_logits, input_ids, tokenizer, max_answer_length: int = 30) -> str:
    start_idx = int(torch.argmax(start_logits))
    end_idx = int(torch.argmax(end_logits))
    if end_idx < start_idx:
        end_idx = start_idx
    if end_idx - start_idx + 1 > max_answer_length:
        end_idx = start_idx + max_answer_length - 1

    tokens = input_ids[start_idx : end_idx + 1]
    return tokenizer.decode(tokens, skip_special_tokens=True)

@dataclass
class CheckListResult:
    category: str
    total: int
    correct: int

def run_checklist(model_dir: str) -> List[CheckListResult]:
    tokenizer, model = load_finetuned_model(model_dir)
    model.eval()

    tests = build_checklist_tests()
    by_cat: Dict[str, List[bool]] = {}

    for ex in tests:
        enc = tokenizer(
            ex.question,
            ex.context,
            return_tensors="pt",
            truncation="only_second",
            max_length=256,
        )
        enc = {k: v.to(model.device) for k, v in enc.items()}

        with torch.no_grad():
            outputs = model(**enc)

        answer_pred = _get_answer_from_logits(
            outputs.start_logits[0],
            outputs.end_logits[0],
            enc["input_ids"][0],
            tokenizer,
        ).strip()

        correct = ex.answer.lower() in answer_pred.lower()
        by_cat.setdefault(ex.category, []).append(correct)

    results: List[CheckListResult] = []
    for cat, vals in by_cat.items():
        results.append(
            CheckListResult(
                category=cat,
                total=len(vals),
                correct=int(sum(vals)),
            )
        )
    return results

def format_checklist_results(results: List[CheckListResult]) -> str:
    lines = ["CheckList-style evaluation:"]
    for r in results:
        acc = r.correct / r.total if r.total > 0 else 0.0
        lines.append(
            f"- {r.category}: {r.correct}/{r.total} ({acc * 100:.1f}% approx. EM)"
        )
    return "\n".join(lines)