"""
Reward functions for GRPO training.
"""

import re
import torch


class ExactMatchReward:
    """
    Computes binary exact-match reward by comparing the final answer
    extracted from decoded text against ground-truth solutions.
    """

    def extract_answer(self, text):
        """
        Extract the final numerical/boxed answer from decoded text.
        Tries multiple patterns in order of specificity.
        """
        if not text:
            return None

        # Pattern 1: \boxed{...}
        boxed_match = re.findall(r'\\boxed\{([^}]*)\}', text)
        if boxed_match:
            return boxed_match[-1].strip()

        # Pattern 2: "The answer is X" or "the answer is X"
        answer_match = re.search(r'[Tt]he\s+answer\s+is\s+[:\s]*([^\.\n,]+)', text)
        if answer_match:
            return answer_match.group(1).strip()

        # Pattern 3: "= X" at the end of text (final equation result)
        eq_match = re.search(r'=\s*([^\.\n,=]+)\s*$', text.strip())
        if eq_match:
            return eq_match.group(1).strip()

        # Pattern 4: last number in the text
        numbers = re.findall(r'-?\d+(?:\.\d+)?(?:/\d+)?', text)
        if numbers:
            return numbers[-1]

        return None

    def normalize_answer(self, answer):
        """Normalize answer string for comparison."""
        if answer is None:
            return None
        answer = str(answer).strip()
        # Remove trailing periods, whitespace
        answer = answer.rstrip('.').strip()
        # Try to evaluate as number for numeric comparison
        try:
            return str(float(answer))
        except (ValueError, TypeError):
            return answer.lower()

    def compute_rewards(self, decoded_texts, gt_solutions):
        """
        Compute binary exact-match rewards.

        Args:
            decoded_texts: list of decoded text strings (length B*G)
            gt_solutions: list of ground-truth solution strings (length B*G)

        Returns:
            torch.Tensor of shape (B*G,) with values in {0, 1}
        """
        rewards = []
        for decoded, gt in zip(decoded_texts, gt_solutions):
            pred_answer = self.normalize_answer(self.extract_answer(decoded))
            gt_answer = self.normalize_answer(self.extract_answer(gt))

            if pred_answer is not None and gt_answer is not None and pred_answer == gt_answer:
                rewards.append(1.0)
            elif decoded and gt and gt in decoded:
                # Fallback: check if gt solution text appears in decoded text
                rewards.append(1.0)
            else:
                rewards.append(0.0)

        return torch.tensor(rewards, dtype=torch.float32)
