from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)


class Evaluator:
    def __init__(self):
        self.smoothing = SmoothingFunction().method3

    def compute_bleu(self, reference: str, hypothesis: str) -> float:
        """
        Compute sentence-level BLEU score between a reference and hypothesis.
        Both should be in the same language (the translated output language).

        Args:
            reference:   The reference/expected translation (str)
            hypothesis:  The model's translation output (str)

        Returns:
            BLEU score as a float between 0.0 and 1.0
        """
        ref_tokens  = nltk.word_tokenize(reference.lower())
        hyp_tokens  = nltk.word_tokenize(hypothesis.lower())

        if not hyp_tokens:
            return 0.0

        score = sentence_bleu(
            [ref_tokens],
            hyp_tokens,
            smoothing_function=self.smoothing
        )
        return round(score, 4)

    def evaluate(self, reference: str, hypothesis: str, confidence: float) -> dict:
        """
        Full evaluation of a single translation.

        Args:
            reference:   User-provided reference translation
            hypothesis:  Model's translation
            confidence:  Language detection confidence (0–100)

        Returns:
            dict with bleu, confidence, bleu_label, confidence_label
        """
        bleu = self.compute_bleu(reference, hypothesis)

        return {
            "bleu":              bleu,
            "bleu_percent":      round(bleu * 100, 2),
            "bleu_label":        self._bleu_label(bleu),
            "confidence":        round(confidence, 2),
            "confidence_label":  self._confidence_label(confidence),
        }

    @staticmethod
    def _bleu_label(score: float) -> str:
        if score >= 0.6:  return "Excellent"
        if score >= 0.4:  return "Good"
        if score >= 0.2:  return "Fair"
        return "Poor"

    @staticmethod
    def _confidence_label(confidence: float) -> str:
        if confidence >= 90:  return "Very High"
        if confidence >= 75:  return "High"
        if confidence >= 50:  return "Moderate"
        return "Low"