"""Run a tiny end-to-end training, resume, checkpoint and generation smoke test.

Uses a temporary synthetic corpus only to verify the engineering pipeline.
It does not create meaningful training data or define FirstAI's identity.
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def run(command, cwd):
    print("+", " ".join(map(str, command)), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def main():
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="firstai-smoke-") as temp:
        work = Path(temp)
        corpus = work / "smoke.txt"
        # Repetition supplies enough tokens for both train and validation splits.
        paragraph = (
            "A machine learns patterns by predicting the next token. "
            "The training loop measures error, computes gradients, and updates weights. "
            "A checkpoint stores model parameters so a later process can continue training. "
            "Validation measures performance on text not used for gradient updates.\n\n"
        )
        corpus.write_text(paragraph * 80, encoding="utf-8")
        checkpoint = work / "checkpoints" / "smoke.pt"
        common = [
            "--data", str(corpus), "--out", str(checkpoint),
            "--batch-size", "2", "--block-size", "16",
            "--layers", "1", "--heads", "2", "--width", "32",
            "--vocab-size", "300", "--min-frequency", "2",
            "--eval-interval", "2", "--checkpoint-interval", "2",
            "--seed", "17",
        ]
        run([sys.executable, "train.py", *common, "--steps", "3"], root)

        best = checkpoint.with_name(checkpoint.stem + ".best.pt")
        if not checkpoint.is_file() or not best.is_file():
            raise RuntimeError("Training did not create both latest and best checkpoints")

        run([
            sys.executable, "train.py", *common,
            "--steps", "1", "--resume",
        ], root)

        latest = __import__("torch").load(checkpoint, map_location="cpu", weights_only=True)
        if latest.get("step") != 4:
            raise RuntimeError(f"Resume expected step 4, found {latest.get('step')}")

        run([
            sys.executable, "generate.py",
            "--checkpoint", str(best),
            "--prompt", "A machine", "--tokens", "12",
            "--temperature", "0.8", "--top-k", "10",
        ], root)

    print("\nFIRSTAI END-TO-END SMOKE TEST: PASS")
    print("Training, best/latest checkpoint creation, resume, reload and generation all worked.")


if __name__ == "__main__":
    main()
