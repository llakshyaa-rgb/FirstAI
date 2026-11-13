"""Run an end-to-end training, checkpoint, resume and generation smoke test.

By default all temporary artifacts are deleted. Use --keep-artifacts to retain
the synthetic corpus and checkpoints for inspection/download from a cloud runtime.
"""
import argparse
import subprocess
import sys
import tempfile
import time
from pathlib import Path
import torch


def run(command, cwd):
    print("+", " ".join(map(str, command)), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-cuda", action="store_true",
                        help="Fail unless a CUDA GPU is attached.")
    parser.add_argument("--keep-artifacts", action="store_true",
                        help="Keep synthetic corpus and smoke checkpoints in output directory.")
    parser.add_argument("--output-dir", default="artifacts/gpu-smoke")
    args = parser.parse_args()

    if args.require_cuda and not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU was required, but torch.cuda.is_available() is False.")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Smoke-test device: {device}", flush=True)
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)
        print(f"CUDA runtime: {torch.version.cuda}", flush=True)
        torch.cuda.reset_peak_memory_stats()

    root = Path(__file__).resolve().parents[1]
    temp_context = None
    if args.keep_artifacts:
        work = (root / args.output_dir).resolve()
        work.mkdir(parents=True, exist_ok=True)
    else:
        temp_context = tempfile.TemporaryDirectory(prefix="firstai-smoke-")
        work = Path(temp_context.name)

    started = time.perf_counter()
    try:
        corpus = work / "smoke.txt"
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
            "--seed", "17", "--amp",
        ]
        run([sys.executable, "train.py", *common, "--steps", "3"], root)

        best = checkpoint.with_name(checkpoint.stem + ".best.pt")
        if not checkpoint.is_file() or not best.is_file():
            raise RuntimeError("Training did not create both latest and best checkpoints")

        run([
            sys.executable, "train.py", *common,
            "--steps", "1", "--resume",
        ], root)

        latest = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if latest.get("step") != 4:
            raise RuntimeError(f"Resume expected step 4, found {latest.get('step')}")

        run([
            sys.executable, "evaluate.py",
            "--checkpoint", str(best), "--data", str(corpus),
            "--batches", "2", "--batch-size", "2",
        ], root)

        run([
            sys.executable, "generate.py",
            "--checkpoint", str(best),
            "--prompt", "A machine", "--tokens", "12",
            "--temperature", "0.8", "--top-k", "10",
        ], root)

        if device == "cuda":
            torch.cuda.synchronize()
            print(f"Peak allocated GPU memory: {torch.cuda.max_memory_allocated() / 1024**2:.1f} MiB")
        print(f"Total smoke-test time: {time.perf_counter() - started:.2f} seconds")
        if args.keep_artifacts:
            print(f"Artifacts retained at: {work}")
            print(f"Latest checkpoint: {checkpoint}")
            print(f"Best checkpoint: {best}")
    finally:
        if temp_context is not None:
            temp_context.cleanup()

    print("\nFIRSTAI END-TO-END SMOKE TEST: PASS")
    print("Training, checkpoint creation, resume, reload and generation all worked.")


if __name__ == "__main__":
    main()
