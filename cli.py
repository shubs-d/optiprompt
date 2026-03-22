"""CLI for OptiPrompt deterministic optimization."""

import argparse
import json
import sys

from app.core.pipeline import OptiPromptPipeline, PipelineConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="OptiPrompt CLI")
    parser.add_argument("prompt", nargs="?", help="Prompt string to optimize")
    parser.add_argument("--mode", choices=["aggressive", "balanced", "safe"], default="balanced")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--include-candidates", action="store_true")
    parser.add_argument("--disable-rl", action="store_true", help="Disable RL strategy selection")
    parser.add_argument("--rl-epsilon", type=float, default=0.05, help="Exploration rate (0.0-1.0)")
    parser.add_argument("--rl-alpha", type=float, default=0.2, help="Q-learning rate (0.0-1.0)")
    parser.add_argument("--rl-gamma", type=float, default=0.0, help="Discount factor (0.0-1.0)")
    parser.add_argument(
        "--rl-q-table-path",
        type=str,
        default=".optiprompt_q_table.json",
        help="JSON path for persisted RL Q-table",
    )
    parser.add_argument("--input-file", type=str, default="", help="Read prompt from text file")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.input_file:
        with open(args.input_file, "r", encoding="utf-8") as fh:
            prompt = fh.read().strip()
    else:
        prompt = (args.prompt or "").strip()

    if not prompt:
        print("Error: prompt is required via positional arg or --input-file", file=sys.stderr)
        return 2

    pipeline = OptiPromptPipeline()
    cfg = PipelineConfig(
        mode=args.mode,
        seed=args.seed,
        include_candidates=args.include_candidates,
        debug=args.debug,
        rl_enabled=not args.disable_rl,
        rl_q_table_path=args.rl_q_table_path,
        rl_alpha=args.rl_alpha,
        rl_gamma=args.rl_gamma,
        rl_epsilon=args.rl_epsilon,
    )
    result = pipeline.optimize(prompt, cfg)
    print(json.dumps(result, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
