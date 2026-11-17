"""Run the agent from the terminal: ``python -m agent "How many customers churned?"``"""

import sys

from .keys import ensure_keys


def main() -> None:
    ensure_keys("TABPFN_TOKEN", "ANTHROPIC_API_KEY", optional=("TYPESAFE_API_KEY",))
    from .agent import PredictiveAgent

    agent = PredictiveAgent()
    if len(sys.argv) > 1:
        print(agent.ask(" ".join(sys.argv[1:])).answer)
        return
    print("Predictive agent. Ask about the telco customers (Ctrl-C to quit).")
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q:
            print(agent.ask(q).answer)


if __name__ == "__main__":
    main()
