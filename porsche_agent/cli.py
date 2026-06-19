import argparse
import sys
from pathlib import Path
from porsche_agent.config import Config
from porsche_agent.llm import create_provider
from porsche_agent.agent import Agent
from porsche_agent.builtin_tools import shell_command, read_file, write_file
from porsche_agent.memory.context import ContextManager
from porsche_agent.memory.short_term import ShortTermMemory
from porsche_agent.memory.long_term import LongTermMemory

BUILTIN_TOOLS = [shell_command, read_file, write_file]


def cmd_chat(args):
    config = Config.from_env()
    llm = create_provider(config)
    system_prompt = args.system_prompt or config.system_prompt
    max_iterations = args.max_iterations or config.max_iterations

    context_mgr = ContextManager(
        llm=llm,
        max_window=config.context_window,
        summary_trigger=config.context_summary_trigger,
    )

    stm = ShortTermMemory() if config.short_term_memory else None

    ltm = None
    if config.long_term_memory:
        ltm = LongTermMemory(
            embedding_client=llm.embedding_client,
            embedding_model=config.embedding_model,
            vector_store_url=config.vector_store_url,
        )

    agent = Agent(
        llm=llm,
        tools=list(BUILTIN_TOOLS),
        system_prompt=system_prompt,
        max_iterations=max_iterations,
        context_manager=context_mgr,
        short_term_memory=stm,
        long_term_memory=ltm,
    )

    print(f"PorscheAgent chat [{config.llm_provider}:{config.model}]")
    print(f"  context window={config.context_window}, stm={stm is not None}, ltm={ltm is not None}")
    while True:
        try:
            line = input("> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        line = line.strip()
        if not line:
            continue
        if line == "/exit":
            break
        try:
            result = agent.run(line)
        except RuntimeError as e:
            print(f"Agent error: {e}")
        else:
            print(result)


def cmd_run(args):
    script = Path(args.script)
    if not script.exists():
        print(f"Script not found: {args.script}")
        sys.exit(1)
    code = compile(script.read_text(), args.script, "exec")
    exec(code, {"__name__": "__main__", "__file__": str(script.absolute())})


def main():
    parser = argparse.ArgumentParser(prog="porsche")
    sub = parser.add_subparsers(dest="command")

    chat = sub.add_parser("chat", help="Start interactive chat session")
    chat.add_argument("--system-prompt", help="System prompt for the agent")
    chat.add_argument(
        "--max-iterations", type=int, default=10, help="Max tool-use iterations"
    )

    run = sub.add_parser("run", help="Execute a script with PorscheAgent")
    run.add_argument("script", help="Path to the Python script")

    args = parser.parse_args()
    if args.command == "chat":
        cmd_chat(args)
    elif args.command == "run":
        cmd_run(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
