"""Command line entry point.

python -m part_equivalency demo            3-turn console conversation
python -m part_equivalency gap             as_agent().as_tool() vs. this pattern
python -m part_equivalency serve [--port]  AG-UI endpoint at POST /chat
"""

from __future__ import annotations

import argparse
import asyncio
import time

from dotenv import load_dotenv

from .agent import create_agent
from .gap import compare, render
from .llm import provider

DEMO_TURNS = (
    "Hi! What do you help with?",
    "I need a replacement cooling fan: 12V DC, 120mm, 0.25A, 3-pin connector",
    "Use the existing part please",
)


async def demo() -> None:
    agent = create_agent()
    session = agent.create_session()  # one conversation; history is kept across turns
    t0 = time.perf_counter()
    seen_calls: set[str] = set()
    for user in DEMO_TURNS:
        print(f"\n👤 {user}\n")
        in_text = False
        async for update in agent.run(user, stream=True, session=session):
            for c in update.contents:
                stamp = f"{time.perf_counter() - t0:5.2f}s"
                if c.type == "text_reasoning":
                    print(f"{chr(10) if in_text else ''}{stamp}   ▸ {(c.text or '').rstrip()}")
                    in_text = False
                elif c.type == "function_call" and c.call_id and c.call_id not in seen_calls:
                    seen_calls.add(c.call_id)  # real models stream a call in several chunks
                    print(f"{chr(10) if in_text else ''}{stamp} 🔧 {c.name} ({c.call_id})")
                    in_text = False
                elif c.type == "text" and c.text:
                    if not in_text:
                        print(f"{stamp} 🤖 ", end="")
                        in_text = True
                    print(c.text, end="", flush=True)
        print()


async def gap() -> None:
    for t in await compare():
        print(render(t))


def serve(host: str, port: int) -> None:
    import uvicorn

    from .server import create_app

    uvicorn.run(create_app(), host=host, port=port)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="part_equivalency", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("demo", help="3-turn console conversation (default)")
    sub.add_parser("gap", help="compare workflow.as_agent().as_tool() with this pattern")
    s = sub.add_parser("serve", help="run the AG-UI endpoint")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)

    load_dotenv()  # optional .env in the working directory

    print(f"[model provider: {provider()}]")
    if args.command == "serve":
        serve(args.host, args.port)
    elif args.command == "gap":
        asyncio.run(gap())
    else:
        asyncio.run(demo())


if __name__ == "__main__":
    main()
