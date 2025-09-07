# -*- coding: utf-8 -*-
"""Parallel Multi-Perspective Discussion System."""
import asyncio
from datetime import datetime
from typing import Any

from agentscope.agent import AgentBase
from agentscope.message import Msg
from agentscope.pipeline import fanout_pipeline


class ExampleAgent(AgentBase):
    """The example agent used to label the time."""

    def __init__(self, name: str) -> None:
        """The constructor of the example agent

        Args:
            name (`str`):
                The agent name.
        """
        super().__init__()
        self.name = name

    async def reply(self, *args: Any, **kwargs: Any) -> Msg:
        """The reply function of the example agent."""
        # we record the start time
        start_time = datetime.now().strftime("%H:%M:%S.%f")
        await self.print(
            Msg(
                self.name,
                f"begins at {start_time}",
                "assistant",
            ),
        )

        # Sleep 3 seconds
        await asyncio.sleep(3)

        end_time = datetime.now().strftime("%H:%M:%S.%f")
        msg = Msg(
            self.name,
            f"finishes at {end_time}",
            "user",
        )
        await self.print(msg)
        return msg

    async def handle_interrupt(
        self,
        *args: Any,
        **kwargs: Any,
    ) -> Msg:
        """We leave this function unimplemented in this example, because we
        won't use the interrupt functionality"""

    async def observe(self, *args: Any, **kwargs: Any) -> None:
        """Similar with the handle_interrupt function, leaving this empty"""


async def main() -> None:
    """The main entry of the concurrent example."""
    alice = ExampleAgent("Alice")
    bob = ExampleAgent("Bob")
    chalice = ExampleAgent("Chalice")

    print("Use 'asyncio.gather' to run the agents concurrently:")
    futures = [alice(), bob(), chalice()]

    await asyncio.gather(*futures)

    print("\n\nUser fanout pipeline to run the agents concurrently:")
    await fanout_pipeline(
        agents=[alice, bob, chalice],
        enable_gather=True,
    )


asyncio.run(main())
