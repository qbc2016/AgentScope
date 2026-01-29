# -*- coding: utf-8 -*-
"""The realtime agent class."""
import asyncio
from asyncio import Queue

import shortuuid

from .. import logger
from ..module import StateModule
from ..realtime import (
    ModelEvents,
    RealtimeModelBase,
    ServerEvents,
    ClientEvents,
)


class RealtimeAgentBase(StateModule):
    """The realtime agent base class. Different from the `AgentBase` class,
    this class is designed for real-time interaction scenarios, such as
    realtime chat, voice assistants, etc.
    """

    def __init__(self, name: str, model: RealtimeModelBase) -> None:
        """Initialize the RealtimeAgentBase class.

        Args:
            name (`str`):
                The name of the agent.
            model (`RealtimeModelBase`):
                The realtime model used by the agent.
        """
        super().__init__()

        self.id = shortuuid.uuid()
        self.name = name
        self.model = model

        # A queue to handle the incoming events from other agents or the
        # frontend.
        self._incoming_queue = Queue()
        self._external_event_handling_task = None

        # The queue to gather model responses.
        self._model_response_queue = Queue()
        self._model_response_handling_task = None

    async def start(self, outgoing_queue: Queue) -> None:
        """Establish a connection for real-time interaction.

        Args:
            outgoing_queue (`Queue`):
                The queue to push messages to the frontend and other agents.
        """
        # Start the realtime model connection.
        await self.model.connect(self._model_response_queue)

        # Start the forwarding loop.
        self._external_event_handling_task = asyncio.create_task(
            self._forward_loop(),
        )

        # Start the response handling loop.
        self._model_response_handling_task = asyncio.create_task(
            self._model_response_loop(outgoing_queue),
        )

    async def stop(self) -> None:
        """Close the connection."""

        if not self._external_event_handling_task.done():
            self._external_event_handling_task.cancel()

        await self.model.disconnect()

    async def _forward_loop(self) -> None:
        """The loop to forward messages from other agents or the frontend to
        the realtime model for processing.

        outside ==> agent ==> realtime model
        """
        while True:
            event = await self._incoming_queue.get()

            match event:
                # TODO: handle both the server and client events, and send
                #  them to the realtime model as needed by the send method.
                # Only handle the events that we need
                case ServerEvents.AgentResponseAudioDeltaEvent() as event:
                    pass

    async def _model_response_loop(self, outgoing_queue: Queue) -> None:
        """The loop to handle model responses and forward them to the
        frontend and other agents.

        realtime model ==> agent ==> outside

        Args:
            outgoing_queue (`Queue`):
                The queue to push messages to the frontend and other agents.
        """
        while True:
            model_event = await self._model_response_queue.get()

            agent_kwargs = {"agent_id": self.id, "agent_name": self.name}

            agent_event = None
            match model_event:
                case ModelEvents.SessionCreatedEvent():
                    # Send the agent ready event to the outside.
                    agent_event = ServerEvents.AgentReadyEvent(**agent_kwargs)

                case ModelEvents.SessionEndedEvent():
                    # Send the agent session ended event to the outside.
                    agent_event = ServerEvents.AgentEndedEvent(**agent_kwargs)

                case ModelEvents.ResponseCreatedEvent() as event:
                    # The agent begins generating a response.
                    agent_event = ServerEvents.AgentResponseCreatedEvent(
                        response_id=event.response_id,
                        **agent_kwargs,
                    )

                case ModelEvents.ResponseDoneEvent() as event:
                    agent_event = ServerEvents.AgentResponseDoneEvent(
                        response_id=event.response_id,
                        input_tokens=event.input_tokens,
                        output_tokens=event.output_tokens,
                        metadata=event.metadata,
                        **agent_kwargs,
                    )

                case ModelEvents.ResponseAudioDeltaEvent() as event:
                    agent_event = ServerEvents.AgentResponseAudioDeltaEvent(
                        response_id=event.response_id,
                        item_id=event.item_id,
                        delta=event.delta,
                        format=event.format,
                        **agent_kwargs,
                    )

                case ModelEvents.ResponseAudioDoneEvent() as event:
                    agent_event = ServerEvents.AgentResponseAudioDoneEvent(
                        response_id=event.response_id,
                        item_id=event.item_id,
                        **agent_kwargs,
                    )

                case ModelEvents.ResponseAudioTranscriptDeltaEvent() as event:
                    agent_event = (
                        ServerEvents.AgentResponseAudioTranscriptDeltaEvent(
                            response_id=event.response_id,
                            item_id=event.item_id,
                            delta=event.delta,
                            **agent_kwargs,
                        )
                    )

                case ModelEvents.ResponseAudioTranscriptDoneEvent() as event:
                    agent_event = (
                        ServerEvents.AgentResponseAudioTranscriptDoneEvent(
                            response_id=event.response_id,
                            item_id=event.item_id,
                            **agent_kwargs,
                        )
                    )

                case ModelEvents.ResponseToolUseDeltaEvent() as event:
                    agent_event = ServerEvents.AgentResponseToolUseDeltaEvent(
                        response_id=event.response_id,
                        item_id=event.item_id,
                        name=event.name,
                        call_id=event.call_id,
                        delta=event.delta,
                        **agent_kwargs,
                    )

                case ModelEvents.ResponseToolUseDoneEvent() as event:
                    pass

                case ModelEvents.InputTranscriptionDeltaEvent() as event:
                    pass

                case ModelEvents.InputTranscriptionDoneEvent() as event:
                    pass

                case ModelEvents.InputStartedEvent() as event:
                    pass

                case ModelEvents.InputDoneEvent() as event:
                    pass

                case ModelEvents.ErrorEvent() as event:
                    pass

                case _:
                    logger.debug(
                        "Unknown model event type: %s",
                        type(model_event),
                    )

            if agent_event is not None:
                # Put the processed response to the outgoing queue.
                await outgoing_queue.put(agent_event)

    async def handle_input(self, event: ClientEvents | ServerEvents) -> None:
        """Handle the input message from the frontend or the other agents."""
        await self._incoming_queue.put(event)
