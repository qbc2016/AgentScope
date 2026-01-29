# -*- coding: utf-8 -*-
"""The websocket events generated from the realtime agent and backend."""
from dataclasses import dataclass
from enum import Enum
from typing import Literal


class ServerEventType(str, Enum):
    """Types of agent events for backend-to-web communication."""

    # Session lifecycle
    SESSION_CREATED = "session_created"
    SESSION_UPDATED = "session_updated"
    SESSION_ENDED = "session_ended"

    # ============== AGENT LIFECYCLE EVENTS ================

    AGENT_READY = "agent_ready"
    """The agent is created and ready to receive inputs."""

    AGENT_ENDED = "agent_ended"
    """The agent ended."""

    # ============== AGENT RESPONSE EVENTS =================

    # Response events
    RESPONSE_CREATED = "response_created"
    """The agent starts generating a response."""

    RESPONSE_CANCELLED = "response_cancelled"
    """The agent's response generation is interrupted/cancelled."""

    RESPONSE_DONE = "response_done"
    """The agent finished generating a response."""

    # ============== Response content events =================

    RESPONSE_AUDIO_DELTA = "response_audio_delta"
    """The agent's response audio data delta."""

    RESPONSE_AUDIO_DONE = "response_audio_done"
    """The agent's response audio data is complete."""

    RESPONSE_AUDIO_TRANSCRIPT_DELTA = "response_audio_transcript_delta"
    """The agent's response audio transcription delta."""

    RESPONSE_AUDIO_TRANSCRIPT_DONE = "response_audio_transcript_done"
    """The agent's response audio transcription is complete."""

    RESPONSE_TOOL_USE_DELTA = "response_tool_use_delta"
    """The agent's response tool use data delta."""

    RESPONSE_TOOL_USE_DONE = "response_tool_use_done"
    """The agent's response tool use data is complete."""

    # ============== INPUT AUDIO TRANSCRIPTION EVENTS =================

    INPUT_TRANSCRIPTION_DELTA = "input_transcription_delta"
    """The input audio transcription delta."""

    INPUT_TRANSCRIPTION_DONE = "input_transcription_done"
    """The input audio transcription is complete."""

    # Input detection
    INPUT_STARTED = "input_started"
    """Detected the start of user input audio."""

    INPUT_DONE = "input_done"
    """Detected the end of user input audio."""

    # ============== ERROR EVENTS =================

    ERROR = "error"


class ServerEvents:
    """Realtime server events."""

    class SessionCreatedEvent:
        """Session created event in the backend"""

        type: ServerEventType = ServerEventType.SESSION_CREATED
        """The event type."""

        session_id: str
        """The session ID."""

    class SessionUpdatedEvent:
        """Session updated event in the backend"""

        type: ServerEventType = ServerEventType.SESSION_UPDATED
        """The event type."""

        session_id: str
        """The session ID."""

    class SessionEndedEvent:
        """Session ended event in the backend"""

        type: ServerEventType = ServerEventType.SESSION_ENDED
        """The event type."""

        session_id: str
        """The session ID."""

    @dataclass
    class AgentReadyEvent:
        """Agent ready event in the backend"""

        agent_id: str
        """The agent ID."""

        agent_name: str
        """The agent name."""

        type: ServerEventType = ServerEventType.AGENT_READY
        """The event type."""

    @dataclass
    class AgentEndedEvent:
        """Agent ended event in the backend"""

        agent_id: str
        """The agent ID."""

        agent_name: str
        """The agent name."""

        type: ServerEventType = ServerEventType.AGENT_ENDED
        """The event type."""

    @dataclass
    class AgentResponseCreatedEvent:
        """Response created event in the backend"""

        response_id: str
        """The response ID."""

        agent_id: str
        """The agent ID."""

        agent_name: str
        """The agent name."""

        type: ServerEventType = ServerEventType.RESPONSE_CREATED
        """The event type."""

    @dataclass
    class AgentResponseDoneEvent:
        """Response done event in the backend"""

        response_id: str
        """The response ID."""

        input_tokens: int
        """The number of input tokens used."""

        output_tokens: int
        """The number of output tokens used."""

        metadata: dict[str, str]
        """Additional metadata about the response."""

        agent_id: str
        """The agent ID."""

        agent_name: str
        """The agent name."""

        type: Literal[
            ServerEventType.RESPONSE_DONE
        ] = ServerEventType.RESPONSE_DONE
        """The event type."""

    @dataclass
    class AgentResponseAudioDeltaEvent:
        """Response audio delta event in the backend"""

        response_id: str
        """The response ID."""

        item_id: str
        """The response item ID."""

        delta: str
        """The audio chunk data, encoded as base64 string."""

        format: dict
        """The audio format information."""

        agent_id: str
        """The agent ID."""

        agent_name: str
        """The agent name."""

        type: Literal[
            ServerEventType.RESPONSE_AUDIO_DELTA
        ] = ServerEventType.RESPONSE_AUDIO_DELTA
        """The event type."""

    @dataclass
    class AgentResponseAudioDoneEvent:
        """Response audio done event in the backend"""

        response_id: str
        """The response ID."""

        item_id: str
        """The response item ID."""

        agent_id: str
        """The agent ID."""

        agent_name: str
        """The agent name."""

        type: Literal[
            ServerEventType.RESPONSE_AUDIO_DONE
        ] = ServerEventType.RESPONSE_AUDIO_DONE

    @dataclass
    class AgentResponseAudioTranscriptDeltaEvent:
        """Response audio transcript delta event in the backend"""

        response_id: str
        """The response ID."""

        item_id: str
        """The response item ID."""

        delta: str
        """The transcript chunk data."""

        agent_id: str
        """The agent ID."""

        agent_name: str
        """The agent name."""

        type: Literal[
            ServerEventType.RESPONSE_AUDIO_TRANSCRIPT_DELTA
        ] = ServerEventType.RESPONSE_AUDIO_TRANSCRIPT_DELTA
        """The event type."""

    @dataclass
    class AgentResponseAudioTranscriptDoneEvent:
        """Response audio transcript done event in the backend"""

        response_id: str
        """The response ID."""

        item_id: str
        """The response item ID."""

        agent_id: str
        """The agent ID."""

        agent_name: str
        """The agent name."""

        type: Literal[
            ServerEventType.RESPONSE_AUDIO_TRANSCRIPT_DONE
        ] = ServerEventType.RESPONSE_AUDIO_TRANSCRIPT_DONE
        """The event type."""

    @dataclass
    class AgentResponseToolUseDeltaEvent:
        """Response tool use delta event in the backend"""

        response_id: str
        """The response ID."""

        item_id: str
        """The response item ID."""

        name: str
        """The tool name."""

        call_id: str
        """The tool call ID."""

        delta: str  # argument JSON string delta
        """The tool use delta data."""

        agent_id: str
        """The agent ID."""

        agent_name: str
        """The agent name."""

        type: Literal[
            ServerEventType.RESPONSE_TOOL_USE_DELTA
        ] = ServerEventType.RESPONSE_TOOL_USE_DELTA
        """The event type."""

    class AgentResponseToolUseDoneEvent:
        """Response tool use done event in the backend"""

        type: ServerEventType = ServerEventType.RESPONSE_TOOL_USE_DONE
        """The event type."""

        response_id: str
        """The response ID."""

        item_id: str
        """The response item ID."""

        call_id: str
        """The tool call ID."""

        agent_id: str
        """The agent ID."""

        agent_name: str
        """The agent name."""

    class InputTranscriptionDeltaEvent:
        """Input transcription delta event in the backend"""

        type: ServerEventType = ServerEventType.INPUT_TRANSCRIPTION_DELTA
        """The event type."""

        delta: str
        """The transcription chunk data."""

        agent_id: str
        """The agent ID."""

        agent_name: str
        """The agent name."""

    class InputTranscriptionDoneEvent:
        """Input transcription done event in the backend"""

        type: ServerEventType = ServerEventType.INPUT_TRANSCRIPTION_DONE
        """The event type."""

        transcript: str
        """The complete transcription text."""

        input_tokens: int
        """The number of input tokens."""

        output_tokens: int
        """The number of output tokens."""

        agent_id: str
        """The agent ID."""

        agent_name: str
        """The agent name."""

    class InputStartedEvent:
        """Input started event in the backend"""

        type: ServerEventType = ServerEventType.INPUT_STARTED
        """The event type."""

        agent_id: str
        """The agent ID."""

        agent_name: str
        """The agent name."""

    class InputDoneEvent:
        """Input done event in the backend"""

        type: ServerEventType = ServerEventType.INPUT_DONE
        """The event type."""

        agent_id: str
        """The agent ID."""

        agent_name: str
        """The agent name."""

    class ErrorEvent:
        """Error event in the backend"""

        type: ServerEventType = ServerEventType.ERROR
        """The event type."""

        error_type: str
        """The error type."""

        code: str
        """The error code."""

        message: str
        """The error message."""
