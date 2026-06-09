# -*- coding: utf-8 -*-
"""Middleware that turns reasoning text into speech and injects it as
``DATA_BLOCK_*`` events into the agent's event stream."""
import uuid
from typing import TYPE_CHECKING, AsyncGenerator, Callable

from ._base import MiddlewareBase
from ..event import (
    DataBlockDeltaEvent,
    DataBlockEndEvent,
    DataBlockStartEvent,
    TextBlockDeltaEvent,
    TextBlockEndEvent,
)
from ..message import Msg, TextBlock
from ..tts import TTSModelBase, TTSResponse

if TYPE_CHECKING:
    from ..agent import Agent


class TTSMiddleware(MiddlewareBase):
    """Synthesize speech for every text block produced during reasoning and
    inject the audio as ``DATA_BLOCK_*`` events into the stream.

    - Non-realtime TTS (``supports_streaming_input=False``): on each
      ``TextBlockEndEvent`` the accumulated text is sent to
      :meth:`TTSModelBase.synthesize`; the resulting audio chunks are
      emitted as one ``DATA_BLOCK_START`` + N ``DATA_BLOCK_DELTA`` +
      ``DATA_BLOCK_END``.
    - Realtime TTS (``supports_streaming_input=True``): each
      ``TextBlockDeltaEvent`` is pushed into the model via
      :meth:`TTSModelBase.push`; any audio produced is emitted immediately.
      On ``TextBlockEndEvent`` :meth:`TTSModelBase.synthesize` is called
      to drain remaining audio, and the data block is closed.

    Each ``DataBlockDeltaEvent.data`` carries an **incremental** base64 PCM
    chunk; the full audio is the concatenation of every delta's decoded
    bytes (the data block is keyed by ``block_id``).
    """

    def __init__(self, tts_model: TTSModelBase) -> None:
        """Initialize the TTS middleware.

        Args:
            tts_model (`TTSModelBase`):
                The TTS model used to synthesize speech for assistant text
                blocks produced during reasoning.
        """
        self.tts = tts_model

    async def on_reasoning(
        self,
        agent: "Agent",
        input_kwargs: dict,
        next_handler: Callable[..., AsyncGenerator],
    ) -> AsyncGenerator:
        # Per-text-block buffers and audio-block bookkeeping
        text_buffers: dict[str, str] = {}
        audio_block_ids: dict[str, str] = {}
        audio_media_types: dict[str, str] = {}

        async with self.tts:
            async for evt in next_handler(**input_kwargs):
                yield evt

                if isinstance(evt, TextBlockDeltaEvent):
                    text_buffers[evt.block_id] = (
                        text_buffers.get(evt.block_id, "") + evt.delta
                    )
                    if self.tts.supports_streaming_input and evt.delta:
                        tts_res = await self.tts.push(
                            Msg(
                                id=evt.block_id,
                                name=agent.name,
                                content=[TextBlock(text=evt.delta)],
                                role="assistant",
                            ),
                        )
                        async for audio_evt in self._emit_chunk(
                            agent=agent,
                            text_block_id=evt.block_id,
                            tts_res=tts_res,
                            audio_block_ids=audio_block_ids,
                            audio_media_types=audio_media_types,
                        ):
                            yield audio_evt

                elif isinstance(evt, TextBlockEndEvent):
                    text = text_buffers.pop(evt.block_id, "")

                    if self.tts.supports_streaming_input:
                        # Drain any remaining audio for this text block
                        res = await self.tts.synthesize()
                        async for audio_evt in self._emit_synth_result(
                            agent=agent,
                            text_block_id=evt.block_id,
                            res=res,
                            audio_block_ids=audio_block_ids,
                            audio_media_types=audio_media_types,
                        ):
                            yield audio_evt
                    elif text.strip():
                        res = await self.tts.synthesize(
                            Msg(
                                name=agent.name,
                                content=[TextBlock(text=text)],
                                role="assistant",
                            ),
                        )
                        async for audio_evt in self._emit_synth_result(
                            agent=agent,
                            text_block_id=evt.block_id,
                            res=res,
                            audio_block_ids=audio_block_ids,
                            audio_media_types=audio_media_types,
                        ):
                            yield audio_evt

                    # Close the audio block (if any was opened for this text)
                    audio_block_id = audio_block_ids.pop(evt.block_id, None)
                    audio_media_types.pop(evt.block_id, None)
                    if audio_block_id is not None:
                        yield DataBlockEndEvent(
                            reply_id=agent.state.reply_id,
                            block_id=audio_block_id,
                        )

    async def _emit_synth_result(
        self,
        agent: "Agent",
        text_block_id: str,
        res: TTSResponse | AsyncGenerator[TTSResponse, None],
        audio_block_ids: dict[str, str],
        audio_media_types: dict[str, str],
    ) -> AsyncGenerator:
        """Normalize ``synthesize()`` returns (single response or async
        generator) into a stream of ``DATA_BLOCK_*`` events."""
        if isinstance(res, AsyncGenerator):
            async for chunk in res:
                async for ae in self._emit_chunk(
                    agent=agent,
                    text_block_id=text_block_id,
                    tts_res=chunk,
                    audio_block_ids=audio_block_ids,
                    audio_media_types=audio_media_types,
                ):
                    yield ae
        else:
            async for ae in self._emit_chunk(
                agent=agent,
                text_block_id=text_block_id,
                tts_res=res,
                audio_block_ids=audio_block_ids,
                audio_media_types=audio_media_types,
            ):
                yield ae

    @staticmethod
    async def _emit_chunk(
        agent: "Agent",
        text_block_id: str,
        tts_res: TTSResponse,
        audio_block_ids: dict[str, str],
        audio_media_types: dict[str, str],
    ) -> AsyncGenerator:
        """Emit one TTSResponse chunk as ``DATA_BLOCK_START`` (if needed)
        followed by ``DATA_BLOCK_DELTA``."""
        if tts_res is None or tts_res.content is None:
            return
        media_type = tts_res.content.source.media_type
        data = tts_res.content.source.data
        if not data:
            return

        audio_block_id = audio_block_ids.get(text_block_id)
        if audio_block_id is None:
            audio_block_id = uuid.uuid4().hex
            audio_block_ids[text_block_id] = audio_block_id
            audio_media_types[text_block_id] = media_type
            yield DataBlockStartEvent(
                reply_id=agent.state.reply_id,
                block_id=audio_block_id,
                media_type=media_type,
            )

        yield DataBlockDeltaEvent(
            reply_id=agent.state.reply_id,
            block_id=audio_block_id,
            data=data,
            media_type=audio_media_types[text_block_id],
        )
