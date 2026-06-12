import type { AgentEvent } from '@agentscope-ai/agentscope/event';
import { useCallback, useEffect, useRef, useState } from 'react';

import { getBaseUrl, getUserId } from '@/api/client';

/**
 * Manages a WebSocket connection to the backend's realtime endpoint
 * (``WS /realtime/{sessionId}``).
 *
 * When ``enabled`` is ``true``, opens the WebSocket and routes incoming
 * ``AgentEvent`` frames through the provided ``processEvent`` callback
 * — the same callback used by ``useMessages`` for SSE events, so audio
 * output, text rendering, and message list updates all work without
 * changes.
 *
 * Exposes ``sendAudio(base64)`` for upstream audio from
 * ``useMicrophone``.
 *
 * @param agentId - The agent that owns the session.
 * @param sessionId - The session to connect.
 * @param enabled - Whether the WebSocket should be open.
 * @param processEvent - Event handler (shared with useMessages).
 * @returns Connection state + upstream sender.
 */
export function useRealtimeSession(
	agentId: string | null,
	sessionId: string | null,
	enabled: boolean,
	processEvent: ((event: AgentEvent) => void) | null,
) {
	const [connected, setConnected] = useState(false);
	const [error, setError] = useState<Error | null>(null);
	const wsRef = useRef<WebSocket | null>(null);

	const processEventRef = useRef(processEvent);
	useEffect(() => {
		processEventRef.current = processEvent;
	}, [processEvent]);

	useEffect(() => {
		if (!enabled || !agentId || !sessionId) {
			setConnected(false);
			return;
		}

		const baseUrl = getBaseUrl();
		const userId = getUserId();

		// Convert http(s):// to ws(s)://
		const wsBase = baseUrl.replace(/^http/, 'ws');
		const url = `${wsBase}/realtime/${sessionId}?agent_id=${encodeURIComponent(agentId)}&user_id=${encodeURIComponent(userId)}`;

		const ws = new WebSocket(url);
		wsRef.current = ws;

		ws.onopen = () => {
			setConnected(true);
			setError(null);
		};

		ws.onmessage = (event: MessageEvent) => {
			try {
				const data = JSON.parse(event.data as string) as AgentEvent;
				processEventRef.current?.(data);
			} catch {
				// skip malformed frames
			}
		};

		ws.onerror = () => {
			setError(new Error('WebSocket connection error'));
		};

		ws.onclose = () => {
			setConnected(false);
			wsRef.current = null;
		};

		return () => {
			ws.close();
			wsRef.current = null;
			setConnected(false);
		};
	}, [enabled, agentId, sessionId]);

	const sendAudio = useCallback((base64Pcm: string) => {
		const ws = wsRef.current;
		if (ws && ws.readyState === WebSocket.OPEN) {
			ws.send(JSON.stringify({ type: 'audio', data: base64Pcm }));
		}
	}, []);

	return { connected, sendAudio, error };
}
