import { EventType } from '@agentscope-ai/agentscope/event';
import type {
	AgentEvent,
	CustomEvent,
	DataBlockStartEvent,
	DataBlockDeltaEvent,
	DataBlockEndEvent,
	ReplyStartEvent,
	UserConfirmResultEvent,
} from '@agentscope-ai/agentscope/event';
import { appendEvent, AssistantMsg, UserMsg } from '@agentscope-ai/agentscope/message';
import type { Msg, ContentBlock, TextBlock } from '@agentscope-ai/agentscope/message';
import type { ToolCallBlock } from '@agentscope-ai/agentscope/message';
import { useState, useCallback, useRef, useEffect, useMemo } from 'react';
import { toast } from 'sonner';

import { sessionApi } from '@/api';
import { chatApi } from '@/api';
import type { ChatQueueItem } from '@/api/chat';
import { useAudioManager } from '@/context/AudioContext';

/**
 * One pending subagent HITL request, projected from a team *member*
 * session onto its *leader* session so the leader UI can render and
 * resolve it. Mirrors the Python payload written by
 * ``SubagentHitlProjector`` and pushed/replayed as a ``CustomEvent``
 * (``name="subagent_require_user_confirm"``).
 */
export type SubagentHitlEntry = {
	worker_session_id: string;
	worker_agent_id: string;
	worker_agent_name: string;
	reply_id: string;
	event_type: 'require_user_confirm' | 'require_external_execution';
	/** The original ``RequireUserConfirmEvent`` payload (serialized). */
	event: { tool_calls?: ToolCallBlock[] } & Record<string, unknown>;
	created_at: string;
};

/**
 * Return true if ``msg`` is an assistant reply currently parked on a
 * pending tool_call (awaiting user confirmation or an external
 * execution result). Used both to detect the "in-flight reply on page
 * load" case and as the SDK-gap workaround that hides the confirm
 * card once the paired tool_result lands.
 */
const hasPendingToolCall = (msg: Msg | undefined): boolean => {
	if (!msg || msg.role !== 'assistant') return false;
	for (const block of msg.content) {
		if (block.type !== 'tool_call') continue;
		const state = (block as ToolCallBlock).state;
		if (state === 'asking' || state === 'submitted') return true;
	}
	return false;
};

const hitlKey = (e: { worker_session_id: string; reply_id: string }) =>
	`${e.worker_session_id}:${e.reply_id}`;

/**
 * Lifecycle phase of the reply currently owned by this session.
 *
 * - ``idle`` — no in-flight reply; the send button is enabled.
 * - ``streaming`` — a reply is in progress (either actively producing
 *   events or parked awaiting HITL). The send button is replaced by a
 *   Stop button. The parked-vs-generating distinction is not tracked
 *   here; HITL cards render themselves from message content when a
 *   ``RequireUserConfirmEvent`` block is present.
 * - ``interrupting`` — the user has requested a stop and we are
 *   waiting for the backend's terminating ``ReplyEndEvent``. Stop
 *   button is shown but disabled so users cannot spam it. Falls back
 *   to ``idle`` after a 10s safety timeout in case the terminating
 *   event never arrives (dropped SSE frame, backend bug, etc.).
 */
export type ReplyPhase = 'idle' | 'streaming' | 'interrupting';

/** Safety fallback: force phase back to idle if REPLY_END is not seen. */
const INTERRUPT_TIMEOUT_MS = 10_000;

/**
 * Manages messages for a single ``(agentId, sessionId)`` pair.
 *
 * Event delivery has two independent channels:
 *
 * - **History** — ``GET /sessions/{sid}/messages`` fetches persisted
 *   ``Msg`` objects (each a complete reply).
 * - **Live stream** — ``GET /sessions/{sid}/stream`` is a long-lived
 *   SSE connection that pushes ``AgentEvent`` deltas as they are
 *   produced by any chat run on this session (user-triggered,
 *   background retrigger, team member message, …).
 *
 * The hook opens the SSE connection immediately after fetching
 * history. User input and human-in-the-loop confirmations are sent
 * via ``POST /chat/`` (fire-and-forget); the resulting events arrive
 * through the already-open SSE connection.
 *
 * ``phase`` is driven by event content, not HTTP lifecycle: it moves
 * to ``streaming`` on ``ReplyStartEvent`` and back to ``idle`` on
 * ``ReplyEndEvent``. Calling ``interrupt()`` moves it to
 * ``interrupting`` until the terminating ``ReplyEndEvent`` arrives (or
 * a 10s safety timeout fires).
 *
 * @param agentId - The agent whose session to subscribe. ``null`` to
 *   skip.
 * @param sessionId - The session to subscribe. ``null`` to skip.
 * @returns Object with ``msgs``, ``loading``, ``phase``, ``error``,
 *   ``send``, ``onUserConfirm``, and ``abort``.
 */
export function useMessages(
	agentId: string | null,
	sessionId: string | null,
	options?: {
		/**
		 * Called when a ``CUSTOM`` event with ``name="team_updated"``
		 * arrives — the team membership has changed (TeamCreate /
		 * AgentCreate / TeamDelete ran). The typical response is to
		 * refetch the session list so the team sidebar updates.
		 */
		onTeamUpdated?: () => void;
		/**
		 * Called when a ``CUSTOM`` event with ``name="state_updated"``
		 * arrives — agent state (tasks / permission) changed during a
		 * tool call. The ``value`` payload contains the latest
		 * ``tasks_context`` and ``permission_context``.
		 */
		onStateUpdated?: (value: Record<string, unknown>) => void;
	},
) {
	const [msgs, setMsgs] = useState<Msg[]>([]);
	const [loading, setLoading] = useState(false);
	const [phase, setPhase] = useState<ReplyPhase>('idle');
	const [error, setError] = useState<Error | null>(null);
	const [queuedItems, setQueuedItems] = useState<ChatQueueItem[]>([]);
	const [optimisticConversationIds, setOptimisticConversationIds] = useState<Set<string>>(
		new Set(),
	);
	// Pending subagent HITL cards projected onto this (leader) session.
	const [subagentHitl, setSubagentHitl] = useState<SubagentHitlEntry[]>([]);

	const msgsRef = useRef<Msg[]>([]);
	const currentReplyRef = useRef<Msg | null>(null);
	const abortRef = useRef<AbortController | null>(null);
	const rafRef = useRef<number | null>(null);
	// Timer that reverts ``interrupting`` back to ``idle`` if the
	// terminating REPLY_END never arrives (dropped SSE frame, etc.).
	const interruptTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
	const queuedItemsRef = useRef<ChatQueueItem[]>([]);
	// IDs of idle-path messages already shown optimistically in the
	// conversation. Matching queue rows stay hidden until started so the
	// same input is not rendered in both places during the server roundtrip.
	const optimisticConversationIdsRef = useRef<Set<string>>(new Set());

	const updateQueuedItems = useCallback((items: ChatQueueItem[]) => {
		queuedItemsRef.current = items;
		setQueuedItems(items);
	}, []);

	const updateOptimisticConversationIds = useCallback((update: (ids: Set<string>) => void) => {
		const next = new Set(optimisticConversationIdsRef.current);
		update(next);
		optimisticConversationIdsRef.current = next;
		setOptimisticConversationIds(next);
	}, []);

	const clearInterruptTimer = useCallback(() => {
		if (interruptTimerRef.current !== null) {
			clearTimeout(interruptTimerRef.current);
			interruptTimerRef.current = null;
		}
	}, []);

	const audioManager = useAudioManager();

	const optionsRef = useRef(options);
	useEffect(() => {
		optionsRef.current = options;
	}, [options]);
	const scheduleUpdate = useCallback(() => {
		if (rafRef.current !== null) return;
		rafRef.current = requestAnimationFrame(() => {
			rafRef.current = null;
			setMsgs([...msgsRef.current]);
		});
	}, []);

	const queueMessages = useCallback((items: ChatQueueItem[]): Msg[] => {
		return items.flatMap((item) => (Array.isArray(item.input) ? item.input : [item.input]));
	}, []);

	/** Pending turns stay in the queue panel until the backend starts them. */
	const applyQueueSnapshot = useCallback(
		(items: ChatQueueItem[]) => updateQueuedItems(items),
		[updateQueuedItems],
	);

	/** Apply a single AgentEvent to the in-progress reply. */
	const processEvent = useCallback(
		(event: AgentEvent) => {
			// Custom events are service-layer notifications, not agent
			// reply content — route them to callbacks and skip appendEvent.
			if (event.type === EventType.CUSTOM) {
				const custom = event as CustomEvent;
				if (custom.name === 'team_updated') {
					optionsRef.current?.onTeamUpdated?.();
				} else if (custom.name === 'state_updated' && custom.value) {
					optionsRef.current?.onStateUpdated?.(custom.value as Record<string, unknown>);
				} else if (custom.name === 'subagent_require_user_confirm') {
					// A team member is asking for confirmation; show (or
					// refresh) its card on this leader view. Dedup by
					// (worker_session_id, reply_id).
					const e = custom.value as unknown as SubagentHitlEntry;
					setSubagentHitl((prev) => [
						...prev.filter((x) => hitlKey(x) !== hitlKey(e)),
						e,
					]);
				} else if (custom.name === 'subagent_user_confirm_result') {
					// The member resolved (or its run ended); clear the card.
					const v = custom.value as { worker_session_id: string; reply_id: string };
					setSubagentHitl((prev) => prev.filter((x) => hitlKey(x) !== hitlKey(v)));
				} else if (custom.name === 'chat_input_started') {
					const value = custom.value as {
						queue_item_id?: string;
						message_ids?: string[];
						queue_item?: ChatQueueItem;
					};
					const startedIds = new Set(value.message_ids ?? []);
					updateOptimisticConversationIds((ids) => {
						for (const messageId of startedIds) ids.delete(messageId);
					});
					const localItem = queuedItemsRef.current.find(
						(item) =>
							item.id === value.queue_item_id ||
							(item.id.startsWith('optimistic:') &&
								queueMessages([item]).some((msg) => startedIds.has(msg.id))),
					);
					const startedItem = value.queue_item ?? localItem;
					if (startedItem) {
						const existingIds = new Set(msgsRef.current.map((msg) => msg.id));
						msgsRef.current = [
							...msgsRef.current,
							...queueMessages([startedItem]).filter(
								(msg) => !existingIds.has(msg.id),
							),
						];
						scheduleUpdate();
					}
					updateQueuedItems(
						queuedItemsRef.current.filter(
							(item) =>
								item.id !== value.queue_item_id &&
								!(
									item.id.startsWith('optimistic:') &&
									queueMessages([item]).some((msg) => startedIds.has(msg.id))
								),
						),
					);
				} else if (custom.name === 'chat_queue_changed') {
					const value = custom.value as { items?: ChatQueueItem[] };
					applyQueueSnapshot(value.items ?? []);
				} else if (custom.name === 'chat_input_failed') {
					const value = custom.value as {
						message?: string;
						message_ids?: string[];
						queue_item_id?: string;
					};
					const failedIds = new Set(value.message_ids ?? []);
					updateQueuedItems(
						queuedItemsRef.current.filter(
							(item) =>
								item.id !== value.queue_item_id &&
								!queueMessages([item]).some((msg) => failedIds.has(msg.id)),
						),
					);
					updateOptimisticConversationIds((ids) => {
						for (const messageId of failedIds) {
							ids.delete(messageId);
						}
					});
					const failure = new Error(
						value.message ?? 'The queued message could not be processed.',
					);
					clearInterruptTimer();
					currentReplyRef.current = null;
					setPhase('idle');
					setError(failure);
					toast.error(failure.message);
				} else if (custom.name === 'chat_input_cancelled') {
					const value = custom.value as {
						message?: string;
						message_ids?: string[];
						queue_item_id?: string;
					};
					const cancelledIds = new Set(value.message_ids ?? []);
					updateQueuedItems(
						queuedItemsRef.current.filter(
							(item) =>
								item.id !== value.queue_item_id &&
								!queueMessages([item]).some((msg) => cancelledIds.has(msg.id)),
						),
					);
					updateOptimisticConversationIds((ids) => {
						for (const messageId of cancelledIds) {
							ids.delete(messageId);
						}
					});
					const message = value.message ?? 'Queued message processing was interrupted.';
					clearInterruptTimer();
					currentReplyRef.current = null;
					setPhase('idle');
					toast.info(message);
				}
				return;
			}
			if (event.type === EventType.REPLY_START) {
				audioManager?.stopAllPlayback();
				const e = event as ReplyStartEvent;
				const msg = AssistantMsg({ id: e.reply_id, name: e.name, content: [] });
				msgsRef.current = [...msgsRef.current, msg];
				currentReplyRef.current = msg;
				clearInterruptTimer();
				setPhase('streaming');
			} else if (event.type === EventType.REPLY_END) {
				if (currentReplyRef.current) {
					appendEvent(currentReplyRef.current, event);
				}
				clearInterruptTimer();
				setPhase('idle');
				currentReplyRef.current = null;
			} else if (currentReplyRef.current) {
				appendEvent(currentReplyRef.current, event);
			}

			// Route streaming audio DataBlocks to the audio manager. They still
			// flow through `appendEvent` above (which builds up `source.data`
			// in the Msg), but MessageBubble reads playback state from the
			// manager so it can show progress and autoplay on completion.
			if (audioManager) {
				if (event.type === EventType.DATA_BLOCK_START) {
					const e = event as DataBlockStartEvent;
					if (e.media_type.startsWith('audio/')) {
						audioManager.start(e.block_id, e.media_type);
					}
				} else if (event.type === EventType.DATA_BLOCK_DELTA) {
					const e = event as DataBlockDeltaEvent;
					if (e.media_type.startsWith('audio/')) {
						audioManager.append(e.block_id, e.data);
					}
				} else if (event.type === EventType.DATA_BLOCK_END) {
					const e = event as DataBlockEndEvent;
					// `end` is a no-op when the block isn't being tracked, so
					// we can call it unconditionally.
					audioManager.end(e.block_id);
				}
			}

			scheduleUpdate();
		},
		[
			scheduleUpdate,
			audioManager,
			clearInterruptTimer,
			updateQueuedItems,
			queueMessages,
			applyQueueSnapshot,
			updateOptimisticConversationIds,
		],
	);

	// ── Lifecycle: fetch history + open SSE stream ──────────────────
	useEffect(() => {
		msgsRef.current = [];
		currentReplyRef.current = null;
		setMsgs([]);
		setError(null);
		clearInterruptTimer();
		setPhase('idle');
		setSubagentHitl([]);
		updateQueuedItems([]);
		updateOptimisticConversationIds((ids) => ids.clear());
		audioManager?.disposeAll();

		if (!agentId || !sessionId) return;

		const controller = new AbortController();
		abortRef.current = controller;
		let cancelled = false;

		const refreshQueue = async () => {
			try {
				const queue = await chatApi.queue(agentId, sessionId, true);
				if (cancelled) return;
				const optimistic = queuedItemsRef.current.filter((item) =>
					item.id.startsWith('optimistic:'),
				);
				const serverMessageIds = new Set(
					queueMessages(queue.items).map((message) => message.id),
				);
				updateQueuedItems([
					...queue.items,
					...optimistic.filter(
						(item) =>
							!queueMessages([item]).some((message) =>
								serverMessageIds.has(message.id),
							),
					),
				]);
			} catch {
				// Queue snapshots are a periodic repair path for missed
				// live-only events. A transient failure is retried later.
			}
		};
		const queueRefreshTimer = window.setInterval(() => void refreshQueue(), 15_000);
		const refreshQueueOnFocus = () => void refreshQueue();
		window.addEventListener('focus', refreshQueueOnFocus);
		window.addEventListener('online', refreshQueueOnFocus);

		(async () => {
			// 1. Fetch persisted history
			setLoading(true);
			try {
				const [{ messages, is_running }, queue] = await Promise.all([
					sessionApi.messages(sessionId, agentId),
					chatApi
						.queue(agentId, sessionId, true)
						.catch(() => ({ items: [] as ChatQueueItem[] })),
				]);
				if (cancelled) return;
				msgsRef.current = messages;
				applyQueueSnapshot(queue.items);
				// If a reply is in flight (running on a worker) OR the
				// tail msg is parked on a pending tool_call (awaiting
				// user confirmation / external execution), initialise the
				// phase to ``streaming`` so the interrupt button is
				// available immediately — otherwise a fresh page load
				// while parked leaves the UI stuck on ``idle`` with no
				// way to abort.
				const tail = messages[messages.length - 1];
				if (is_running || hasPendingToolCall(tail)) {
					setPhase('streaming');
					if (hasPendingToolCall(tail)) {
						// Prime the ref so continuation events (no fresh
						// REPLY_START) apply to the right msg.
						currentReplyRef.current = tail ?? null;
					}
				}
				scheduleUpdate();
			} catch (e) {
				if (!cancelled) setError(e as Error);
				return;
			} finally {
				if (!cancelled) setLoading(false);
			}

			// 2. Open SSE long connection for live events
			try {
				for await (const event of sessionApi.streamEvents(
					sessionId,
					agentId,
					controller.signal,
				)) {
					if (cancelled) break;
					processEvent(event);
				}
			} catch (e) {
				if ((e as Error).name !== 'AbortError' && !cancelled) {
					setError(e as Error);
				}
			}
		})();

		return () => {
			cancelled = true;
			window.clearInterval(queueRefreshTimer);
			window.removeEventListener('focus', refreshQueueOnFocus);
			window.removeEventListener('online', refreshQueueOnFocus);
			controller.abort();
			abortRef.current = null;
			clearInterruptTimer();
		};
	}, [
		agentId,
		sessionId,
		scheduleUpdate,
		processEvent,
		audioManager,
		clearInterruptTimer,
		updateQueuedItems,
		applyQueueSnapshot,
		updateOptimisticConversationIds,
		queueMessages,
	]);

	/**
	 * Send a user message. The first idle input appears in the conversation
	 * immediately (with its matching queue row hidden); later inputs remain
	 * in the editable queue until ``chat_input_started`` arrives.
	 *
	 * @param content - The message content blocks.
	 */
	const send = useCallback(
		async (content: ContentBlock[]) => {
			if (!agentId || !sessionId) return;

			const userMsg = UserMsg({ name: 'user', content });
			const optimisticId = `optimistic:${crypto.randomUUID()}`;
			const showInConversationImmediately =
				phase === 'idle' && queuedItemsRef.current.length === 0;
			if (showInConversationImmediately) {
				updateOptimisticConversationIds((ids) => ids.add(userMsg.id));
				msgsRef.current = [...msgsRef.current, userMsg];
				scheduleUpdate();
			}
			updateQueuedItems([
				...queuedItemsRef.current,
				{
					id: optimisticId,
					created_at: userMsg.created_at,
					input: userMsg,
				},
			]);
			try {
				const response = await chatApi.trigger({
					agent_id: agentId,
					session_id: sessionId,
					input: userMsg,
				});
				const queueItemId = response.queue_item_id;
				if (queueItemId) {
					updateQueuedItems(
						queuedItemsRef.current.map((item) =>
							item.id === optimisticId ? { ...item, id: queueItemId } : item,
						),
					);
				}
			} catch (e) {
				// The server never accepted this input. Remove the
				// optimistic queue row and restore the caller's draft.
				updateQueuedItems(
					queuedItemsRef.current.filter((item) => item.id !== optimisticId),
				);
				if (showInConversationImmediately) {
					updateOptimisticConversationIds((ids) => ids.delete(userMsg.id));
					msgsRef.current = msgsRef.current.filter((msg) => msg.id !== userMsg.id);
					scheduleUpdate();
				}
				setError(e as Error);
				throw e;
			}
		},
		[
			agentId,
			sessionId,
			phase,
			scheduleUpdate,
			updateQueuedItems,
			updateOptimisticConversationIds,
		],
	);

	/** Replace the editable text of one still-pending queue item. */
	const updateQueued = useCallback(
		async (itemId: string, text: string) => {
			if (!agentId || !sessionId) return;
			const item = queuedItemsRef.current.find((candidate) => candidate.id === itemId);
			if (!item || Array.isArray(item.input)) {
				throw new Error('Only a single pending message can be edited.');
			}

			let replacedText = false;
			const content: ContentBlock[] = [];
			for (const block of item.input.content) {
				if (block.type !== 'text') {
					content.push(block);
				} else if (!replacedText) {
					replacedText = true;
					content.push({ ...block, text });
				}
			}
			if (!replacedText) {
				const now = new Date().toISOString();
				const textBlock: TextBlock = {
					id: crypto.randomUUID(),
					type: 'text',
					text,
					created_at: now,
					finished_at: now,
				};
				content.unshift(textBlock);
			}
			const updatedInput: Msg = { ...item.input, content };
			try {
				const response = await chatApi.updateQueued(itemId, {
					agent_id: agentId,
					session_id: sessionId,
					input: updatedInput,
				});
				applyQueueSnapshot(response.items);
			} catch (e) {
				setError(e as Error);
				throw e;
			}
		},
		[agentId, sessionId, applyQueueSnapshot],
	);

	/** Delete one still-pending queue item. */
	const deleteQueued = useCallback(
		async (itemId: string) => {
			if (!agentId || !sessionId) return;
			try {
				const response = await chatApi.deleteQueued(itemId, agentId, sessionId);
				applyQueueSnapshot(response.items);
			} catch (e) {
				setError(e as Error);
				throw e;
			}
		},
		[agentId, sessionId, applyQueueSnapshot],
	);

	/** Persist an exact pending-item permutation and reconcile conflicts. */
	const reorderQueued = useCallback(
		async (itemIds: string[]) => {
			if (!agentId || !sessionId) return;
			try {
				const response = await chatApi.reorderQueued(agentId, sessionId, itemIds);
				applyQueueSnapshot(response.items);
			} catch (e) {
				setError(e as Error);
				// A 409 means another client or the dispatcher changed the
				// queue. Refresh before allowing another reorder attempt.
				try {
					const response = await chatApi.queue(agentId, sessionId);
					applyQueueSnapshot(response.items);
				} catch {
					// Preserve the original mutation error. The API client
					// reports a refresh failure independently.
				}
				throw e;
			}
		},
		[agentId, sessionId, applyQueueSnapshot],
	);

	/** Move one pending item by a single position. */
	const moveQueued = useCallback(
		async (itemId: string, direction: -1 | 1) => {
			const current = queuedItemsRef.current;
			const index = current.findIndex((item) => item.id === itemId);
			const nextIndex = index + direction;
			if (index < 0 || nextIndex < 0 || nextIndex >= current.length) return;
			const reordered = [...current];
			[reordered[index], reordered[nextIndex]] = [reordered[nextIndex], reordered[index]];
			await reorderQueued(reordered.map((item) => item.id));
		},
		[reorderQueued],
	);

	/**
	 * Confirm or deny a tool call (human-in-the-loop). Fires a
	 * ``POST /chat/`` with a ``UserConfirmResultEvent``; events
	 * arrive via SSE.
	 *
	 * @param toolCall - The tool call block to confirm/deny.
	 * @param confirm - Whether the user confirmed.
	 * @param replyId - The reply id the tool call belongs to.
	 * @param rules - Optional permission rules to attach.
	 */
	const onUserConfirm = useCallback(
		async (
			toolCall: ToolCallBlock,
			confirm: boolean,
			replyId: string,
			rules?: ToolCallBlock['suggested_rules'],
		) => {
			if (!agentId || !sessionId) return;

			// Restore the ref so continuation events (no REPLY_START)
			// have a target.
			currentReplyRef.current = msgsRef.current.find((m) => m.id === replyId) ?? null;

			const event: UserConfirmResultEvent = {
				type: EventType.USER_CONFIRM_RESULT,
				id: crypto.randomUUID(),
				created_at: new Date().toISOString(),
				reply_id: replyId,
				confirm_results: [
					{ confirmed: confirm, tool_call: toolCall, rules: rules ?? null },
				],
			};

			try {
				await chatApi.trigger({
					agent_id: agentId,
					session_id: sessionId,
					input: event,
				});
			} catch (e) {
				setError(e as Error);
			}
		},
		[agentId, sessionId],
	);

	/** Abort the current SSE connection. */
	const abort = useCallback(() => {
		abortRef.current?.abort();
	}, []);

	/**
	 * Request interruption of the in-progress reply (running or parked
	 * on HITL). Optimistically moves ``phase`` to ``interrupting`` so
	 * the UI can disable the Stop button; the phase reverts to
	 * ``idle`` when the backend's terminating ``ReplyEndEvent``
	 * arrives via SSE (or after a 10s safety timeout, in case that
	 * event is lost).
	 *
	 * Backend contract:
	 * - 202: interrupt was accepted (cancel signal broadcast for a
	 *   running reply, or wakeup enqueued for a parked one). The
	 *   resulting ``ReplyEndEvent`` arrives through the SSE stream and
	 *   drives the phase transition.
	 * - Idle sessions are a silent no-op at the agent layer, so
	 *   spamming this callback is safe.
	 */
	const interrupt = useCallback(async () => {
		if (!agentId || !sessionId) return;
		// Only escalate to ``interrupting`` if a reply is actually in
		// flight; if we're already idle (SSE completed just before the
		// click) leave the phase alone.
		setPhase((prev) => (prev === 'streaming' ? 'interrupting' : prev));
		clearInterruptTimer();
		interruptTimerRef.current = setTimeout(() => {
			interruptTimerRef.current = null;
			setPhase((prev) => (prev === 'interrupting' ? 'idle' : prev));
		}, INTERRUPT_TIMEOUT_MS);
		try {
			await sessionApi.interrupt(sessionId, agentId);
		} catch (e) {
			clearInterruptTimer();
			setPhase((prev) => (prev === 'interrupting' ? 'idle' : prev));
			setError(e as Error);
		}
	}, [agentId, sessionId, clearInterruptTimer]);

	/**
	 * Confirm or deny a tool call that a *team member* is awaiting,
	 * from this leader view (design §3.6 — backend routing).
	 *
	 * The result is POSTed to the **leader** session (the
	 * ``(agentId, sessionId)`` this hook is bound to), NOT the worker.
	 * The backend resolves ``reply_id`` → worker session via the
	 * leader's pending hash and forwards the event to the worker's
	 * continuation. The client never addresses the worker directly —
	 * ``entry.worker_*`` ids are used only for local dedup / clearing.
	 *
	 * @param entry - The pending subagent HITL entry being resolved.
	 * @param toolCall - The tool call block to confirm/deny.
	 * @param confirm - Whether the user confirmed.
	 * @param rules - Optional permission rules to attach.
	 */
	const onSubagentConfirm = useCallback(
		async (
			entry: SubagentHitlEntry,
			toolCall: ToolCallBlock,
			confirm: boolean,
			rules?: ToolCallBlock['suggested_rules'],
		) => {
			if (!agentId || !sessionId) return;

			const event: UserConfirmResultEvent = {
				type: EventType.USER_CONFIRM_RESULT,
				id: crypto.randomUUID(),
				created_at: new Date().toISOString(),
				reply_id: entry.reply_id, // worker's reply_id; backend maps it
				confirm_results: [
					{ confirmed: confirm, tool_call: toolCall, rules: rules ?? null },
				],
			};

			// Optimistically clear; the backend's clear event re-confirms.
			setSubagentHitl((prev) => prev.filter((x) => hitlKey(x) !== hitlKey(entry)));

			try {
				// Post to the leader front door — backend routes to the
				// worker session (§3.6). Do NOT address the worker here.
				await chatApi.trigger({
					agent_id: agentId,
					session_id: sessionId,
					input: event,
				});
			} catch (e) {
				setError(e as Error);
			}
		},
		[agentId, sessionId],
	);

	const visibleQueuedItems = useMemo(
		() =>
			queuedItems.filter(
				(item) =>
					!queueMessages([item]).some((msg) => optimisticConversationIds.has(msg.id)),
			),
		[optimisticConversationIds, queueMessages, queuedItems],
	);

	return {
		msgs,
		loading,
		phase,
		queuedItems: visibleQueuedItems,
		queuedCount: visibleQueuedItems.length,
		updateQueued,
		deleteQueued,
		moveQueued,
		reorderQueued,
		error,
		send,
		onUserConfirm,
		onSubagentConfirm,
		subagentHitl,
		abort,
		interrupt,
	};
}
