import { getTextContent } from '@agentscope-ai/agentscope/message';
import { ArrowDown, ArrowUp, Check, GripVertical, Loader2, Pencil, Trash2, X } from 'lucide-react';
import { useState } from 'react';

import type { ChatQueueItem } from '@/api/chat';
import { Button } from '@/components/ui/button';
import { useTranslation } from '@/i18n/useI18n';

interface QueuedMessagesProps {
	items: ChatQueueItem[];
	onUpdate: (itemId: string, text: string) => Promise<void>;
	onDelete: (itemId: string) => Promise<void>;
	onMove: (itemId: string, direction: -1 | 1) => Promise<void>;
	onReorder: (itemIds: string[]) => Promise<void>;
}

const itemText = (item: ChatQueueItem): string => {
	const messages = Array.isArray(item.input) ? item.input : [item.input];
	return messages
		.map((message) => getTextContent(message))
		.filter(Boolean)
		.join('\n');
};

/** Editable FIFO shown immediately above the composer. */
export function QueuedMessages({
	items,
	onUpdate,
	onDelete,
	onMove,
	onReorder,
}: QueuedMessagesProps) {
	const { t } = useTranslation();
	const [editingId, setEditingId] = useState<string | null>(null);
	const [draft, setDraft] = useState('');
	const [busyId, setBusyId] = useState<string | null>(null);
	const [draggedId, setDraggedId] = useState<string | null>(null);
	const [dropTargetId, setDropTargetId] = useState<string | null>(null);

	if (items.length === 0) return null;

	const run = async (itemId: string, action: () => Promise<void>) => {
		setBusyId(itemId);
		try {
			await action();
		} catch {
			// The shared API client already displays the server detail.
			// Keep the item/draft visible so the user can retry.
		} finally {
			setBusyId(null);
		}
	};

	const dropBefore = (targetId: string) => {
		if (!draggedId || draggedId === targetId) return;
		const reordered = [...items];
		const sourceIndex = reordered.findIndex((item) => item.id === draggedId);
		const targetIndex = reordered.findIndex((item) => item.id === targetId);
		if (sourceIndex < 0 || targetIndex < 0) return;
		const [moved] = reordered.splice(sourceIndex, 1);
		reordered.splice(targetIndex, 0, moved);
		void run(draggedId, () => onReorder(reordered.map((item) => item.id)));
		setDraggedId(null);
		setDropTargetId(null);
	};

	return (
		<div className="mb-2 max-h-52 overflow-y-auto rounded-xl border bg-background/95 p-2 shadow-sm">
			<div className="px-1 pb-1 text-xs font-medium text-muted-foreground">
				{t('chatQueue.title', { count: items.length })}
			</div>
			<div className="space-y-1">
				{items.map((item, index) => {
					const editing = editingId === item.id;
					const busy = busyId === item.id;
					const editable = !Array.isArray(item.input);
					return (
						<div
							key={item.id}
							onDragOver={(event) => {
								event.preventDefault();
								event.dataTransfer.dropEffect = 'move';
								setDropTargetId(item.id);
							}}
							onDragLeave={() =>
								setDropTargetId((current) => (current === item.id ? null : current))
							}
							onDrop={(event) => {
								event.preventDefault();
								dropBefore(item.id);
							}}
							className={`flex items-center gap-1 rounded-lg border px-2 py-1.5 transition-colors ${
								dropTargetId === item.id
									? 'border-primary bg-primary/5'
									: 'border-transparent bg-muted/50'
							}`}
						>
							<div
								draggable={!editing && !busy}
								onDragStart={(event) => {
									setDraggedId(item.id);
									event.dataTransfer.effectAllowed = 'move';
									event.dataTransfer.setData('text/plain', item.id);
								}}
								onDragEnd={() => {
									setDraggedId(null);
									setDropTargetId(null);
								}}
								className="cursor-grab touch-none text-muted-foreground active:cursor-grabbing"
								title={t('chatQueue.drag')}
								aria-label={t('chatQueue.drag')}
							>
								<GripVertical className="size-4" />
							</div>
							<span className="w-5 shrink-0 text-center text-xs text-muted-foreground">
								{index + 1}
							</span>
							{editing ? (
								<textarea
									value={draft}
									onChange={(event) => setDraft(event.target.value)}
									rows={2}
									autoFocus
									className="min-w-0 flex-1 resize-none rounded-md border bg-background px-2 py-1 text-sm outline-none focus:ring-1 focus:ring-ring"
								/>
							) : (
								<p
									className="min-w-0 flex-1 truncate text-sm"
									title={itemText(item)}
								>
									{itemText(item) || t('chatQueue.attachmentOnly')}
								</p>
							)}

							{busy ? (
								<Loader2 className="size-4 shrink-0 animate-spin text-muted-foreground" />
							) : editing ? (
								<>
									<Button
										type="button"
										variant="ghost"
										size="icon-sm"
										disabled={!draft.trim()}
										title={t('chatQueue.save')}
										onClick={() =>
											void run(item.id, async () => {
												await onUpdate(item.id, draft.trim());
												setEditingId(null);
											})
										}
									>
										<Check className="size-3.5" />
									</Button>
									<Button
										type="button"
										variant="ghost"
										size="icon-sm"
										title={t('chatQueue.cancel')}
										onClick={() => setEditingId(null)}
									>
										<X className="size-3.5" />
									</Button>
								</>
							) : (
								<>
									<Button
										type="button"
										variant="ghost"
										size="icon-sm"
										disabled={index === 0}
										title={t('chatQueue.moveUp')}
										onClick={() => void run(item.id, () => onMove(item.id, -1))}
									>
										<ArrowUp className="size-3.5" />
									</Button>
									<Button
										type="button"
										variant="ghost"
										size="icon-sm"
										disabled={index === items.length - 1}
										title={t('chatQueue.moveDown')}
										onClick={() => void run(item.id, () => onMove(item.id, 1))}
									>
										<ArrowDown className="size-3.5" />
									</Button>
									<Button
										type="button"
										variant="ghost"
										size="icon-sm"
										disabled={!editable}
										title={t('chatQueue.edit')}
										onClick={() => {
											setDraft(itemText(item));
											setEditingId(item.id);
										}}
									>
										<Pencil className="size-3.5" />
									</Button>
									<Button
										type="button"
										variant="ghost"
										size="icon-sm"
										title={t('chatQueue.delete')}
										onClick={() => void run(item.id, () => onDelete(item.id))}
									>
										<Trash2 className="size-3.5 text-destructive" />
									</Button>
								</>
							)}
						</div>
					);
				})}
			</div>
		</div>
	);
}
