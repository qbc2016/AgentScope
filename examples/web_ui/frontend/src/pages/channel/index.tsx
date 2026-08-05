import { Circle, Pencil, Plus, Power, PowerOff, Trash2 } from 'lucide-react';
import * as React from 'react';

import type { ChannelRecord, ChannelTypeSchema } from '@/api';
import { channelApi } from '@/api';
import { DeleteDialog } from '@/components/dialog/DeleteDialog';
import { Avatar, AvatarFallback } from '@/components/ui/avatar';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Skeleton } from '@/components/ui/skeleton';
import { useAgents } from '@/hooks/useAgents';
import { useChannels } from '@/hooks/useChannels';
import { useTranslation } from '@/i18n/useI18n';
import { CreateChannelDialog } from '@/pages/channel/create-channel-dialog';
import { EditChannelDialog } from '@/pages/channel/edit-channel-dialog';
import { avatarTint } from '@/utils/common';

function ChannelCard({
	channel,
	typeName,
	agentName,
	onEnable,
	onDisable,
	onEdit,
	onDelete,
}: {
	channel: ChannelRecord;
	typeName: string;
	agentName: string;
	onEnable: () => void;
	onDisable: () => void;
	onEdit: () => void;
	onDelete: () => void;
}) {
	const { t } = useTranslation();

	return (
		<Card className="shadow-sm hover:shadow transition-shadow">
			<CardHeader className="pb-3">
				<div className="flex items-center gap-3">
					<Avatar className="size-9 rounded-lg">
						<AvatarFallback
							className="rounded-lg"
							style={avatarTint(channel.channel_type)}
						>
							{typeName.slice(0, 1).toUpperCase()}
						</AvatarFallback>
					</Avatar>
					<div className="min-w-0 flex-1">
						<CardTitle className="truncate text-sm font-semibold">{typeName}</CardTitle>
						<span className="font-mono text-[11px] text-muted-foreground">
							{channel.id.slice(0, 8)}
						</span>
					</div>
					<div className="flex items-center gap-1">
						{channel.enabled ? (
							<Button
								size="icon-sm"
								variant="ghost"
								onClick={onDisable}
								tooltip={t('channel.disable')}
							>
								<PowerOff className="size-3.5" />
							</Button>
						) : (
							<Button
								size="icon-sm"
								variant="ghost"
								onClick={onEnable}
								tooltip={t('channel.enable')}
							>
								<Power className="size-3.5" />
							</Button>
						)}
						<Button
							size="icon-sm"
							variant="ghost"
							onClick={onEdit}
							tooltip={t('common.edit')}
						>
							<Pencil className="size-3.5" />
						</Button>
						<Button
							size="icon-sm"
							variant="ghost"
							className="text-destructive"
							onClick={onDelete}
							tooltip={t('common.delete')}
						>
							<Trash2 className="size-3.5" />
						</Button>
					</div>
				</div>
			</CardHeader>
			<CardContent className="flex flex-col gap-2 text-sm">
				<div className="flex justify-between items-center">
					<span className="text-muted-foreground">{t('common.agent')}</span>
					<span className="truncate max-w-[120px]">{agentName}</span>
				</div>
				<div className="flex justify-between items-center">
					<span className="text-muted-foreground">{t('channel.routing')}</span>
					<span className="font-mono text-xs">
						{t('channel.rules', { count: channel.routing.bindings.length })}
					</span>
				</div>
				<div className="flex justify-between items-center">
					<span className="text-muted-foreground">{t('channel.status')}</span>
					<Badge variant={channel.enabled ? 'default' : 'secondary'} className="text-xs">
						<Circle
							className={`size-2 ${channel.enabled ? 'fill-green-400 text-green-400' : 'fill-gray-400 text-gray-400'}`}
						/>
						{channel.enabled ? t('channel.connected') : t('common.disabled')}
					</Badge>
				</div>
			</CardContent>
		</Card>
	);
}

function ChannelTypeCard({ type, onPick }: { type: ChannelTypeSchema; onPick: () => void }) {
	return (
		<button
			onClick={onPick}
			className="group flex items-center gap-3 rounded-xl border bg-card p-4 text-left transition hover:border-ring/40 hover:shadow-sm"
		>
			<Avatar className="size-10 rounded-lg">
				<AvatarFallback className="rounded-lg" style={avatarTint(type.channel_type)}>
					{type.display_name.slice(0, 1).toUpperCase()}
				</AvatarFallback>
			</Avatar>
			<div className="min-w-0 flex-1">
				<div className="truncate text-sm font-semibold">{type.display_name}</div>
				<div className="font-mono text-[11px] text-muted-foreground">
					{type.channel_type}
				</div>
			</div>
			<Plus className="size-4 text-muted-foreground opacity-0 transition group-hover:opacity-100" />
		</button>
	);
}

export function ChannelPage() {
	const { t } = useTranslation();
	const { channels, loading, refetch, enable, disable, remove } = useChannels();
	const { agents } = useAgents();
	const [types, setTypes] = React.useState<ChannelTypeSchema[]>([]);
	const [createType, setCreateType] = React.useState<string | null>(null);
	const [editTarget, setEditTarget] = React.useState<ChannelRecord | null>(null);
	const [deleteTarget, setDeleteTarget] = React.useState<ChannelRecord | null>(null);

	React.useEffect(() => {
		channelApi
			.listTypes()
			.then(setTypes)
			.catch(() => {});
	}, []);

	const typeName = (channelType: string) =>
		types.find((ct) => ct.channel_type === channelType)?.display_name ?? channelType;
	const agentName = (agentId: string) =>
		agents.find((a) => a.id === agentId)?.data.name ?? agentId.slice(0, 8);

	return (
		<div className="w-full h-full flex flex-col bg-sidebar overflow-hidden">
			<div className="flex items-center p-4 flex-shrink-0">
				<span className="text-2xl font-semibold">{t('channel.title')}</span>
			</div>

			<div className="flex-1 overflow-y-auto rounded-t-3xl bg-white p-6">
				{loading ? (
					<div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
						{Array.from({ length: 3 }).map((_, i) => (
							<Skeleton key={i} className="h-40 rounded-lg" />
						))}
					</div>
				) : (
					<>
						{channels.length > 0 && (
							<section className="mb-2">
								<div className="mb-4 flex items-center gap-2">
									<span className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
										{t('channel.sectionActive')}
									</span>
									<span className="rounded bg-muted px-2 py-0.5 font-mono text-[10px] text-muted-foreground">
										{channels.length}
									</span>
								</div>
								<div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
									{channels.map((ch) => (
										<ChannelCard
											key={ch.id}
											channel={ch}
											typeName={typeName(ch.channel_type)}
											agentName={agentName(
												ch.routing.bindings[ch.routing.bindings.length - 1]
													?.agent_id ?? '',
											)}
											onEnable={() => enable(ch.id)}
											onDisable={() => disable(ch.id)}
											onEdit={() => setEditTarget(ch)}
											onDelete={() => setDeleteTarget(ch)}
										/>
									))}
								</div>
							</section>
						)}

						<div className="my-8 flex items-center gap-4">
							<span className="flex items-center gap-2 text-xs font-semibold text-muted-foreground">
								<Plus className="size-3.5 text-primary" />
								{t('channel.sectionAdd')}
							</span>
							<div className="flex-1 border-t border-dashed" />
						</div>

						<div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
							{types.map((ct) => (
								<ChannelTypeCard
									key={ct.channel_type}
									type={ct}
									onPick={() => setCreateType(ct.channel_type)}
								/>
							))}
						</div>
					</>
				)}
			</div>

			<CreateChannelDialog
				open={createType !== null}
				initialType={createType ?? undefined}
				onOpenChange={(open) => !open && setCreateType(null)}
				onCreated={refetch}
			/>

			<EditChannelDialog
				channel={editTarget}
				open={!!editTarget}
				onOpenChange={(open) => !open && setEditTarget(null)}
				onUpdated={refetch}
			/>

			{deleteTarget && (
				<DeleteDialog
					open={!!deleteTarget}
					onOpenChange={(open) => !open && setDeleteTarget(null)}
					title={t('common.deleteTitle', {
						entity: t('channel.deleteEntity'),
						name: `${typeName(deleteTarget.channel_type)} · ${deleteTarget.id.slice(0, 8)}`,
					})}
					description={t('common.deleteDescription')}
					onConfirm={async () => {
						await remove(deleteTarget.id);
						setDeleteTarget(null);
					}}
				/>
			)}
		</div>
	);
}
