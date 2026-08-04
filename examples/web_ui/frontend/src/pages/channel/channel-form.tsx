import * as React from 'react';

import type {
	ChannelBinding,
	ChannelRecord,
	ChannelTypeSchema,
	ChatModelConfig,
	CreateChannelRequest,
	PermissionMode,
	UpdateChannelRequest,
} from '@/api';
import { ModelParametersPopover } from '@/components/popover/ModelParametersPopover';
import { LlmSelect } from '@/components/select/LlmSelect';
import { PermissionModeSelect } from '@/components/select/PermissionModeSelect';
import { Field, FieldGroup, FieldLabel } from '@/components/ui/field';
import { Input } from '@/components/ui/input';
import {
	Select,
	SelectContent,
	SelectItem,
	SelectTrigger,
	SelectValue,
} from '@/components/ui/select';
import { Switch } from '@/components/ui/switch';
import { useAvailableModels } from '@/hooks/useAvailableModels';
import { useTranslation } from '@/i18n/useI18n';
import { BindingsEditor } from '@/pages/channel/bindings-editor';

export interface ChannelFormValue {
	channelType: string;
	credentials: Record<string, string>;
	platformConfig: Record<string, unknown>;
	bindings: ChannelBinding[];
	chatModelConfig: ChatModelConfig | null;
	fallbackChatModelConfig: ChatModelConfig | null;
	permissionMode: PermissionMode;
	showToolProcess: boolean;
	showThinking: boolean;
}

export function defaultChannelForm(agentId = ''): ChannelFormValue {
	return {
		channelType: 'feishu',
		credentials: {},
		platformConfig: {},
		bindings: [
			{
				match_key: 'chat_id',
				match_value: '*',
				agent_id: agentId,
				session_scope: 'per_chat',
			},
		],
		chatModelConfig: null,
		fallbackChatModelConfig: null,
		permissionMode: 'default' as PermissionMode,
		showToolProcess: false,
		showThinking: false,
	};
}

export function channelFormFromRecord(record: ChannelRecord): ChannelFormValue {
	return {
		channelType: record.channel_type,
		credentials: {},
		platformConfig: record.platform_config ?? {},
		bindings: record.routing.bindings,
		chatModelConfig: record.session.chat_model_config,
		fallbackChatModelConfig: record.session.fallback_chat_model_config ?? null,
		permissionMode: record.session.permission_mode,
		showToolProcess: record.presentation.show_tool_process,
		showThinking: record.presentation.show_thinking,
	};
}

function sessionAndPresentation(v: ChannelFormValue) {
	return {
		session: {
			chat_model_config: v.chatModelConfig as ChatModelConfig,
			fallback_chat_model_config: v.fallbackChatModelConfig,
			permission_mode: v.permissionMode,
		},
		presentation: {
			show_tool_process: v.showToolProcess,
			show_thinking: v.showThinking,
		},
	};
}

export function toCreateRequest(v: ChannelFormValue): CreateChannelRequest {
	return {
		channel_type: v.channelType,
		credentials: v.credentials,
		platform_config: v.platformConfig,
		routing: { bindings: v.bindings },
		enabled: true,
		...sessionAndPresentation(v),
	};
}

export function toUpdateRequest(v: ChannelFormValue): UpdateChannelRequest {
	return {
		routing: { bindings: v.bindings },
		...sessionAndPresentation(v),
	};
}

interface Agent {
	id: string;
	name: string;
}

interface Props {
	value: ChannelFormValue;
	onChange: (value: ChannelFormValue) => void;
	agents: Agent[];
	channelTypes: ChannelTypeSchema[];
	/** Create mode exposes type + credential fields; edit mode locks them. */
	mode: 'create' | 'edit';
}

export function ChannelForm({ value, onChange, agents, channelTypes, mode }: Props) {
	const { t } = useTranslation();
	const { groups } = useAvailableModels();

	const set = <K extends keyof ChannelFormValue>(key: K, v: ChannelFormValue[K]) =>
		onChange({ ...value, [key]: v });

	const typeSchema = React.useMemo(
		() => channelTypes.find((ct) => ct.channel_type === value.channelType),
		[channelTypes, value.channelType],
	);

	const credentialFields = React.useMemo(() => {
		const schema = typeSchema?.credentials_schema as
			| { properties?: Record<string, Record<string, unknown>>; required?: string[] }
			| undefined;
		if (!schema?.properties) return [];
		const required = schema.required ?? [];
		return Object.entries(schema.properties).map(([key, def]) => ({
			key,
			title: (def.title as string) || key,
			description: def.description as string | undefined,
			format: def.format as string | undefined,
			required: required.includes(key),
		}));
	}, [typeSchema]);

	const selectedModelCard = React.useMemo(() => {
		if (!value.chatModelConfig) return null;
		const items = groups[value.chatModelConfig.type];
		if (!items) return null;
		for (const { models } of items) {
			const card = models.find((m) => m.name === value.chatModelConfig!.model);
			if (card) return card;
		}
		return null;
	}, [groups, value.chatModelConfig?.type, value.chatModelConfig?.model]);

	return (
		<FieldGroup className="[&>[data-orientation=horizontal]>:last-child]:w-48">
			<Field orientation="horizontal">
				<FieldLabel>{t('channel.create.channelType')}</FieldLabel>
				{mode === 'create' ? (
					<Select
						value={value.channelType}
						onValueChange={(v) =>
							onChange({ ...value, channelType: v, credentials: {} })
						}
					>
						<SelectTrigger size="sm">
							<SelectValue />
						</SelectTrigger>
						<SelectContent>
							{channelTypes.map((ct) => (
								<SelectItem key={ct.channel_type} value={ct.channel_type}>
									{ct.display_name}
								</SelectItem>
							))}
						</SelectContent>
					</Select>
				) : (
					<span className="text-sm">{typeSchema?.display_name ?? value.channelType}</span>
				)}
			</Field>

			{mode === 'create' &&
				credentialFields.map((field) => (
					<Field key={field.key}>
						<FieldLabel>
							{field.title}
							{field.required && ' *'}
						</FieldLabel>
						<Input
							className="h-8 text-sm"
							type={field.format === 'password' ? 'password' : 'text'}
							value={value.credentials[field.key] || ''}
							onChange={(e) =>
								set('credentials', {
									...value.credentials,
									[field.key]: e.target.value,
								})
							}
							placeholder={field.description || field.title}
						/>
					</Field>
				))}

			<Field orientation="horizontal">
				<FieldLabel>{t('common.model')}</FieldLabel>
				<div className="flex items-center gap-1">
					<LlmSelect
						value={value.chatModelConfig}
						onChange={(v) => set('chatModelConfig', v)}
					/>
					<ModelParametersPopover
						selectedModel={value.chatModelConfig}
						modelCard={selectedModelCard}
						onChange={(parameters) =>
							value.chatModelConfig &&
							set('chatModelConfig', { ...value.chatModelConfig, parameters })
						}
						selectedFallbackModel={value.fallbackChatModelConfig}
						onFallbackChange={(cfg) => set('fallbackChatModelConfig', cfg)}
					/>
				</div>
			</Field>

			<Field orientation="horizontal">
				<FieldLabel>{t('channel.create.permissionMode')}</FieldLabel>
				<PermissionModeSelect
					className="w-full"
					value={value.permissionMode}
					onChange={(v) => set('permissionMode', v)}
				/>
			</Field>

			<Field>
				<FieldLabel>{t('channel.routing')}</FieldLabel>
				<span className="mb-1 text-xs text-muted-foreground">
					{t('channel.routingDesc')}
				</span>
				<BindingsEditor
					value={value.bindings}
					onChange={(b) => set('bindings', b)}
					agents={agents}
				/>
			</Field>

			<Field>
				<div className="flex flex-row items-center justify-between">
					<div className="flex flex-col gap-y-0.5">
						<FieldLabel>{t('channel.create.showToolMessages')}</FieldLabel>
						<span className="text-xs text-muted-foreground">
							{t('channel.create.showToolMessagesDesc')}
						</span>
					</div>
					<Switch
						checked={value.showToolProcess}
						onCheckedChange={(v) => set('showToolProcess', v)}
					/>
				</div>
			</Field>

			<Field>
				<div className="flex flex-row items-center justify-between">
					<div className="flex flex-col gap-y-0.5">
						<FieldLabel>{t('channel.create.showThinking')}</FieldLabel>
						<span className="text-xs text-muted-foreground">
							{t('channel.create.showThinkingDesc')}
						</span>
					</div>
					<Switch
						checked={value.showThinking}
						onCheckedChange={(v) => set('showThinking', v)}
					/>
				</div>
			</Field>
		</FieldGroup>
	);
}

export function isChannelFormValid(v: ChannelFormValue, mode: 'create' | 'edit'): boolean {
	if (!v.chatModelConfig) return false;
	if (v.bindings.length === 0) return false;
	if (v.bindings.some((b) => !b.agent_id)) return false;
	if (mode === 'create' && !v.channelType) return false;
	return true;
}
