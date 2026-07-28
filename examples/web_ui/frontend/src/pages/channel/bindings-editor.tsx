import { Plus, Trash2 } from 'lucide-react';

import type { ChannelBinding, SessionScope } from '@/api';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import {
	Select,
	SelectContent,
	SelectItem,
	SelectTrigger,
	SelectValue,
} from '@/components/ui/select';
import { useTranslation } from '@/i18n/useI18n';

const SESSION_SCOPES: SessionScope[] = ['per_chat', 'per_chat_user'];

interface Agent {
	id: string;
	name: string;
}

interface Props {
	value: ChannelBinding[];
	onChange: (bindings: ChannelBinding[]) => void;
	agents: Agent[];
}

/**
 * Editor for a channel's routing rules. The last row is always the
 * catch-all (match_value === '*'); extra rules are matched, in order,
 * before it. First match wins.
 */
export function BindingsEditor({ value, onChange, agents }: Props) {
	const { t } = useTranslation();

	const update = (i: number, patch: Partial<ChannelBinding>) => {
		onChange(value.map((b, idx) => (idx === i ? { ...b, ...patch } : b)));
	};

	const removeRule = (i: number) => {
		onChange(value.filter((_, idx) => idx !== i));
	};

	const addRule = () => {
		const catchAll = value[value.length - 1];
		const rule: ChannelBinding = {
			match_key: 'chat_id',
			match_value: '',
			agent_id: catchAll?.agent_id ?? agents[0]?.id ?? '',
			session_scope: 'per_chat',
		};
		// Insert before the catch-all so it stays last.
		onChange([...value.slice(0, -1), rule, ...value.slice(-1)]);
	};

	return (
		<div className="flex flex-col gap-2">
			{value.map((binding, i) => {
				const isCatchAll = i === value.length - 1;
				return (
					<div key={i} className="flex items-center gap-1.5">
						{isCatchAll ? (
							<span className="flex-1 text-xs text-muted-foreground">
								{t('channel.binding.catchAll')}
							</span>
						) : (
							<>
								<Input
									className="h-8 w-24 text-xs"
									value={binding.match_key}
									onChange={(e) => update(i, { match_key: e.target.value })}
									placeholder="chat_id"
								/>
								<Input
									className="h-8 flex-1 text-xs"
									value={binding.match_value}
									onChange={(e) => update(i, { match_value: e.target.value })}
									placeholder={t('channel.binding.matchValue')}
								/>
							</>
						)}

						<Select
							value={binding.agent_id}
							onValueChange={(v) => update(i, { agent_id: v })}
						>
							<SelectTrigger size="sm" className="w-28">
								<SelectValue placeholder={t('common.selectAgent')} />
							</SelectTrigger>
							<SelectContent>
								{agents.map((a) => (
									<SelectItem key={a.id} value={a.id}>
										{a.name}
									</SelectItem>
								))}
							</SelectContent>
						</Select>

						<Select
							value={binding.session_scope}
							onValueChange={(v) => update(i, { session_scope: v as SessionScope })}
						>
							<SelectTrigger size="sm" className="w-32">
								<SelectValue />
							</SelectTrigger>
							<SelectContent>
								{SESSION_SCOPES.map((s) => (
									<SelectItem key={s} value={s}>
										{t(`channel.sessionScope.${s}`)}
									</SelectItem>
								))}
							</SelectContent>
						</Select>

						{!isCatchAll && (
							<Button
								size="icon-sm"
								variant="ghost"
								className="text-destructive"
								onClick={() => removeRule(i)}
							>
								<Trash2 className="size-3.5" />
							</Button>
						)}
					</div>
				);
			})}

			<Button variant="ghost" size="sm" className="self-start" onClick={addRule}>
				<Plus className="size-3.5" />
				{t('channel.binding.addRule')}
			</Button>
		</div>
	);
}
