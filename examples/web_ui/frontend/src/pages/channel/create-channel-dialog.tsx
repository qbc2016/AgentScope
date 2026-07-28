import { Loader2, PlusCircle, X } from 'lucide-react';
import * as React from 'react';

import type { ChannelTypeSchema } from '@/api';
import { channelApi } from '@/api';
import { Button } from '@/components/ui/button';
import {
	Dialog,
	DialogContent,
	DialogDescription,
	DialogFooter,
	DialogHeader,
	DialogTitle,
} from '@/components/ui/dialog';
import { useAgents } from '@/hooks/useAgents';
import { useTranslation } from '@/i18n/useI18n';
import {
	ChannelForm,
	defaultChannelForm,
	isChannelFormValid,
	toCreateRequest,
} from '@/pages/channel/channel-form';

interface Props {
	open: boolean;
	onOpenChange: (open: boolean) => void;
	onCreated?: () => void;
}

export function CreateChannelDialog({ open, onOpenChange, onCreated }: Props) {
	const { t } = useTranslation();
	const { agents } = useAgents();
	const [form, setForm] = React.useState(defaultChannelForm);
	const [channelTypes, setChannelTypes] = React.useState<ChannelTypeSchema[]>([]);
	const [loading, setLoading] = React.useState(false);
	const [error, setError] = React.useState('');

	React.useEffect(() => {
		if (open) {
			setForm(defaultChannelForm(agents[0]?.id ?? ''));
			setError('');
			channelApi
				.listTypes()
				.then(setChannelTypes)
				.catch(() => {});
		}
	}, [open, agents]);

	const agentList = React.useMemo(
		() => agents.map((a) => ({ id: a.id, name: a.data.name })),
		[agents],
	);
	const valid = isChannelFormValid(form, 'create');

	const handleSubmit = async () => {
		setError('');
		if (!valid) return;
		setLoading(true);
		try {
			await channelApi.create(toCreateRequest(form));
			onCreated?.();
			onOpenChange(false);
		} catch (e) {
			setError(e instanceof Error ? e.message : String(e));
		} finally {
			setLoading(false);
		}
	};

	return (
		<Dialog open={open} onOpenChange={onOpenChange}>
			<DialogContent className="!w-[560px] !max-w-[560px]">
				<DialogHeader>
					<DialogTitle>{t('channel.create.title')}</DialogTitle>
					<DialogDescription>{t('channel.create.description')}</DialogDescription>
				</DialogHeader>

				<div className="no-scrollbar -mx-4 max-h-[75vh] overflow-y-auto px-4">
					<ChannelForm
						mode="create"
						value={form}
						onChange={setForm}
						agents={agentList}
						channelTypes={channelTypes}
					/>
					{error && <p className="mt-2 text-sm text-destructive">{error}</p>}
				</div>

				<DialogFooter>
					<Button variant="ghost" onClick={() => onOpenChange(false)} disabled={loading}>
						<X className="size-3.5" />
						{t('common.cancel')}
					</Button>
					<Button onClick={handleSubmit} disabled={loading || !valid}>
						{loading ? (
							<Loader2 className="size-3.5 animate-spin" />
						) : (
							<PlusCircle className="size-3.5" />
						)}
						{loading ? t('common.creating') : t('common.create')}
					</Button>
				</DialogFooter>
			</DialogContent>
		</Dialog>
	);
}
