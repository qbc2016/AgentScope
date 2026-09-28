import { useQuery } from '@tanstack/react-query';

import { credentialApi, realtimeModelApi } from '@/api';
import type { CredentialView, RealtimeModelCard } from '@/api';

export interface CredentialWithRealtimeModels {
	credential: CredentialView;
	models: RealtimeModelCard[];
}

async function fetchGroups(): Promise<Record<string, CredentialWithRealtimeModels[]>> {
	const { credentials } = await credentialApi.list();
	const result: Record<string, CredentialWithRealtimeModels[]> = {};

	await Promise.all(
		credentials.map(async (credential) => {
			const provider = credential.data.type as string | undefined;
			if (!provider) return;
			try {
				const { models } = await realtimeModelApi.list(provider);
				if (models.length === 0) return;
				if (!result[provider]) result[provider] = [];
				result[provider].push({
					credential,
					models: [...models].sort((a, b) =>
						b.name.localeCompare(a.name, undefined, { numeric: true }),
					),
				});
			} catch {
				// A credential without realtime support is not an error for this picker.
			}
		}),
	);

	return result;
}

export const AVAILABLE_REALTIME_MODELS_KEY = ['available-realtime-models'];

export function useAvailableRealtimeModels() {
	const query = useQuery({
		queryKey: AVAILABLE_REALTIME_MODELS_KEY,
		queryFn: fetchGroups,
	});
	return {
		groups: query.data ?? {},
		loading: query.isPending,
		error: query.error as Error | null,
		refetch: () => void query.refetch(),
	};
}
