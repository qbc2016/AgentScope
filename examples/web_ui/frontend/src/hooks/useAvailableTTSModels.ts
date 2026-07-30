import { useState, useEffect, useCallback } from 'react';

import { credentialApi, ttsModelApi } from '@/api';
import type { CredentialView, TTSModelCard } from '@/api';

export interface CredentialWithTTSModels {
	credential: CredentialView;
	models: TTSModelCard[];
}

/**
 * Fetches endpoint-specific TTS models for each concrete credential.
 * Credentials/providers that expose no TTS models are omitted.
 */
export function useAvailableTTSModels() {
	const [groups, setGroups] = useState<Record<string, CredentialWithTTSModels[]>>({});
	const [loading, setLoading] = useState(false);
	const [error, setError] = useState<Error | null>(null);

	const refetch = useCallback(async () => {
		setLoading(true);
		setError(null);
		try {
			const { credentials } = await credentialApi.list();
			const result: Record<string, CredentialWithTTSModels[]> = {};
			const remoteCredentials: CredentialView[] = [];

			await Promise.all(
				credentials.map(async (credential) => {
					const type = credential.data.type as string | undefined;
					if (!type) return;
					if (!result[type]) result[type] = [];
					try {
						const isRemote = type === 'remote_tts_credential';
						const { models } = await ttsModelApi.list(type);
						if (models.length > 0) {
							result[type].push({ credential, models });
						}
						if (isRemote) remoteCredentials.push(credential);
					} catch {
						// Provider doesn't support TTS — skip silently
					}
				}),
			);

			// Remove provider groups with no TTS models
			for (const key of Object.keys(result)) {
				if (result[key].length === 0) delete result[key];
			}

			setGroups(result);

			// Remote discovery only enriches the already usable manual model
			// entry. Publish each result independently so a slow or unavailable
			// endpoint cannot hide the other credentials.
			await Promise.all(
				remoteCredentials.map(async (credential) => {
					try {
						const { models } = await ttsModelApi.list(
							'remote_tts_credential',
							credential.id,
						);
						setGroups((current) => ({
							...current,
							remote_tts_credential: (
								current.remote_tts_credential ?? []
							).map((item) =>
								item.credential.id === credential.id
									? { credential, models }
									: item,
							),
						}));
					} catch {
						// Manual model IDs remain available from the static card.
					}
				}),
			);
		} catch (e) {
			setError(e as Error);
		} finally {
			setLoading(false);
		}
	}, []);

	useEffect(() => {
		refetch();
	}, [refetch]);

	return { groups, loading, error, refetch };
}
