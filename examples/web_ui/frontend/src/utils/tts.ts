/**
 * TTS utility functions shared across voice-profile and popover.
 */

const ENGINE_CREDENTIAL_TYPE: Record<string, string> = {
	cosyvoice: 'dashscope_credential',
	dashscope_tts: 'dashscope_credential',
	openai_tts: 'openai_credential',
	gemini_tts: 'gemini_credential',
	kokoro: 'local_tts_credential',
	chatterbox: 'local_tts_credential',
	luxtts: 'local_tts_credential',
	tada: 'local_tts_credential',
	remote_tts: 'remote_tts_credential',
	voicebox: 'voicebox_credential',
};

/** Return the credential provider type required by a TTS engine. */
export function credentialTypeForEngine(engine: string): string | undefined {
	return ENGINE_CREDENTIAL_TYPE[engine];
}

/**
 * Check whether a TTS model name belongs to a given engine.
 * This is the single source of truth for engine → model prefix matching.
 */
export function isModelForEngine(modelName: string, engine: string): boolean {
	switch (engine) {
		case 'dashscope_tts':
			return modelName.startsWith('qwen');
		case 'cosyvoice':
			return modelName.startsWith('cosyvoice');
		case 'openai_tts':
			return modelName.startsWith('tts-') || modelName.includes('tts');
		case 'gemini_tts':
			return modelName.includes('gemini');
		case 'kokoro':
			return modelName === 'kokoro' || modelName.startsWith('kokoro');
		case 'chatterbox':
			return modelName === 'chatterbox' || modelName.startsWith('chatterbox');
		case 'luxtts':
			return modelName === 'luxtts' || modelName.startsWith('luxtts');
		case 'tada':
			return modelName === 'tada' || modelName.startsWith('tada');
		case 'voicebox':
			return modelName === 'voicebox' || modelName.startsWith('voicebox');
		case 'remote_tts':
			return modelName === 'remote-tts';
		default:
			return modelName === engine || modelName.includes(engine);
	}
}
