import {
	AudioWaveform,
	Plus,
	Pencil,
	Trash2,
	Mic,
	AlertCircle,
	Upload,
	Loader2,
} from 'lucide-react';
import { useState, useEffect, useCallback, useMemo, useRef } from 'react';

import { voiceProfileApi, ttsModelApi } from '@/api';
import type { VoiceProfileRecord, VoiceProfileData, EngineInfo, TTSModelCard } from '@/api';
import { DeleteDialog } from '@/components/dialog/DeleteDialog';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardAction, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import {
	Dialog,
	DialogContent,
	DialogFooter,
	DialogHeader,
	DialogTitle,
} from '@/components/ui/dialog';
import { Empty, EmptyDescription, EmptyHeader, EmptyTitle } from '@/components/ui/empty';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import {
	Select,
	SelectContent,
	SelectItem,
	SelectTrigger,
	SelectValue,
} from '@/components/ui/select';
import { Textarea } from '@/components/ui/textarea';
import { useTranslation } from '@/i18n/useI18n';
import { credentialTypeForEngine, isModelForEngine } from '@/utils/tts';

const ENGINE_LABELS: Record<string, string> = {
	cosyvoice: 'CosyVoice',
	dashscope_tts: 'DashScope TTS',
	openai_tts: 'OpenAI TTS',
	gemini_tts: 'Gemini TTS',
	kokoro: 'Kokoro (Local)',
	chatterbox: 'Chatterbox (Local)',
	luxtts: 'LuxTTS (Local)',
	tada: 'TADA (Local)',
	voicebox: 'Voicebox (Local)',
};

/** Read a File as base64 string (without the data URL prefix). */
function fileToBase64(file: File): Promise<string> {
	return new Promise((resolve, reject) => {
		const reader = new FileReader();
		reader.onload = () => {
			const result = reader.result as string;
			const base64 = result.split(',')[1] || '';
			resolve(base64);
		};
		reader.onerror = () => reject(reader.error);
		reader.readAsDataURL(file);
	});
}

const MAX_AUDIO_SIZE_MB = 10;
const MAX_AUDIO_SIZE_BYTES = MAX_AUDIO_SIZE_MB * 1024 * 1024;

const LOCAL_ENGINES = new Set(['kokoro', 'chatterbox', 'luxtts', 'tada']);

const KOKORO_LANGUAGE_CODES = ['a', 'b', 'e', 'f', 'h', 'i', 'j', 'p', 'z'] as const;
type KokoroLanguageCode = (typeof KOKORO_LANGUAGE_CODES)[number];

function isKokoroLanguageCode(value: unknown): value is KokoroLanguageCode {
	return typeof value === 'string' && KOKORO_LANGUAGE_CODES.includes(value as KokoroLanguageCode);
}

function kokoroLanguageFromVoice(voice: string): KokoroLanguageCode | undefined {
	const prefix = voice.trim()[0];
	return isKokoroLanguageCode(prefix) ? prefix : undefined;
}

function VoiceProfileDialog({
	open,
	onOpenChange,
	profile,
	onSaved,
}: {
	open: boolean;
	onOpenChange: (open: boolean) => void;
	profile?: VoiceProfileRecord | null;
	onSaved: () => void;
}) {
	const { t } = useTranslation();
	const [name, setName] = useState('');
	const [engine, setEngine] = useState<string | undefined>(undefined);
	const [model, setModel] = useState<string | undefined>(undefined);
	const [modelOptions, setModelOptions] = useState<TTSModelCard[]>([]);
	const [modelsLoading, setModelsLoading] = useState(false);
	const [voice, setVoice] = useState('');
	const [kokoroLanguage, setKokoroLanguage] = useState<KokoroLanguageCode>('a');
	const [speed, setSpeed] = useState(1);
	const [saving, setSaving] = useState(false);
	const [availableEngines, setAvailableEngines] = useState<string[]>([]);
	const [engineDetails, setEngineDetails] = useState<EngineInfo[]>([]);
	const [enginesLoading, setEnginesLoading] = useState(false);
	const [cloning, setCloning] = useState(false);
	const [cloneError, setCloneError] = useState('');
	const [cloneUrl, setCloneUrl] = useState('');
	const [consentId, setConsentId] = useState('');
	const [refAudioBase64, setRefAudioBase64] = useState<string | null>(null);
	const [refAudioFilename, setRefAudioFilename] = useState('');
	const [referenceText, setReferenceText] = useState('');
	const [credentialId, setCredentialId] = useState<string | null>(null);
	const fileInputRef = useRef<HTMLInputElement>(null);
	const consentFileRef = useRef<HTMLInputElement>(null);
	const localRefInputRef = useRef<HTMLInputElement>(null);

	useEffect(() => {
		if (!open) return;
		setEnginesLoading(true);
		voiceProfileApi
			.availableEngines()
			.then((res) => {
				setAvailableEngines(res.engines);
				setEngineDetails(res.engine_details || []);
			})
			.finally(() => setEnginesLoading(false));
	}, [open]);

	useEffect(() => {
		if (profile) {
			setName(profile.data.name);
			setEngine(profile.data.engine || undefined);
			setModel(profile.data.model || undefined);
			setVoice(profile.data.voice || '');
			const meta = profile.data.metadata as Record<string, unknown> | null;
			const storedLanguage = meta?.lang_code;
			setKokoroLanguage(
				isKokoroLanguageCode(storedLanguage)
					? storedLanguage
					: kokoroLanguageFromVoice(profile.data.voice || '') || 'a',
			);
			setSpeed(typeof meta?.speed === 'number' ? meta.speed : 1);
			setRefAudioBase64((meta?.reference_audio_base64 as string) || null);
			setRefAudioFilename('');
			setReferenceText((meta?.reference_text as string) || '');
			setCredentialId(profile.data.credential_id || null);
		} else {
			setName('');
			setEngine(undefined);
			setModel(undefined);
			setVoice('');
			setKokoroLanguage('a');
			setSpeed(1);
			setRefAudioBase64(null);
			setRefAudioFilename('');
			setReferenceText('');
			setCredentialId(null);
		}
	}, [profile, open]);

	useEffect(() => {
		if (!engine) {
			setModelOptions([]);
			setModel(undefined);
			setCredentialId(null);
			return;
		}
		// Auto-select credential for this engine
		const engineInfo = engineDetails.find((d) => d.name === engine);
		const creds = engineInfo?.credentials || [];
		setCredentialId((prev) => {
			if (prev && creds.some((c) => c.id === prev)) return prev;
			return creds.length === 1 ? creds[0].id : null;
		});

		const credType = credentialTypeForEngine(engine);
		if (!credType) {
			setModelOptions([]);
			return;
		}
		setModelsLoading(true);
		ttsModelApi
			.list(credType)
			.then((res) => {
				const filtered = res.models.filter((m) => isModelForEngine(m.name, engine!));
				setModelOptions(filtered);
				setModel((prev) => {
					const inList = filtered.some((m) => m.name === prev);
					if (inList) return prev;
					const engineInfo = engineDetails.find((d) => d.name === engine);
					if (engineInfo?.voice_cloning) {
						const cloneModel = filtered.find((m) => m.voice_cloning);
						if (cloneModel) return cloneModel.name;
					}
					return filtered.length > 0 ? filtered[0].name : undefined;
				});
			})
			.catch(() => setModelOptions([]))
			.finally(() => setModelsLoading(false));
	}, [engine, engineDetails]);

	const canClone = engine
		? (engineDetails.find((d) => d.name === engine)?.voice_cloning ?? false)
		: false;

	const selectedModelSupportsClone =
		engine === 'openai_tts'
			? true
			: model
				? (modelOptions.find((m) => m.name === model)?.voice_cloning ?? false)
				: false;

	const cloneSupportsUpload = engine === 'dashscope_tts' || engine === 'openai_tts';

	const isLocal = engine ? LOCAL_ENGINES.has(engine) : false;

	const handleLocalRefUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
		const file = e.target.files?.[0];
		if (!file) return;
		if (file.size > MAX_AUDIO_SIZE_BYTES) {
			setCloneError(t('voiceProfile.fileTooLarge', { max: MAX_AUDIO_SIZE_MB }));
			return;
		}
		try {
			const base64 = await fileToBase64(file);
			setRefAudioBase64(base64);
			setRefAudioFilename(file.name);
			setCloneError('');
		} catch (err: unknown) {
			const msg = err instanceof Error ? err.message : String(err);
			setCloneError(msg);
		} finally {
			if (localRefInputRef.current) localRefInputRef.current.value = '';
		}
	};

	const selectedModelProperties = useMemo<Record<string, Record<string, unknown>>>(() => {
		if (!model) return {};
		const card = modelOptions.find((m) => m.name === model);
		if (!card) return {};
		const properties = (card.parameter_schema as Record<string, unknown>)?.properties as
			| Record<string, Record<string, unknown>>
			| undefined;
		return properties ?? {};
	}, [model, modelOptions]);

	const voiceSchema = selectedModelProperties?.voice;
	const speedSchema = selectedModelProperties?.speed;
	const canConfigureVoice = Boolean(voiceSchema) || modelOptions.length === 0;
	const speedMinimum = typeof speedSchema?.minimum === 'number' ? speedSchema.minimum : 0.5;
	const speedMaximum = typeof speedSchema?.maximum === 'number' ? speedSchema.maximum : 2;
	const displayedVoiceOptions = useMemo(() => {
		const voiceOptions = (voiceSchema?.enum as string[] | undefined) || [];
		return engine === 'kokoro'
			? voiceOptions.filter((option) => kokoroLanguageFromVoice(option) === kokoroLanguage)
			: voiceOptions;
	}, [engine, kokoroLanguage, voiceSchema]);

	useEffect(() => {
		if (engine !== 'kokoro' || displayedVoiceOptions.length === 0) return;
		if (!displayedVoiceOptions.includes(voice)) {
			setVoice(displayedVoiceOptions[0]);
		}
	}, [displayedVoiceOptions, engine, voice]);

	useEffect(() => {
		const profileMetadata = profile?.data.metadata as Record<string, unknown> | null;
		if (!speedSchema || typeof profileMetadata?.speed === 'number') return;
		const defaultSpeed = speedSchema.default;
		setSpeed(typeof defaultSpeed === 'number' ? defaultSpeed : 1);
	}, [profile, speedSchema]);

	const handleConsentUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
		const file = e.target.files?.[0];
		if (!file || !credentialId) return;
		if (file.size > MAX_AUDIO_SIZE_BYTES) {
			setCloneError(t('voiceProfile.fileTooLarge', { max: MAX_AUDIO_SIZE_MB }));
			return;
		}
		setCloning(true);
		setCloneError('');
		try {
			const base64 = await fileToBase64(file);
			const res = await voiceProfileApi.uploadOpenAIConsent({
				credential_id: credentialId,
				audio_base64: base64,
				audio_filename: file.name,
			});
			setConsentId(res.consent_id);
		} catch (err: unknown) {
			const msg = err instanceof Error ? err.message : String(err);
			setCloneError(msg);
		} finally {
			setCloning(false);
			if (consentFileRef.current) {
				consentFileRef.current.value = '';
			}
		}
	};

	const handleCloneUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
		const file = e.target.files?.[0];
		if (!file || !engine || !credentialId) return;
		if (file.size > MAX_AUDIO_SIZE_BYTES) {
			setCloneError(t('voiceProfile.fileTooLarge', { max: MAX_AUDIO_SIZE_MB }));
			return;
		}
		if (engine !== 'openai_tts' && !model) {
			setCloneError(t('voiceProfile.modelPlaceholder'));
			return;
		}
		if (engine === 'openai_tts' && !consentId.trim()) {
			setCloneError(t('voiceProfile.consentRequired'));
			return;
		}
		setCloning(true);
		setCloneError('');
		try {
			const base64 = await fileToBase64(file);
			const res = await voiceProfileApi.cloneVoice({
				engine,
				credential_id: credentialId,
				model,
				audio_base64: base64,
				audio_filename: file.name,
				...(engine === 'openai_tts' ? { consent: consentId.trim() } : {}),
			});
			setVoice(res.voice_id);
			setCredentialId(res.credential_id);
		} catch (err: unknown) {
			const msg = err instanceof Error ? err.message : String(err);
			setCloneError(msg);
		} finally {
			setCloning(false);
			if (fileInputRef.current) {
				fileInputRef.current.value = '';
			}
		}
	};

	const handleCloneFromUrl = async () => {
		if (!engine || !model || !credentialId) {
			setCloneError(t('voiceProfile.modelPlaceholder'));
			return;
		}
		if (!cloneUrl.trim()) {
			setCloneError(t('voiceProfile.cloneUrlPlaceholder'));
			return;
		}
		setCloning(true);
		setCloneError('');
		try {
			const res = await voiceProfileApi.cloneVoice({
				engine,
				credential_id: credentialId,
				model,
				audio_url: cloneUrl.trim(),
			});
			setVoice(res.voice_id);
			setCredentialId(res.credential_id);
			setCloneUrl('');
		} catch (err: unknown) {
			const msg = err instanceof Error ? err.message : String(err);
			setCloneError(msg);
		} finally {
			setCloning(false);
		}
	};

	const handleSave = async () => {
		if (!name.trim()) return;
		if (engine === 'tada' && !refAudioBase64) {
			setCloneError(t('voiceProfile.tadaReferenceAudioRequired'));
			return;
		}
		setSaving(true);
		try {
			const metadata: Record<string, unknown> = {};
			if (engine === 'kokoro') {
				metadata.lang_code = kokoroLanguage;
			}
			if (speedSchema) {
				metadata.speed = Math.min(speedMaximum, Math.max(speedMinimum, speed));
			}
			if (isLocal && canClone && refAudioBase64) {
				metadata.reference_audio_base64 = refAudioBase64;
				if (referenceText.trim()) {
					metadata.reference_text = referenceText.trim();
				}
			}
			const data: VoiceProfileData = {
				name: name.trim(),
				engine: engine || null,
				model: model || null,
				credential_id: credentialId,
				source: engine
					? engineDetails.find((d) => d.name === engine)?.source || null
					: null,
				voice: canConfigureVoice ? voice.trim() || null : null,
				metadata: Object.keys(metadata).length > 0 ? metadata : null,
			};
			if (profile) {
				await voiceProfileApi.update(profile.id, data);
			} else {
				await voiceProfileApi.create(data);
			}
			onSaved();
			onOpenChange(false);
		} finally {
			setSaving(false);
		}
	};

	return (
		<Dialog open={open} onOpenChange={onOpenChange}>
			<DialogContent className="sm:max-w-md">
				<DialogHeader>
					<DialogTitle>
						{profile ? t('voiceProfile.editTitle') : t('voiceProfile.createTitle')}
					</DialogTitle>
				</DialogHeader>
				<div className="flex flex-col gap-4 py-4">
					<div className="flex flex-col gap-2">
						<Label htmlFor="vp-name">{t('voiceProfile.name')}</Label>
						<Input
							id="vp-name"
							value={name}
							onChange={(e) => setName(e.target.value)}
							placeholder={t('voiceProfile.namePlaceholder')}
						/>
					</div>
					<div className="flex flex-col gap-2">
						<Label htmlFor="vp-engine">{t('voiceProfile.engine')}</Label>
						{!enginesLoading && availableEngines.length === 0 ? (
							<Alert variant="destructive">
								<AlertCircle className="size-4" />
								<AlertDescription>{t('voiceProfile.noEngines')}</AlertDescription>
							</Alert>
						) : (
							<Select
								value={engine}
								onValueChange={setEngine}
								disabled={enginesLoading}
							>
								<SelectTrigger id="vp-engine">
									<SelectValue
										placeholder={t('voiceProfile.enginePlaceholder')}
									/>
								</SelectTrigger>
								<SelectContent>
									{availableEngines.map((eng) => (
										<SelectItem key={eng} value={eng}>
											{ENGINE_LABELS[eng] ?? eng}
										</SelectItem>
									))}
								</SelectContent>
							</Select>
						)}
					</div>
					{engine &&
						(() => {
							const engineInfo = engineDetails.find((d) => d.name === engine);
							const creds = engineInfo?.credentials || [];
							if (creds.length === 0) return null;
							if (creds.length === 1) {
								return (
									<div className="flex flex-col gap-2">
										<Label htmlFor="vp-credential">
											{t('voiceProfile.credential')}
										</Label>
										<p className="text-sm text-muted-foreground px-3 py-2 border rounded-md bg-muted/30">
											{creds[0].label}
										</p>
									</div>
								);
							}
							return (
								<div className="flex flex-col gap-2">
									<Label htmlFor="vp-credential">
										{t('voiceProfile.credential')}
									</Label>
									<Select
										value={credentialId || undefined}
										onValueChange={(val) => setCredentialId(val)}
									>
										<SelectTrigger id="vp-credential">
											<SelectValue
												placeholder={t(
													'voiceProfile.credentialPlaceholder',
												)}
											/>
										</SelectTrigger>
										<SelectContent>
											{creds.map((c) => (
												<SelectItem key={c.id} value={c.id}>
													{c.label}
												</SelectItem>
											))}
										</SelectContent>
									</Select>
								</div>
							);
						})()}
					{engine && (
						<div className="flex flex-col gap-2">
							<Label htmlFor="vp-model">{t('voiceProfile.model')}</Label>
							{modelsLoading ? (
								<div className="flex items-center gap-2 text-sm text-muted-foreground">
									<Loader2 className="size-4 animate-spin" />
									{t('voiceProfile.loadingModels')}
								</div>
							) : modelOptions.length > 0 ? (
								<Select value={model} onValueChange={setModel}>
									<SelectTrigger id="vp-model">
										<SelectValue
											placeholder={t('voiceProfile.modelPlaceholder')}
										/>
									</SelectTrigger>
									<SelectContent>
										{modelOptions.map((m) => (
											<SelectItem key={m.name} value={m.name}>
												{m.label || m.name}
												{m.voice_cloning && ' (VC)'}
											</SelectItem>
										))}
									</SelectContent>
								</Select>
							) : (
								<Input
									id="vp-model"
									value={model || ''}
									onChange={(e) => setModel(e.target.value || undefined)}
									placeholder={t('voiceProfile.modelPlaceholder')}
								/>
							)}
						</div>
					)}
					{engine === 'kokoro' && (
						<div className="flex flex-col gap-2">
							<Label htmlFor="vp-language">{t('voiceProfile.language')}</Label>
							<Select
								value={kokoroLanguage}
								onValueChange={(value) =>
									setKokoroLanguage(value as KokoroLanguageCode)
								}
							>
								<SelectTrigger id="vp-language">
									<SelectValue
										placeholder={t('voiceProfile.languagePlaceholder')}
									/>
								</SelectTrigger>
								<SelectContent>
									{KOKORO_LANGUAGE_CODES.map((code) => (
										<SelectItem key={code} value={code}>
											{t(`voiceProfile.kokoroLanguages.${code}`)}
										</SelectItem>
									))}
								</SelectContent>
							</Select>
						</div>
					)}
					{engine && canConfigureVoice && (
						<div className="flex flex-col gap-2">
							<Label htmlFor="vp-voice">{t('voiceProfile.voice')}</Label>
							{displayedVoiceOptions.length > 0 &&
							(!canClone ||
								isLocal ||
								!voice ||
								displayedVoiceOptions.includes(voice)) ? (
								<Select value={voice} onValueChange={setVoice}>
									<SelectTrigger id="vp-voice">
										<SelectValue
											placeholder={t('voiceProfile.voicePlaceholderPreset')}
										/>
									</SelectTrigger>
									<SelectContent>
										{displayedVoiceOptions.map((v) => (
											<SelectItem key={v} value={v}>
												{v}
											</SelectItem>
										))}
									</SelectContent>
								</Select>
							) : (
								<Input
									id="vp-voice"
									value={voice}
									onChange={(e) => setVoice(e.target.value)}
									placeholder={
										canClone
											? t('voiceProfile.voicePlaceholderClone')
											: t('voiceProfile.voicePlaceholderPreset')
									}
								/>
							)}
						</div>
					)}
					{engine && speedSchema && (
						<div className="flex flex-col gap-2">
							<Label htmlFor="vp-speed">{t('voiceProfile.speed')}</Label>
							<Input
								id="vp-speed"
								type="number"
								min={speedMinimum}
								max={speedMaximum}
								step="0.1"
								value={speed}
								onChange={(event) => setSpeed(Number(event.target.value))}
							/>
							<p className="text-xs text-muted-foreground">
								{t('voiceProfile.speedHint')}
							</p>
						</div>
					)}
					{canClone && selectedModelSupportsClone && !isLocal && (
						<div className="flex flex-col gap-2">
							<Label>{t('voiceProfile.cloneVoice')}</Label>
							{engine === 'openai_tts' && (
								<div className="flex flex-col gap-2">
									<p className="text-xs text-muted-foreground">
										{t('voiceProfile.openaiCloneHint')}
									</p>
									<div className="flex items-center gap-2">
										<Input
											value={consentId}
											onChange={(e) => setConsentId(e.target.value)}
											placeholder={t('voiceProfile.consentPlaceholder')}
											className="flex-1"
										/>
										<input
											ref={consentFileRef}
											type="file"
											accept="audio/*"
											className="hidden"
											onChange={handleConsentUpload}
										/>
										<Button
											type="button"
											variant="outline"
											size="sm"
											disabled={cloning || !credentialId}
											onClick={() => consentFileRef.current?.click()}
										>
											{t('voiceProfile.uploadConsent')}
										</Button>
									</div>
								</div>
							)}
							{cloneSupportsUpload ? (
								<div className="flex items-center gap-2">
									<input
										ref={fileInputRef}
										type="file"
										accept="audio/*"
										className="hidden"
										onChange={handleCloneUpload}
									/>
									<Button
										type="button"
										variant="outline"
										size="sm"
										disabled={cloning || !credentialId}
										onClick={() => fileInputRef.current?.click()}
									>
										{cloning ? (
											<Loader2 className="size-4 mr-1.5 animate-spin" />
										) : (
											<Upload className="size-4 mr-1.5" />
										)}
										{cloning
											? t('voiceProfile.cloning')
											: t('voiceProfile.uploadAudio')}
									</Button>
								</div>
							) : (
								<div className="flex flex-col gap-2">
									<Textarea
										value={cloneUrl}
										onChange={(e) => setCloneUrl(e.target.value)}
										placeholder={t('voiceProfile.cloneUrlPlaceholder')}
										rows={3}
										className="text-sm break-all resize-none"
									/>
									<Button
										type="button"
										variant="outline"
										size="sm"
										disabled={cloning || !cloneUrl.trim() || !credentialId}
										onClick={handleCloneFromUrl}
									>
										{cloning ? (
											<Loader2 className="size-4 mr-1.5 animate-spin" />
										) : (
											<Upload className="size-4 mr-1.5" />
										)}
										{cloning
											? t('voiceProfile.cloning')
											: t('voiceProfile.clone')}
									</Button>
								</div>
							)}
							{cloneError && <p className="text-sm text-destructive">{cloneError}</p>}
						</div>
					)}
					{isLocal && canClone && selectedModelSupportsClone && (
						<div className="flex flex-col gap-2">
							<Label>
								{t('voiceProfile.referenceAudio')}
								{engine === 'tada' && (
									<span className="text-destructive ml-0.5">*</span>
								)}
							</Label>
							<p className="text-xs text-muted-foreground">
								{engine === 'tada'
									? t('voiceProfile.tadaReferenceAudioRequired')
									: t('voiceProfile.localCloneHint')}
							</p>
							<div className="flex items-center gap-2">
								<input
									ref={localRefInputRef}
									type="file"
									accept="audio/*"
									className="hidden"
									onChange={handleLocalRefUpload}
								/>
								<Button
									type="button"
									variant="outline"
									size="sm"
									onClick={() => localRefInputRef.current?.click()}
								>
									<Upload className="size-4 mr-1.5" />
									{refAudioBase64
										? t('voiceProfile.replaceAudio')
										: t('voiceProfile.uploadAudio')}
								</Button>
								{refAudioBase64 && (
									<span className="text-xs text-muted-foreground">
										{refAudioFilename || t('voiceProfile.audioUploaded')}
									</span>
								)}
							</div>
							{engine === 'tada' && (
								<div className="flex flex-col gap-2 mt-1">
									<Label htmlFor="vp-ref-text">
										{t('voiceProfile.referenceTextLabel')}
									</Label>
									<Input
										id="vp-ref-text"
										value={referenceText}
										onChange={(e) => setReferenceText(e.target.value)}
										placeholder={t('voiceProfile.referenceTextPlaceholder')}
									/>
								</div>
							)}
							{cloneError && <p className="text-sm text-destructive">{cloneError}</p>}
						</div>
					)}
				</div>
				<DialogFooter>
					<Button variant="outline" onClick={() => onOpenChange(false)}>
						{t('voiceProfile.cancel')}
					</Button>
					<Button
						onClick={handleSave}
						disabled={
							saving ||
							enginesLoading ||
							modelsLoading ||
							!name.trim() ||
							!engine ||
							!credentialId ||
							!model ||
							(canConfigureVoice && !voice.trim()) ||
							(engine === 'tada' && !refAudioBase64)
						}
					>
						{saving ? t('voiceProfile.saving') : t('voiceProfile.save')}
					</Button>
				</DialogFooter>
			</DialogContent>
		</Dialog>
	);
}

export function VoiceProfilePage() {
	const { t } = useTranslation();
	const [profiles, setProfiles] = useState<VoiceProfileRecord[]>([]);
	const [initialLoading, setInitialLoading] = useState(true);
	const [dialogOpen, setDialogOpen] = useState(false);
	const [editProfile, setEditProfile] = useState<VoiceProfileRecord | null>(null);
	const [deleteProfile, setDeleteProfile] = useState<VoiceProfileRecord | null>(null);
	const [engineDetails, setEngineDetails] = useState<EngineInfo[]>([]);

	const loadProfiles = useCallback(async () => {
		try {
			const res = await voiceProfileApi.list();
			setProfiles(res.profiles);
		} finally {
			setInitialLoading(false);
		}
	}, []);

	useEffect(() => {
		loadProfiles();
		voiceProfileApi.availableEngines().then((res) => {
			setEngineDetails(res.engine_details);
		});
	}, [loadProfiles]);

	const getCredentialLabel = (profile: VoiceProfileRecord): string | null => {
		if (!profile.data.credential_id || !profile.data.engine) return null;
		const info = engineDetails.find((d) => d.name === profile.data.engine);
		const cred = info?.credentials.find((c) => c.id === profile.data.credential_id);
		return cred?.label || null;
	};

	const handleDelete = async () => {
		if (!deleteProfile) return;
		await voiceProfileApi.delete(deleteProfile.id);
		setDeleteProfile(null);
		loadProfiles();
	};

	return (
		<div className="flex flex-col h-full p-6 gap-6 overflow-auto">
			<div className="flex items-center justify-between">
				<div className="flex flex-col gap-1">
					<div className="flex items-center gap-3">
						<AudioWaveform className="size-6 text-muted-foreground" />
						<h1 className="text-2xl font-semibold tracking-tight">
							{t('voiceProfile.title')}
						</h1>
					</div>
					<p className="text-sm text-muted-foreground ml-9">
						{t('voiceProfile.subtitle')}
					</p>
				</div>
				<Button
					onClick={() => {
						setEditProfile(null);
						setDialogOpen(true);
					}}
				>
					<Plus className="size-4 mr-1.5" />
					{t('voiceProfile.newProfile')}
				</Button>
			</div>

			{initialLoading ? (
				<div className="grid gap-4 grid-cols-1 md:grid-cols-2 lg:grid-cols-3">
					{[1, 2, 3].map((i) => (
						<Card key={i} className="animate-pulse">
							<CardHeader>
								<div className="h-5 w-32 bg-muted rounded" />
							</CardHeader>
							<CardContent>
								<div className="h-4 w-24 bg-muted rounded" />
							</CardContent>
						</Card>
					))}
				</div>
			) : profiles.length === 0 ? (
				<Empty>
					<EmptyHeader>
						<Mic className="size-10 text-muted-foreground" />
					</EmptyHeader>
					<EmptyTitle>{t('voiceProfile.noProfiles')}</EmptyTitle>
					<EmptyDescription>{t('voiceProfile.noProfilesDescription')}</EmptyDescription>
				</Empty>
			) : (
				<div className="grid gap-4 grid-cols-1 md:grid-cols-2 lg:grid-cols-3">
					{profiles.map((profile) => (
						<Card key={profile.id}>
							<CardHeader>
								<CardTitle className="flex items-center gap-2">
									<AudioWaveform className="size-4 text-muted-foreground" />
									{profile.data.name}
								</CardTitle>
								<CardAction>
									<Button
										size="icon-sm"
										variant="ghost"
										onClick={async () => {
											const fullProfile = await voiceProfileApi.get(
												profile.id,
											);
											setEditProfile(fullProfile);
											setDialogOpen(true);
										}}
									>
										<Pencil className="size-3.5" />
									</Button>
									<Button
										size="icon-sm"
										variant="ghost"
										onClick={() => setDeleteProfile(profile)}
									>
										<Trash2 className="size-3.5" />
									</Button>
								</CardAction>
							</CardHeader>
							<CardContent>
								<div className="flex flex-wrap gap-2">
									{(() => {
										const credLabel = getCredentialLabel(profile);
										if (!credLabel) return null;
										return <Badge variant="secondary">{credLabel}</Badge>;
									})()}
									{profile.data.engine && (
										<Badge variant="secondary">
											{ENGINE_LABELS[profile.data.engine] ??
												profile.data.engine}
										</Badge>
									)}
									{profile.data.model && (
										<Badge variant="secondary">{profile.data.model}</Badge>
									)}
									{profile.data.voice && (
										<Badge variant="secondary">{profile.data.voice}</Badge>
									)}
								</div>
							</CardContent>
						</Card>
					))}
				</div>
			)}

			<VoiceProfileDialog
				open={dialogOpen}
				onOpenChange={setDialogOpen}
				profile={editProfile}
				onSaved={loadProfiles}
			/>

			<DeleteDialog
				open={!!deleteProfile}
				onOpenChange={(open) => {
					if (!open) setDeleteProfile(null);
				}}
				onConfirm={handleDelete}
				title={t('voiceProfile.deleteTitle')}
				description={t('voiceProfile.deleteDescription', {
					name: deleteProfile?.data.name,
				})}
			/>
		</div>
	);
}
