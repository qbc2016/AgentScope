import { SlidersHorizontal } from 'lucide-react';
import { useCallback, useEffect, useState } from 'react';

import type { ChatModelConfig, ModelCard } from '@/api';
import { Button } from '@/components/ui/button';
import {
	DropdownMenu,
	DropdownMenuCheckboxItem,
	DropdownMenuContent,
	DropdownMenuLabel,
	DropdownMenuSeparator,
	DropdownMenuSub,
	DropdownMenuSubContent,
	DropdownMenuSubTrigger,
	DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu.tsx';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
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

interface ParameterProperty {
	type?: string;
	title?: string;
	description?: string;
	default?: unknown;
	minimum?: number;
	maximum?: number;
	exclusiveMinimum?: number;
	exclusiveMaximum?: number;
	enum?: unknown[];
	properties?: Record<string, ParameterProperty>;
	required?: string[];
	anyOf?: ParameterProperty[];
	$ref?: string;
}

const SCALAR_TYPES = new Set(['boolean', 'number', 'integer', 'string']);

interface ParameterSchema {
	title?: string;
	description?: string;
	type?: string;
	properties?: Record<string, ParameterProperty>;
	required?: string[];
}

interface Props {
	/** Currently selected primary model — used to read the parameter schema. */
	selectedModel: ChatModelConfig | null;
	/** Model card describing the primary model's parameter schema. */
	modelCard: ModelCard | null;
	/** Called when the user edits the primary model's parameters. */
	onChange: (parameters: Record<string, unknown>) => void;
	/** Currently selected fallback model. `null` means no fallback configured. */
	selectedFallbackModel: ChatModelConfig | null;
	/** Called when the user picks a fallback model or clears the selection. */
	onFallbackChange: (config: ChatModelConfig | null) => void;
}

/** Resolve a property's effective scalar type, ignoring ``null`` in ``anyOf``. */
function getScalarType(prop: ParameterProperty): string | null {
	const direct = prop.type;
	if (direct && SCALAR_TYPES.has(direct)) return direct;
	for (const variant of prop.anyOf ?? []) {
		if (variant.type && variant.type !== 'null' && SCALAR_TYPES.has(variant.type)) {
			return variant.type;
		}
	}
	return null;
}

/** If ``prop`` (directly or via ``anyOf``) describes an object with nested
 *  ``properties``, return that shape; otherwise null. */
function getObjectShape(
	prop: ParameterProperty,
): { properties: Record<string, ParameterProperty>; required: string[] } | null {
	if (prop.type === 'object' && prop.properties) {
		return { properties: prop.properties, required: prop.required ?? [] };
	}
	for (const variant of prop.anyOf ?? []) {
		if (variant.type === 'object' && variant.properties) {
			return { properties: variant.properties, required: variant.required ?? [] };
		}
	}
	return null;
}

interface ScalarFieldProps {
	id: string;
	label: string;
	required: boolean;
	prop: ParameterProperty;
	value: unknown;
	onChange: (next: unknown) => void;
}

/** Renders one scalar (boolean / number / enum / string) parameter input. */
function ScalarField({ id, label, required, prop, value, onChange }: ScalarFieldProps) {
	const effectiveType = getScalarType(prop) ?? 'string';
	const isBoolean = effectiveType === 'boolean';
	const isNumber = effectiveType === 'number' || effectiveType === 'integer';
	const enumValues = Array.isArray(prop.enum) ? (prop.enum as unknown[]) : null;

	if (isBoolean) {
		return (
			<>
				<Label htmlFor={id} className="whitespace-nowrap">
					{label}
				</Label>
				<Switch
					id={id}
					checked={value !== undefined ? !!value : !!prop.default}
					onCheckedChange={(checked) => onChange(!!checked)}
				/>
			</>
		);
	}

	if (enumValues) {
		const current = value !== undefined ? String(value) : (prop.default as string) ?? '';
		return (
			<>
				<Label htmlFor={id} className="whitespace-nowrap">
					{label}
					{required && <span className="text-destructive ml-0.5">*</span>}
				</Label>
				<Select value={current} onValueChange={(v) => onChange(v)}>
					<SelectTrigger id={id} size="sm" className="w-full">
						<SelectValue placeholder={String(prop.default ?? '')} />
					</SelectTrigger>
					<SelectContent>
						{enumValues.map((opt) => (
							<SelectItem key={String(opt)} value={String(opt)}>
								{String(opt)}
							</SelectItem>
						))}
					</SelectContent>
				</Select>
			</>
		);
	}

	return (
		<>
			<Label htmlFor={id} className="whitespace-nowrap">
				{label}
				{required && <span className="text-destructive ml-0.5">*</span>}
			</Label>
			<Input
				id={id}
				type={isNumber ? 'number' : 'text'}
				value={value !== undefined ? String(value) : ''}
				placeholder={prop.default !== undefined ? String(prop.default) : undefined}
				min={prop.minimum}
				max={prop.maximum}
				step={isNumber && effectiveType === 'number' ? 'any' : undefined}
				onChange={(e) => {
					const raw = e.target.value;
					if (isNumber) {
						onChange(raw === '' ? undefined : Number(raw));
					} else {
						onChange(raw);
					}
				}}
				onBlur={(e) => {
					if (!isNumber || e.target.value === '') return;
					let num = Number(e.target.value);
					if (prop.minimum !== undefined && num < prop.minimum) num = prop.minimum;
					if (prop.maximum !== undefined && num > prop.maximum) num = prop.maximum;
					if (prop.exclusiveMinimum !== undefined && num <= prop.exclusiveMinimum)
						num =
							prop.exclusiveMinimum +
							(effectiveType === 'integer' ? 1 : Number.EPSILON);
					if (prop.exclusiveMaximum !== undefined && num >= prop.exclusiveMaximum)
						num =
							prop.exclusiveMaximum -
							(effectiveType === 'integer' ? 1 : Number.EPSILON);
					if (num !== Number(e.target.value)) onChange(num);
				}}
			/>
		</>
	);
}

/**
 * A unified settings dropdown for the active chat model. Exposes two
 * sub-menus:
 *   - "Fallback model": pick a backup model invoked when the primary fails.
 *   - "Parameters": edit the primary model's inference parameters inline.
 *
 * The trigger is disabled until a primary model is selected, since both
 * sub-menus are meaningless without one.
 */
export function ModelParametersPopover({
	selectedModel,
	modelCard,
	onChange,
	selectedFallbackModel,
	onFallbackChange,
}: Props) {
	const [values, setValues] = useState<Record<string, unknown>>({});
	const { t } = useTranslation();
	const { groups } = useAvailableModels();

	const schema = modelCard?.parameter_schema as ParameterSchema | undefined;
	const properties = schema?.properties ?? {};
	const required = schema?.required ?? [];

	// Classify each property as scalar | nested-object | unsupported. Anything
	// unsupported (bare $ref with no inlined schema, arrays, etc.) is dropped
	// so user keystrokes can't round-trip as an invalid string value.
	type Entry =
		| { kind: 'scalar'; key: string; prop: ParameterProperty }
		| {
				kind: 'object';
				key: string;
				prop: ParameterProperty;
				shape: { properties: Record<string, ParameterProperty>; required: string[] };
		  };
	const entries: Entry[] = Object.entries(properties).flatMap(([key, prop]) => {
		const shape = getObjectShape(prop);
		if (shape) return [{ kind: 'object', key, prop, shape } as Entry];
		if (getScalarType(prop)) return [{ kind: 'scalar', key, prop } as Entry];
		return [];
	});

	useEffect(() => {
		setValues(selectedModel?.parameters ?? {});
	}, [selectedModel?.model]);

	const commit = useCallback(
		(next: Record<string, unknown>) => {
			setValues(next);
			onChange(next);
		},
		[onChange],
	);

	const setScalar = useCallback(
		(key: string, value: unknown) => {
			const next = { ...values };
			if (value === '' || value === undefined) delete next[key];
			else next[key] = value;
			commit(next);
		},
		[values, commit],
	);

	// Updates a single field inside a nested object value (e.g. ``audio.voice``).
	// Drops the parent key entirely when no child fields remain set.
	const setNested = useCallback(
		(
			parentKey: string,
			childKey: string,
			value: unknown,
			parentDefaults: Record<string, unknown>,
		) => {
			const parent: Record<string, unknown> = {
				...parentDefaults,
				...((values[parentKey] as Record<string, unknown> | undefined) ?? {}),
			};
			if (value === '' || value === undefined) delete parent[childKey];
			else parent[childKey] = value;

			const next = { ...values };
			if (Object.keys(parent).length === 0) delete next[parentKey];
			else next[parentKey] = parent;
			commit(next);
		},
		[values, commit],
	);

	const handleSelectFallback = (type: string, credentialId: string, model: string) => {
		onFallbackChange({
			type,
			credential_id: credentialId,
			model,
			parameters: {},
		});
	};

	const disabled = !selectedModel;
	const hasFallbackOptions = Object.keys(groups).length > 0;

	return (
		<DropdownMenu>
			<DropdownMenuTrigger asChild>
				<Button variant="ghost" size="icon-sm" disabled={disabled}>
					<SlidersHorizontal />
				</Button>
			</DropdownMenuTrigger>
			<DropdownMenuContent align="start" className="min-w-40">
				{/* ----- Fallback model selection ----- */}
				{/* The sub-trigger reflects the current fallback selection so the
				    user can see the active model without drilling in. The label
				    is truncated with an ellipsis to keep the trigger on one line
				    when the model name is long. */}
				<DropdownMenuSub>
					<DropdownMenuSubTrigger>
						<span className="truncate">
							{selectedFallbackModel
								? t('model-parameters.fallbackLabelWithModel', {
										model: selectedFallbackModel.model,
									})
								: t('model-parameters.fallbackLabel')}
						</span>
					</DropdownMenuSubTrigger>
					<DropdownMenuSubContent className="max-h-72 overflow-y-auto">
						{!hasFallbackOptions ? (
							<div className="px-2 py-3 text-center text-sm text-muted-foreground">
								<p>{t('llm-select.empty.title')}</p>
							</div>
						) : (
							Object.entries(groups).map(([type, items], idx) => (
								<div key={type}>
									{idx > 0 && <DropdownMenuSeparator />}
									<DropdownMenuLabel>
										{type.replace(/_credential$/, '')}
									</DropdownMenuLabel>
									{items.flatMap(({ credential, models }) =>
										models.map((m) => {
											const isSelected =
												selectedFallbackModel?.credential_id ===
													credential.id &&
												selectedFallbackModel?.model === m.name;
											// Checkbox-style indicator: a check appears next
											// to the active fallback. Toggling off the active
											// item clears the selection (mirrors the explicit
											// "No fallback" entry below).
											return (
												<DropdownMenuCheckboxItem
													key={`${credential.id}-${m.name}`}
													checked={isSelected}
													onCheckedChange={(checked) => {
														if (checked) {
															handleSelectFallback(
																type,
																credential.id,
																m.name,
															);
														} else {
															onFallbackChange(null);
														}
													}}
												>
													{m.label}
												</DropdownMenuCheckboxItem>
											);
										}),
									)}
								</div>
							))
						)}
						<DropdownMenuSeparator />
						{/* "No fallback" is checked exactly when no fallback is set,
						    making the list behave like a single-select group. Clicking
						    it while already checked is a no-op. */}
						<DropdownMenuCheckboxItem
							checked={!selectedFallbackModel}
							onCheckedChange={(checked) => {
								if (checked) onFallbackChange(null);
							}}
						>
							{t('llm-select.noFallback')}
						</DropdownMenuCheckboxItem>
					</DropdownMenuSubContent>
				</DropdownMenuSub>

				{/* ----- Primary model parameters ----- */}
				<DropdownMenuSub>
					<DropdownMenuSubTrigger>
						{t('model-parameters.parametersLabel')}
					</DropdownMenuSubTrigger>
					<DropdownMenuSubContent className="w-80 max-h-96 overflow-y-auto p-3">
						<div className="mb-3">
							<p className="text-sm font-medium">{t('model-parameters.title')}</p>
							<p className="text-muted-foreground text-xs">
								{t('model-parameters.description')}
							</p>
						</div>
						{entries.length === 0 ? (
							<p className="text-muted-foreground text-xs">
								{t('model-parameters.empty')}
							</p>
						) : (
							<div
								className="flex flex-col gap-y-3"
								// Keep clicks/keys inside the form from
								// closing the dropdown menu.
								onPointerDown={(e) => e.stopPropagation()}
								onKeyDown={(e) => e.stopPropagation()}
							>
								{entries.map((entry) => {
									if (entry.kind === 'scalar') {
										const { key, prop } = entry;
										return (
											<div
												key={key}
												className="grid grid-cols-[auto_1fr] items-center gap-x-3"
											>
												<ScalarField
													id={`param-${key}`}
													label={prop.title ?? key}
													required={required.includes(key)}
													prop={prop}
													value={values[key]}
													onChange={(v) => setScalar(key, v)}
												/>
											</div>
										);
									}

									const { key, prop, shape } = entry;
									const parentValue =
										(values[key] as Record<string, unknown> | undefined) ?? {};
									// Use any ``default: {...}`` declared on the parent as the
									// starting point for the merged child dict, so editing one
									// child doesn't blow away unspecified sibling defaults.
									const parentDefaults =
										(prop.default as Record<string, unknown> | undefined) ?? {};
									return (
										<div key={key} className="flex flex-col gap-y-2">
											<div className="text-xs font-medium text-muted-foreground">
												{prop.title ?? key}
											</div>
											<div className="grid grid-cols-[auto_1fr] items-center gap-x-3 gap-y-3 pl-2">
												{Object.entries(shape.properties).map(
													([childKey, childProp]) => {
														if (!getScalarType(childProp)) return null;
														// Surface the parent's ``default`` as each child's
														// placeholder when the child has none of its own,
														// so users see the model-card defaults at a glance.
														const propWithDefault =
															childProp.default !== undefined
																? childProp
																: {
																		...childProp,
																		default: parentDefaults[childKey],
																	};
														return (
															<ScalarField
																key={childKey}
																id={`param-${key}-${childKey}`}
																label={childProp.title ?? childKey}
																required={shape.required.includes(childKey)}
																prop={propWithDefault}
																value={parentValue[childKey]}
																onChange={(v) =>
																	setNested(
																		key,
																		childKey,
																		v,
																		parentDefaults,
																	)
																}
															/>
														);
													},
												)}
											</div>
										</div>
									);
								})}
							</div>
						)}
					</DropdownMenuSubContent>
				</DropdownMenuSub>
			</DropdownMenuContent>
		</DropdownMenu>
	);
}
