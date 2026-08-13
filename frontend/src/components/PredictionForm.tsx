import { zodResolver } from '@hookform/resolvers/zod'
import { Check, Copy, GitMerge, Loader2, Play } from 'lucide-react'
import { useEffect } from 'react'
import { useForm } from 'react-hook-form'
import { z } from 'zod'

import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Field, Input, Select } from '@/components/ui/field'
import {
  FEATURE_MODES,
  SKEW_MODE_DESCRIPTIONS,
  SKEW_MODE_LABELS,
  SKEW_MODES,
  type FeatureMode,
  type PredictionRequest,
} from '@/lib/api'
import { cn } from '@/lib/utils'

const formSchema = z.object({
  pickup_datetime: z.string().min(1, 'Pickup date and time is required.'),
  trip_distance_miles: z.number().gt(0).max(500),
  passenger_count: z.number().int().min(1).max(8),
  pickup_location_id: z.number().int().min(1),
  dropoff_location_id: z.number().int().min(1),
  feature_mode: z.enum(FEATURE_MODES),
  skew_mode: z.enum(SKEW_MODES),
})

type FormValues = z.infer<typeof formSchema>

const defaultValues: FormValues = {
  pickup_datetime: '2024-01-08T13:30',
  trip_distance_miles: 4.5,
  passenger_count: 2,
  pickup_location_id: 132,
  dropoff_location_id: 236,
  feature_mode: 'broken',
  skew_mode: 'distance_unit',
}

interface PredictionFormProps {
  onSubmit: (request: PredictionRequest) => Promise<void>
  onModeChange: (mode: FeatureMode) => void
  isPending: boolean
}

export function PredictionForm({ onSubmit, onModeChange, isPending }: PredictionFormProps) {
  const {
    register,
    handleSubmit,
    watch,
    setValue,
    formState: { errors },
  } = useForm<FormValues>({ resolver: zodResolver(formSchema), defaultValues })

  const selectedMode = watch('feature_mode')
  const selectedSkew = watch('skew_mode')

  useEffect(() => onModeChange(selectedMode), [onModeChange, selectedMode])

  const submit = handleSubmit(async (values) => {
    await onSubmit({
      ...values,
      pickup_datetime: `${values.pickup_datetime}:00Z`,
    })
  })

  return (
    <Card>
      <CardHeader className="flex-row items-center justify-between">
        <div>
          <CardTitle>Experiment</CardTitle>
          <p className="mt-1 text-sm text-ink-muted">Keep the trip fixed. Change only the architecture.</p>
        </div>
        <span className="font-mono text-xs text-ink-subtle">POST /predict</span>
      </CardHeader>

      <CardContent>
        <form onSubmit={submit} noValidate className="space-y-6">
          <fieldset>
            <legend className="mb-2 text-sm font-600 text-ink">Feature architecture</legend>
            <div className="grid grid-cols-2 gap-2 rounded-lg bg-surface-muted p-1.5">
              <ModeButton
                active={selectedMode === 'broken'}
                icon={<Copy className="h-4 w-4" />}
                title="Broken"
                detail="Duplicated paths"
                onClick={() => setValue('feature_mode', 'broken')}
              />
              <ModeButton
                active={selectedMode === 'correct'}
                icon={<GitMerge className="h-4 w-4" />}
                title="Correct"
                detail="Shared transform"
                onClick={() => setValue('feature_mode', 'correct')}
                correct
              />
            </div>
            <input type="hidden" {...register('feature_mode')} />
          </fieldset>

          <div className="grid gap-4 sm:grid-cols-2">
            <Field
              label="Pickup date and time (UTC)"
              htmlFor="pickup_datetime"
              className="sm:col-span-2"
              error={errors.pickup_datetime?.message}
              hint="Kept in UTC so the same demo produces the same result on every machine."
            >
              <Input id="pickup_datetime" type="datetime-local" {...register('pickup_datetime')} />
            </Field>
            <Field label="Distance (miles)" htmlFor="trip_distance_miles" error={errors.trip_distance_miles?.message}>
              <Input id="trip_distance_miles" type="number" step="0.1" {...register('trip_distance_miles', { valueAsNumber: true })} />
            </Field>
            <Field label="Passengers" htmlFor="passenger_count" error={errors.passenger_count?.message}>
              <Input id="passenger_count" type="number" {...register('passenger_count', { valueAsNumber: true })} />
            </Field>
            <Field label="Pickup zone" htmlFor="pickup_location_id" error={errors.pickup_location_id?.message}>
              <Input id="pickup_location_id" type="number" {...register('pickup_location_id', { valueAsNumber: true })} />
            </Field>
            <Field label="Dropoff zone" htmlFor="dropoff_location_id" error={errors.dropoff_location_id?.message}>
              <Input id="dropoff_location_id" type="number" {...register('dropoff_location_id', { valueAsNumber: true })} />
            </Field>
            <Field label="Skew scenario" htmlFor="skew_mode" className="sm:col-span-2" error={errors.skew_mode?.message}>
              <Select id="skew_mode" {...register('skew_mode')}>
                {SKEW_MODES.map((mode) => <option key={mode} value={mode}>{SKEW_MODE_LABELS[mode]}</option>)}
              </Select>
            </Field>
          </div>

          <div className={cn('rounded-lg border px-4 py-3 text-sm', selectedMode === 'correct' ? 'border-ok/25 bg-ok-soft' : 'border-line bg-surface-muted')}>
            <div className="flex gap-2.5">
              {selectedMode === 'correct' ? <Check className="mt-0.5 h-4 w-4 shrink-0 text-ok" /> : null}
              <p className="text-ink-muted">
                {selectedMode === 'correct'
                  ? `“${SKEW_MODE_LABELS[selectedSkew]}” stays selected for comparison, but shared.py removes the fault injection point.`
                  : SKEW_MODE_DESCRIPTIONS[selectedSkew]}
              </p>
            </div>
          </div>

          <Button type="submit" disabled={isPending} className="w-full">
            {isPending ? <><Loader2 className="h-4 w-4 animate-spin" />Comparing paths…</> : <><Play className="h-4 w-4" />Run parity check</>}
          </Button>
        </form>
      </CardContent>
    </Card>
  )
}

function ModeButton({ active, icon, title, detail, onClick, correct = false }: { active: boolean; icon: React.ReactNode; title: string; detail: string; onClick: () => void; correct?: boolean }) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-pressed={active}
      className={cn(
        'flex items-center gap-3 rounded-md border px-3 py-3 text-left transition-all',
        active ? (correct ? 'border-ok/35 bg-surface text-ok shadow-sm' : 'border-warn/35 bg-surface text-warn shadow-sm') : 'border-transparent text-ink-muted hover:bg-surface/60',
      )}
    >
      {icon}
      <span><span className="block text-sm font-600 text-ink">{title}</span><span className="block text-xs">{detail}</span></span>
    </button>
  )
}
