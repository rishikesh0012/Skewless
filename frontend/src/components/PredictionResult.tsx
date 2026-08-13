import { AlertTriangle, Check, CircleCheck, Inbox, X } from 'lucide-react'

import { Badge } from '@/components/ui/badge'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Table, TableCell, TableHead } from '@/components/ui/table'
import { SKEW_MODE_LABELS, type PredictionResult } from '@/lib/api'

const currency = new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' })
const number = new Intl.NumberFormat('en-US', { maximumFractionDigits: 6 })

interface ResultPanelProps {
  result: PredictionResult | null
  error: string | null
  isPending: boolean
}

export function ResultPanel({ result, error, isPending }: ResultPanelProps) {
  return (
    <Card>
      <CardHeader className="flex-row items-center justify-between">
        <div>
          <CardTitle>Parity report</CardTitle>
          <p className="mt-1 text-sm text-ink-muted">Training reference versus scored serving vector.</p>
        </div>
        {result ? <Badge tone={result.parity.matched ? 'ok' : 'alert'}>{result.feature_mode}</Badge> : null}
      </CardHeader>
      <CardContent>
        {error ? (
          <Placeholder tone="alert" icon={<AlertTriangle className="h-5 w-5" />} title="Request failed" body={error} />
        ) : isPending ? (
          <SkeletonResult />
        ) : result ? (
          <Result result={result} />
        ) : (
          <Placeholder tone="neutral" icon={<Inbox className="h-5 w-5" />} title="Ready to compare" body="Run the default distance-unit scenario, then switch only the feature architecture to Correct." />
        )}
      </CardContent>
    </Card>
  )
}

function Result({ result }: { result: PredictionResult }) {
  const parity = result.parity

  return (
    <div className="space-y-6">
      <div className="grid gap-3 sm:grid-cols-2">
        <Metric label="Predicted fare" value={currency.format(result.predicted_fare_amount)} large />
        <Metric label="Feature parity" value={`${parity.matched_count} / ${parity.total_features}`} detail="features matched" />
      </div>

      <div className={`rounded-lg border px-4 py-4 ${parity.matched ? 'border-ok/30 bg-ok-soft' : 'border-alert/25 bg-alert-soft'}`}>
        <div className="flex items-start gap-3">
          {parity.matched ? <CircleCheck className="mt-0.5 h-5 w-5 shrink-0 text-ok" /> : <AlertTriangle className="mt-0.5 h-5 w-5 shrink-0 text-alert" />}
          <div>
            <p className="font-600 text-ink">{parity.matched ? 'Perfect parity' : 'Training-serving skew detected'}</p>
            <p className="mt-1 text-sm text-ink-muted">
              {parity.matched
                ? result.feature_mode === 'correct'
                  ? 'Both vectors came from shared.py. The selected fault had nowhere to enter.'
                  : 'The independent paths happened to agree for this request.'
                : `${parity.mismatch_count} feature${parity.mismatch_count === 1 ? '' : 's'} diverged before the serving vector was scored.`}
            </p>
          </div>
        </div>
      </div>

      <div className="flex flex-wrap gap-x-6 gap-y-2 text-xs text-ink-subtle">
        <span><strong className="font-600 text-ink-muted">Scenario:</strong> {SKEW_MODE_LABELS[result.requested_skew_mode]}</span>
        <span><strong className="font-600 text-ink-muted">Applied:</strong> {result.applied_skew_mode === 'none' ? 'No fault' : SKEW_MODE_LABELS[result.applied_skew_mode]}</span>
        <span className="font-mono">{result.model_name}</span>
      </div>

      <Table>
        <thead>
          <tr>
            <TableHead>Feature</TableHead>
            <TableHead className="text-right">Training</TableHead>
            <TableHead className="text-right">Serving</TableHead>
            <TableHead className="text-right">Status</TableHead>
          </tr>
        </thead>
        <tbody>
          {parity.comparisons.map((item) => (
            <tr key={item.feature} className={item.matched ? '' : 'bg-alert-soft/65'}>
              <TableCell className="font-mono text-xs">{item.feature}</TableCell>
              <TableCell className="text-right font-mono tabular-nums">{number.format(item.training_value)}</TableCell>
              <TableCell className={`text-right font-mono tabular-nums ${item.matched ? '' : 'font-600 text-alert'}`}>{number.format(item.serving_value)}</TableCell>
              <TableCell className="text-right">
                {item.matched ? (
                  <span className="inline-flex items-center gap-1 text-xs text-ok"><Check className="h-3.5 w-3.5" />Match</span>
                ) : (
                  <span className="inline-flex items-center gap-1 text-xs font-600 text-alert"><X className="h-3.5 w-3.5" />Δ {number.format(item.absolute_difference)}</span>
                )}
              </TableCell>
            </tr>
          ))}
        </tbody>
      </Table>
    </div>
  )
}

function Metric({ label, value, detail, large = false }: { label: string; value: string; detail?: string; large?: boolean }) {
  return (
    <div className="rounded-lg border border-line bg-surface-muted px-4 py-4">
      <p className="font-mono text-[10px] tracking-[0.14em] text-ink-subtle uppercase">{label}</p>
      <p className={`mt-1 font-mono font-600 tracking-tight text-ink tabular-nums ${large ? 'text-4xl' : 'text-3xl'}`}>{value}</p>
      {detail ? <p className="mt-1 text-xs text-ink-subtle">{detail}</p> : null}
    </div>
  )
}

function SkeletonResult() {
  return <div className="space-y-4" aria-busy="true"><span className="sr-only">Comparing features…</span><div className="grid grid-cols-2 gap-3"><div className="h-24 animate-pulse rounded-lg bg-surface-muted" /><div className="h-24 animate-pulse rounded-lg bg-surface-muted" /></div><div className="h-20 animate-pulse rounded-lg bg-surface-muted" /><div className="h-64 animate-pulse rounded-lg bg-surface-muted" /></div>
}

function Placeholder({ tone, icon, title, body }: { tone: 'neutral' | 'alert'; icon: React.ReactNode; title: string; body: string }) {
  const alert = tone === 'alert'
  return (
    <div className={`flex gap-3 rounded-lg border px-4 py-5 ${alert ? 'border-alert/30 bg-alert-soft' : 'border-line bg-surface-muted'}`}>
      <span className={alert ? 'text-alert' : 'text-ink-subtle'}>{icon}</span>
      <div><p className="font-600 text-ink">{title}</p><p className="mt-1 text-sm text-ink-muted">{body}</p></div>
    </div>
  )
}
