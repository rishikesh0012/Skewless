import { ArrowRight, Check, Copy, GitMerge } from 'lucide-react'

import type { FeatureMode } from '@/lib/api'
import { cn } from '@/lib/utils'

export function PipelineStrip({ featureMode }: { featureMode: FeatureMode }) {
  const correct = featureMode === 'correct'

  return (
    <section
      className={cn(
        'overflow-hidden rounded-panel border bg-surface shadow-sm transition-colors',
        correct ? 'border-ok/30' : 'border-warn/35',
      )}
    >
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-line px-5 py-4 sm:px-6">
        <div>
          <p className="font-mono text-xs font-600 tracking-[0.14em] text-ink-subtle uppercase">
            Active architecture
          </p>
          <p className="mt-1 font-medium text-ink">
            {correct ? 'One shared transformation' : 'Two duplicated transformations'}
          </p>
        </div>
        <span
          className={cn(
            'inline-flex items-center gap-2 rounded-full px-3 py-1.5 text-xs font-600',
            correct ? 'bg-ok-soft text-ok' : 'bg-warn-soft text-warn',
          )}
        >
          {correct ? <Check className="h-3.5 w-3.5" /> : <Copy className="h-3.5 w-3.5" />}
          {correct ? 'Parity by design' : 'Skew can enter here'}
        </span>
      </div>

      <div className="grid gap-3 px-5 py-5 sm:grid-cols-[1fr_auto_1.2fr_auto_1fr] sm:items-center sm:px-6">
        <Stage eyebrow="Input" title="Raw taxi trip" />
        <Arrow />
        {correct ? (
          <div className="rounded-lg border border-ok/30 bg-ok-soft px-4 py-3 text-center">
            <GitMerge className="mx-auto h-5 w-5 text-ok" />
            <p className="mt-1 font-mono text-sm font-600 text-ink">shared.py</p>
            <p className="text-xs text-ink-muted">called by both paths</p>
          </div>
        ) : (
          <div className="grid grid-cols-2 gap-2">
            <Path label="Training" file="canonical.py" />
            <Path label="Serving" file="online.py" warning />
          </div>
        )}
        <Arrow />
        <Stage eyebrow="Evidence" title="9-feature parity" />
      </div>
    </section>
  )
}

function Stage({ eyebrow, title }: { eyebrow: string; title: string }) {
  return (
    <div className="rounded-lg border border-line bg-surface-muted px-4 py-3 text-center">
      <p className="font-mono text-[10px] tracking-wider text-ink-subtle uppercase">{eyebrow}</p>
      <p className="mt-0.5 text-sm font-600 text-ink">{title}</p>
    </div>
  )
}

function Path({ label, file, warning = false }: { label: string; file: string; warning?: boolean }) {
  return (
    <div className={cn('rounded-lg border px-3 py-3 text-center', warning ? 'border-warn/30 bg-warn-soft' : 'border-line bg-surface-muted')}>
      <p className="text-[10px] tracking-wider text-ink-subtle uppercase">{label}</p>
      <p className="mt-0.5 font-mono text-xs font-600 text-ink">{file}</p>
    </div>
  )
}

function Arrow() {
  return <ArrowRight aria-hidden className="mx-auto hidden h-4 w-4 text-ink-subtle sm:block" />
}
