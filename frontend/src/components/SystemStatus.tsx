import { AlertTriangle, CircleCheck, Loader2 } from 'lucide-react'
import { useEffect, useState } from 'react'

import { fetchHealth, fetchModelInfo } from '@/lib/api'

type Status =
  | { state: 'loading' }
  | { state: 'online'; model: string; features: number }
  | { state: 'offline' }

export function SystemStatus() {
  const [status, setStatus] = useState<Status>({ state: 'loading' })

  useEffect(() => {
    let cancelled = false
    Promise.all([fetchHealth(), fetchModelInfo()])
      .then(([health, info]) => {
        if (!cancelled) setStatus({ state: 'online', model: health.model_name, features: info.feature_names.length })
      })
      .catch(() => {
        if (!cancelled) setStatus({ state: 'offline' })
      })
    return () => { cancelled = true }
  }, [])

  return (
    <div className="rounded-panel border border-line bg-surface px-5 py-4 shadow-sm">
      <p className="font-mono text-[10px] tracking-[0.14em] text-ink-subtle uppercase">Demo status</p>
      {status.state === 'loading' ? (
        <div className="mt-2 flex items-center gap-2 text-sm text-ink-muted"><Loader2 className="h-4 w-4 animate-spin" />Connecting to FastAPI…</div>
      ) : status.state === 'offline' ? (
        <div className="mt-2 flex items-center gap-2 text-sm text-alert"><AlertTriangle className="h-4 w-4" />API is offline</div>
      ) : (
        <div className="mt-2 space-y-2">
          <div className="flex items-center gap-2 text-sm font-600 text-ok"><CircleCheck className="h-4 w-4" />Ready to score</div>
          <div className="flex flex-wrap gap-x-4 gap-y-1 font-mono text-xs text-ink-subtle"><span>{status.model}</span><span>{status.features} features</span></div>
        </div>
      )}
    </div>
  )
}
