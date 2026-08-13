import { useState } from 'react'

import { PipelineStrip } from '@/components/PipelineStrip'
import { PredictionForm } from '@/components/PredictionForm'
import { ResultPanel } from '@/components/PredictionResult'
import { SiteHeader } from '@/components/SiteHeader'
import { SystemStatus } from '@/components/SystemStatus'
import {
  ApiError,
  predict,
  type FeatureMode,
  type PredictionRequest,
  type PredictionResult,
} from '@/lib/api'

export default function App() {
  const [result, setResult] = useState<PredictionResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [isPending, setIsPending] = useState(false)
  const [featureMode, setFeatureMode] = useState<FeatureMode>('broken')

  async function handleSubmit(request: PredictionRequest) {
    setIsPending(true)
    setError(null)

    try {
      setResult(await predict(request))
    } catch (caught) {
      setResult(null)
      setError(
        caught instanceof ApiError
          ? caught.message
          : 'Unexpected error while scoring the trip.',
      )
    } finally {
      setIsPending(false)
    }
  }

  return (
    <div className="min-h-dvh">
      <SiteHeader />

      <main className="mx-auto max-w-7xl space-y-8 px-5 py-10 sm:px-8 sm:py-14">
        <section className="grid gap-8 lg:grid-cols-[minmax(0,1fr)_22rem] lg:items-end">
          <div className="max-w-3xl">
            <p className="font-mono text-xs font-600 tracking-[0.16em] text-accent uppercase">
              Training-serving feature parity
            </p>
            <h1 className="mt-3 text-4xl font-700 tracking-[-0.035em] text-ink sm:text-5xl lg:text-6xl">
              Same model. Different features. Different outcome.
            </h1>
            <p className="mt-5 max-w-2xl text-lg leading-8 text-ink-muted">
              Compare duplicated training and serving transformations, inject a
              realistic fault, then switch to one shared transformation and watch
              parity return—all on the same taxi trip.
            </p>
          </div>
          <SystemStatus />
        </section>

        <PipelineStrip featureMode={featureMode} />

        <div className="grid items-start gap-6 lg:grid-cols-[minmax(0,0.92fr)_minmax(0,1.08fr)]">
          <PredictionForm
            onSubmit={handleSubmit}
            onModeChange={setFeatureMode}
            isPending={isPending}
          />
          <ResultPanel result={result} error={error} isPending={isPending} />
        </div>
      </main>

      <footer className="mt-8 border-t border-line">
        <div className="mx-auto flex max-w-7xl flex-wrap justify-between gap-2 px-5 py-6 text-sm text-ink-subtle sm:px-8">
          <p>Skewless — Training-Serving Feature Parity</p>
          <p className="font-mono text-xs">FastAPI · joblib · LightGBM · React</p>
        </div>
      </footer>
    </div>
  )
}
