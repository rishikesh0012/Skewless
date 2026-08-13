export const FEATURE_MODES = ['broken', 'correct'] as const
export type FeatureMode = (typeof FEATURE_MODES)[number]

export const SKEW_MODES = ['none', 'distance_unit', 'timezone'] as const
export type SkewMode = (typeof SKEW_MODES)[number]

export const SKEW_MODE_LABELS: Record<SkewMode, string> = {
  none: 'None — baseline parity',
  distance_unit: 'Distance unit — miles interpreted as km',
  timezone: 'Timezone — UTC instead of New York',
}

export const SKEW_MODE_DESCRIPTIONS: Record<SkewMode, string> = {
  none: 'No fault is injected. The independent paths should produce the same nine values.',
  distance_unit:
    'The serving path multiplies distance by 1.609344, so the model receives kilometres in a feature trained as miles.',
  timezone:
    'The serving path derives calendar features in UTC while the training path uses America/New_York.',
}

export interface FeatureComparison {
  feature: string
  training_value: number
  serving_value: number
  matched: boolean
  absolute_difference: number
}

export interface FeatureMismatch extends FeatureComparison {
  relative_difference: number
}

export interface ParityReport {
  matched: boolean
  matched_count: number
  mismatch_count: number
  total_features: 9
  comparisons: FeatureComparison[]
  mismatches: FeatureMismatch[]
}

export interface PredictionResult {
  predicted_fare_amount: number
  model_name: string
  feature_mode: FeatureMode
  requested_skew_mode: SkewMode
  applied_skew_mode: SkewMode
  parity: ParityReport
}

export interface PredictionRequest {
  pickup_datetime: string
  pickup_location_id: number
  dropoff_location_id: number
  passenger_count: number | null
  trip_distance_miles: number
  feature_mode: FeatureMode
  skew_mode: SkewMode
}

export interface HealthResponse {
  status: 'ok'
  model_name: string
}

export interface ModelInfoResponse {
  model_name: string
  feature_names: string[]
  metadata: Record<string, unknown>
}

const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL ?? 'http://localhost:8000'
).replace(/\/$/, '')

export class ApiError extends Error {}

async function request<T>(route: string, init?: RequestInit): Promise<T> {
  let response: Response

  try {
    response = await fetch(`${API_BASE_URL}${route}`, init)
  } catch {
    throw new ApiError(
      `Cannot reach the FastAPI service at ${API_BASE_URL}. Start it with uvicorn api.main:app --reload.`,
    )
  }

  if (!response.ok) {
    const detail = await response.text()
    throw new ApiError(
      `${route} failed with ${response.status}. ${detail.slice(0, 300)}`,
    )
  }

  return (await response.json()) as T
}

export function predict(requestBody: PredictionRequest) {
  return request<PredictionResult>('/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(requestBody),
  })
}

export function fetchHealth() {
  return request<HealthResponse>('/health')
}

export function fetchModelInfo() {
  return request<ModelInfoResponse>('/model-info')
}
