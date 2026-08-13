import { cva, type VariantProps } from 'class-variance-authority'
import type { ComponentProps } from 'react'

import { cn } from '@/lib/utils'

const badgeVariants = cva(
  'inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 font-mono text-xs font-medium',
  {
    variants: {
      tone: {
        neutral: 'border-line-strong bg-surface-muted text-ink-muted',
        ok: 'border-ok/30 bg-ok-soft text-ok',
        warn: 'border-warn/30 bg-warn-soft text-warn',
        alert: 'border-alert/30 bg-alert-soft text-alert',
        accent: 'border-accent/30 bg-accent-soft text-accent',
      },
    },
    defaultVariants: { tone: 'neutral' },
  },
)

type BadgeProps = ComponentProps<'span'> & VariantProps<typeof badgeVariants>

export function Badge({ className, tone, ...props }: BadgeProps) {
  return <span className={cn(badgeVariants({ tone }), className)} {...props} />
}
