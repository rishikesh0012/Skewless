import type { ComponentProps, ReactNode } from 'react'

import { cn } from '@/lib/utils'

const controlStyles =
  'h-11 w-full rounded-md border border-line-strong bg-surface px-3 font-mono text-sm text-ink transition-colors duration-200 placeholder:text-ink-subtle hover:border-accent/50 focus:border-accent focus:outline-none disabled:opacity-50'

export function Label({ className, ...props }: ComponentProps<'label'>) {
  return (
    <label
      className={cn(
        'block text-sm font-medium text-ink-muted select-none',
        className,
      )}
      {...props}
    />
  )
}

export function Input({ className, ...props }: ComponentProps<'input'>) {
  return <input className={cn(controlStyles, className)} {...props} />
}

export function Select({ className, ...props }: ComponentProps<'select'>) {
  return (
    <select
      // min-w-0 stops the widest option from setting the field's min-content width.
      className={cn(controlStyles, 'min-w-0 cursor-pointer font-sans', className)}
      {...props}
    />
  )
}

interface FieldProps {
  label: string
  htmlFor: string
  hint?: string
  error?: string
  className?: string
  children: ReactNode
}

export function Field({
  label,
  htmlFor,
  hint,
  error,
  className,
  children,
}: FieldProps) {
  return (
    <div className={cn('min-w-0 space-y-1.5', className)}>
      <Label htmlFor={htmlFor}>{label}</Label>
      {children}
      {error ? (
        <p role="alert" className="text-sm font-medium text-alert">
          {error}
        </p>
      ) : hint ? (
        <p className="text-xs text-ink-subtle">{hint}</p>
      ) : null}
    </div>
  )
}
