/**
 * Minimal Accordion component for collapsible content.
 */

import * as React from 'react';
import { ChevronDown } from 'lucide-react';
import { cn } from '../../lib/utils';

interface AccordionItemProps {
  title: React.ReactNode;
  children: React.ReactNode;
  defaultOpen?: boolean;
  disabled?: boolean;
  className?: string;
}

export function AccordionItem({
  title,
  children,
  defaultOpen = false,
  disabled = false,
  className,
}: AccordionItemProps): JSX.Element {
  const [isOpen, setIsOpen] = React.useState(defaultOpen);

  // If disabled, just show the title without collapse functionality
  if (disabled) {
    return (
      <div className={cn('border rounded-md overflow-hidden', className)}>
        <div className="px-3 py-2 text-sm font-medium">{title}</div>
      </div>
    );
  }

  return (
    <div className={cn('border rounded-md overflow-hidden', className)}>
      <button
        type="button"
        onClick={() => setIsOpen(!isOpen)}
        className="flex w-full items-center justify-between px-3 py-2 text-left text-sm font-medium hover:bg-muted/50 transition-colors"
      >
        {title}
        <ChevronDown
          className={cn(
            'h-4 w-4 shrink-0 transition-transform duration-200',
            isOpen && 'rotate-180',
          )}
        />
      </button>
      {isOpen && <div className="px-3 py-2 text-sm border-t overflow-x-auto">{children}</div>}
    </div>
  );
}

export function Accordion({ children, className }: { children: React.ReactNode; className?: string }): JSX.Element {
  return <div className={cn('space-y-2', className)}>{children}</div>;
}
