/**
 * ReasoningAccordion component - collapsible list of reasoning events.
 *
 * Displays reasoning steps in an accordion format, allowing users to
 * expand individual steps to view detailed metadata.
 *
 * ITERATION_START events act as group toggles - when collapsed (default),
 * they hide all subsequent events until ITERATION_COMPLETE. Each event
 * within an iteration remains individually collapsible.
 */

import * as React from 'react';
import { ChevronDown } from 'lucide-react';
import { Accordion, AccordionItem } from '../ui/accordion';
import { ReasoningEventType, type ReasoningEvent } from '../../types/openai';
import { ReasoningStep, ReasoningStepMetadata, hasDisplayableContent } from './ReasoningStep';
import { cn } from '../../lib/utils';

export interface ReasoningAccordionProps {
  events: ReasoningEvent[];
  className?: string;
}

/**
 * Renders a collapsible list of reasoning events.
 *
 * Each event can be expanded to view metadata and error details.
 * ITERATION_START events control visibility of their iteration group.
 *
 * @example
 * ```tsx
 * <ReasoningAccordion
 *   events={[
 *     { type: ReasoningEventType.IterationStart, step_iteration: 1, metadata: {} },
 *     { type: ReasoningEventType.Planning, step_iteration: 1, metadata: {} },
 *     { type: ReasoningEventType.IterationComplete, step_iteration: 1, metadata: {} }
 *   ]}
 * />
 * ```
 */
export const ReasoningAccordion = React.memo<ReasoningAccordionProps>(
  ({ events, className }) => {
    // Track which iterations are expanded (default: all collapsed)
    const [expandedIterations, setExpandedIterations] = React.useState<Set<number>>(new Set());

    // Toggle iteration expansion
    const toggleIteration = React.useCallback((stepIteration: number) => {
      setExpandedIterations((prev) => {
        const next = new Set(prev);
        if (next.has(stepIteration)) {
          next.delete(stepIteration);
        } else {
          next.add(stepIteration);
        }
        return next;
      });
    }, []);

    // Track which events belong to which iteration
    const eventIterations = React.useMemo(() => {
      const map = new Map<number, number>(); // event index -> iteration number
      let currentIteration: number | null = null;

      events.forEach((event, index) => {
        if (event.type === ReasoningEventType.IterationStart) {
          currentIteration = event.step_iteration;
          map.set(index, currentIteration);
        } else if (currentIteration !== null) {
          map.set(index, currentIteration);
          if (event.type === ReasoningEventType.IterationComplete) {
            currentIteration = null;
          }
        }
      });

      return map;
    }, [events]);

    return (
      <Accordion className={className}>
        {events.map((event, index) => {
          const showStep = event.type !== ReasoningEventType.ReasoningComplete;
          const eventIteration = eventIterations.get(index);
          const isIterationStart = event.type === ReasoningEventType.IterationStart;
          const isIterationExpanded = eventIteration !== undefined && expandedIterations.has(eventIteration);

          // Hide events within collapsed iterations (except ITERATION_START itself)
          if (eventIteration !== undefined && !isIterationStart && !isIterationExpanded) {
            return null;
          }

          // ITERATION_START acts as a toggle for the entire iteration group
          if (isIterationStart) {
            return (
              <div key={index} className="border rounded-md overflow-hidden">
                <button
                  type="button"
                  onClick={() => toggleIteration(event.step_iteration)}
                  className="flex w-full items-center justify-between px-3 py-2 text-left text-sm font-medium hover:bg-muted/50 transition-colors"
                >
                  <ReasoningStep event={event} showStep={showStep} />
                  <ChevronDown
                    className={cn(
                      'h-4 w-4 shrink-0 transition-transform duration-200',
                      isIterationExpanded && 'rotate-180',
                    )}
                  />
                </button>
              </div>
            );
          }

          // Regular event - disable collapsing if no content
          const disabled = !hasDisplayableContent(event);

          // Nested events within an iteration should be indented
          const isNested = eventIteration !== undefined;

          return (
            <div key={index} className={cn(isNested && 'ml-4 mr-2')}>
              <AccordionItem
                title={<ReasoningStep event={event} showStep={showStep} />}
                disabled={disabled}
              >
                <ReasoningStepMetadata event={event} />
              </AccordionItem>
            </div>
          );
        })}
      </Accordion>
    );
  },
);

ReasoningAccordion.displayName = 'ReasoningAccordion';
