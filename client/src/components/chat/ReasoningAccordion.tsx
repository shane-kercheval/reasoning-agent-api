/**
 * ReasoningAccordion component - collapsible list of reasoning events.
 *
 * Displays reasoning steps in an accordion format, allowing users to
 * expand individual steps to view detailed metadata.
 */

import * as React from 'react';
import { Accordion, AccordionItem } from '../ui/accordion';
import { ReasoningEventType, type ReasoningEvent } from '../../types/openai';
import { ReasoningStep, ReasoningStepMetadata, hasDisplayableContent } from './ReasoningStep';

export interface ReasoningAccordionProps {
  events: ReasoningEvent[];
  className?: string;
}

/**
 * Renders a collapsible list of reasoning events.
 *
 * Each event can be expanded to view metadata and error details.
 *
 * @example
 * ```tsx
 * <ReasoningAccordion
 *   events={[
 *     { type: ReasoningEventType.Planning, step_iteration: 1, metadata: {} },
 *     { type: ReasoningEventType.ToolResult, step_iteration: 1, metadata: { result: "..." } }
 *   ]}
 * />
 * ```
 */
export const ReasoningAccordion = React.memo<ReasoningAccordionProps>(
  ({ events, className }) => {
    return (
      <Accordion className={className}>
        {events.map((event, index) => {
          // Don't show step number for ReasoningComplete
          const showStep = event.type !== ReasoningEventType.ReasoningComplete;

          // Disable collapsing if event has no displayable content
          const disabled = !hasDisplayableContent(event);

          return (
            <AccordionItem
              key={index}
              title={<ReasoningStep event={event} showStep={showStep} />}
              disabled={disabled}
            >
              <ReasoningStepMetadata event={event} />
            </AccordionItem>
          );
        })}
      </Accordion>
    );
  },
);

ReasoningAccordion.displayName = 'ReasoningAccordion';
