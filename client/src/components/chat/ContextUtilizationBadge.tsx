/**
 * ContextUtilizationBadge - Visual display of LLM context window utilization.
 *
 * Shows token usage with color-coded progress bar and detailed breakdown.
 */

import * as React from 'react';
import { Tooltip, TooltipContent, TooltipTrigger } from '../ui/tooltip';
import { cn } from '../../lib/utils';

interface ContextUtilization {
  model_name: string;
  strategy: 'low' | 'medium' | 'full';
  model_max_tokens: number;
  max_input_tokens: number;
  input_tokens_used: number;
  messages_included: number;
  messages_excluded: number;
  breakdown: {
    system_messages: number;
    user_messages: number;
    assistant_messages: number;
  };
}

interface ContextUtilizationBadgeProps {
  contextUtilization: ContextUtilization;
  className?: string;
}

/**
 * Format large numbers with commas.
 */
function formatNumber(num: number): string {
  return num.toLocaleString();
}

/**
 * Renders context utilization with stacked bar visualization and detailed tooltip.
 */
export const ContextUtilizationBadge: React.FC<ContextUtilizationBadgeProps> = ({
  contextUtilization,
  className,
}) => {
  // max_input_tokens is already strategy-adjusted from backend
  const allowedTokens = contextUtilization.max_input_tokens;
  const modelMaxTokens = contextUtilization.model_max_tokens;

  // Calculate forbidden space percentage based on original model max
  const forbiddenTokens = modelMaxTokens - allowedTokens;
  const forbiddenPct = (forbiddenTokens / modelMaxTokens) * 100;

  // Calculate percentage of each message type relative to MODEL MAX (for visualization)
  const systemPctOfMax = (contextUtilization.breakdown.system_messages / modelMaxTokens) * 100;
  const userPctOfMax = (contextUtilization.breakdown.user_messages / modelMaxTokens) * 100;
  const assistantPctOfMax = (contextUtilization.breakdown.assistant_messages / modelMaxTokens) * 100;

  // Calculate percentage within used tokens for tooltip
  const total = contextUtilization.input_tokens_used;
  const systemPct = total > 0 ? (contextUtilization.breakdown.system_messages / total) * 100 : 0;
  const userPct = total > 0 ? (contextUtilization.breakdown.user_messages / total) * 100 : 0;
  const assistantPct = total > 0 ? (contextUtilization.breakdown.assistant_messages / total) * 100 : 0;

  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <div className={cn('inline-flex items-center gap-2', className)}>
          {/* Stacked bar chart with border for clarity */}
          <div className="w-32 h-3 rounded-full overflow-hidden bg-muted/30 flex border border-border">
            {/* System messages - Purple (from left) */}
            {systemPctOfMax > 0 && (
              <div
                className="h-full bg-purple-500 transition-all duration-300"
                style={{ width: `${systemPctOfMax}%` }}
              />
            )}
            {/* User messages - Blue (from left) */}
            {userPctOfMax > 0 && (
              <div
                className="h-full bg-blue-500 transition-all duration-300"
                style={{ width: `${userPctOfMax}%` }}
              />
            )}
            {/* Assistant messages - Teal/Turquoise (from left) */}
            {assistantPctOfMax > 0 && (
              <div
                className="h-full bg-teal-500 transition-all duration-300"
                style={{ width: `${assistantPctOfMax}%` }}
              />
            )}
            {/* Spacer to push forbidden space to the right */}
            <div className="flex-1" />
            {/* Forbidden space - dark gray (from right) */}
            {forbiddenPct > 0 && (
              <div
                className="h-full bg-muted-foreground/30 transition-all duration-300"
                style={{ width: `${forbiddenPct}%` }}
              />
            )}
          </div>

          {/* Utilization percentage */}
          <div className="text-xs font-mono text-muted-foreground whitespace-nowrap">
            {((contextUtilization.input_tokens_used / allowedTokens) * 100).toFixed(1)}%
          </div>
        </div>
      </TooltipTrigger>

      <TooltipContent side="top" className="max-w-xs">
        <div className="space-y-2">
          {/* Header */}
          <div className="font-semibold text-sm border-b border-border pb-2">
            Context Window Utilization
          </div>

          {/* Overall stats */}
          <div className="space-y-1 text-xs">
            <div className="flex justify-between gap-4">
              <span className="text-gray-400">Model:</span>
              <span className="font-mono">{contextUtilization.model_name}</span>
            </div>
            <div className="flex justify-between gap-4">
              <span className="text-gray-400">Strategy:</span>
              <span className="font-medium capitalize">{contextUtilization.strategy}</span>
            </div>
            <div className="flex justify-between gap-4">
              <span className="text-gray-400">Allowed Tokens:</span>
              <span className="font-mono">{formatNumber(contextUtilization.max_input_tokens)}</span>
            </div>
            <div className="flex justify-between gap-4">
              <span className="text-gray-400">Model Max:</span>
              <span className="font-mono">{formatNumber(contextUtilization.model_max_tokens)}</span>
            </div>
            <div className="flex justify-between gap-4">
              <span className="text-gray-400">Tokens Used:</span>
              <span className="font-mono">{formatNumber(contextUtilization.input_tokens_used)}</span>
            </div>
          </div>

          {/* Token breakdown */}
          <div className="border-t border-border pt-2">
            <div className="text-xs font-semibold mb-1">Token Breakdown</div>
            <div className="space-y-1 text-xs">
              {/* System messages */}
              {contextUtilization.breakdown.system_messages > 0 && (
                <div className="flex justify-between gap-4 items-center">
                  <div className="flex items-center gap-2">
                    <div className="w-2 h-2 rounded-full bg-purple-500" />
                    <span className="text-gray-400">System:</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="font-mono">{formatNumber(contextUtilization.breakdown.system_messages)}</span>
                    <span className="text-gray-400 text-[10px]">({systemPct.toFixed(1)}%)</span>
                  </div>
                </div>
              )}

              {/* User messages */}
              {contextUtilization.breakdown.user_messages > 0 && (
                <div className="flex justify-between gap-4 items-center">
                  <div className="flex items-center gap-2">
                    <div className="w-2 h-2 rounded-full bg-blue-500" />
                    <span className="text-gray-400">User:</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="font-mono">{formatNumber(contextUtilization.breakdown.user_messages)}</span>
                    <span className="text-gray-400 text-[10px]">({userPct.toFixed(1)}%)</span>
                  </div>
                </div>
              )}

              {/* Assistant messages */}
              {contextUtilization.breakdown.assistant_messages > 0 && (
                <div className="flex justify-between gap-4 items-center">
                  <div className="flex items-center gap-2">
                    <div className="w-2 h-2 rounded-full bg-green-500" />
                    <span className="text-gray-400">Assistant:</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="font-mono">{formatNumber(contextUtilization.breakdown.assistant_messages)}</span>
                    <span className="text-gray-400 text-[10px]">({assistantPct.toFixed(1)}%)</span>
                  </div>
                </div>
              )}
            </div>
          </div>

          {/* Messages info */}
          <div className="border-t border-border pt-2">
            <div className="text-xs space-y-1">
              <div className="flex justify-between gap-4">
                <span className="text-gray-400">Messages Included:</span>
                <span className="font-mono">{contextUtilization.messages_included}</span>
              </div>
              {contextUtilization.messages_excluded > 0 && (
                <div className="flex justify-between gap-4">
                  <span className="text-orange-500 dark:text-orange-400">Messages Excluded:</span>
                  <span className="font-mono text-orange-500 dark:text-orange-400">{contextUtilization.messages_excluded}</span>
                </div>
              )}
            </div>
          </div>
        </div>
      </TooltipContent>
    </Tooltip>
  );
};

ContextUtilizationBadge.displayName = 'ContextUtilizationBadge';
