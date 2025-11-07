/**
 * Mock for react-markdown in tests.
 * Renders children as plain text without actual markdown processing.
 */

import React from 'react';

interface ReactMarkdownProps {
  children: string;
  rehypePlugins?: unknown[];
  [key: string]: unknown;
}

const ReactMarkdown: React.FC<ReactMarkdownProps> = ({ children }) => {
  return <div>{children}</div>;
};

export default ReactMarkdown;
