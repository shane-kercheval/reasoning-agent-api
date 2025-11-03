/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // Reasoning event colors (can be customized later)
        reasoning: {
          search: '#3b82f6',      // blue
          thinking: '#a855f7',    // purple
          action: '#f97316',      // orange
          complete: '#22c55e',    // green
        },
      },
    },
  },
  plugins: [],
}
