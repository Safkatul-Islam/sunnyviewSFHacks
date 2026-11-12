import { defineConfig } from "vitest/config"
import { fileURLToPath } from "node:url"

export default defineConfig({
  resolve: {
    alias: {
      "@": fileURLToPath(new URL("./", import.meta.url)),
    },
  },
  test: {
    environment: "node",
    include: ["lib/**/*.test.ts", "app/**/*.test.ts", "components/**/*.test.ts", "components/**/*.test.tsx"],
    coverage: {
      provider: "v8",
      include: ["lib/**/*.ts", "app/api/**/*.ts", "components/**/*.ts"],
      exclude: [
        "**/*.test.ts",
        "**/*.test.tsx",
        "components/**/types.ts",
        "**/*.d.ts",
        "lib/**/*.tsx",
        "lib/*-types.ts",
        "test/**",
      ],
      reporter: ["text-summary", "text"],
      // CI fails if coverage drops below these floors (currently ~90% lines, ~91% branches).
      thresholds: { lines: 85, statements: 85, branches: 85, functions: 90 },
    },
  },
})
