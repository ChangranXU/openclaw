import type { OpenClawPluginApi } from "openclaw/plugin-sdk";
import { emptyPluginConfigSchema } from "openclaw/plugin-sdk";
import { createDiagnosticsLangfuseService } from "./src/service.js";

const plugin = {
  id: "diagnostics-langfuse",
  name: "Diagnostics Langfuse",
  description: "Export diagnostics events to Langfuse for LLM tracing visualization",
  configSchema: emptyPluginConfigSchema(),
  register(api: OpenClawPluginApi) {
    api.registerService(createDiagnosticsLangfuseService());
  },
};

export default plugin;
