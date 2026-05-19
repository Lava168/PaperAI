const agents = [
  {
    id: "orchestrator",
    name: "Orchestrator Agent",
    file: "agents/ORCHESTRATOR_AGENT.md",
    use: "Coordinate a full manuscript from materials to revision plan.",
  },
  {
    id: "literature",
    name: "Literature Agent",
    file: "agents/LITERATURE_AGENT.md",
    use: "Frame related work, research gaps, and citation needs.",
  },
  {
    id: "outline",
    name: "Outline Agent",
    file: "agents/OUTLINE_AGENT.md",
    use: "Create titles, abstract plans, contributions, and section structure.",
  },
  {
    id: "methods",
    name: "Methods Agent",
    file: "agents/METHODS_AGENT.md",
    use: "Write reproducible Methods from datasets, code, configs, and protocols.",
  },
  {
    id: "results",
    name: "Results Agent",
    file: "agents/RESULTS_AGENT.md",
    use: "Write Results from figures, tables, metrics, and statistical tests.",
  },
  {
    id: "figure_table",
    name: "Figure Table Agent",
    file: "agents/FIGURE_TABLE_AGENT.md",
    use: "Write standalone captions, table titles, and table notes.",
  },
  {
    id: "discussion",
    name: "Discussion Agent",
    file: "agents/DISCUSSION_AGENT.md",
    use: "Write Discussion, limitations, future work, and conclusion.",
  },
  {
    id: "citation",
    name: "Citation Agent",
    file: "agents/CITATION_AGENT.md",
    use: "Check citations, missing references, and novelty claim risk.",
  },
  {
    id: "reviewer",
    name: "Reviewer Agent",
    file: "agents/REVIEWER_AGENT.md",
    use: "Review a manuscript like a critical peer reviewer.",
  },
];

const workflowSteps = [
  ["Intake", "Collect project path, paper type, venue, figures, tables, and notes."],
  ["Inventory", "List materials and map each source to a claim or section."],
  ["Claim Map", "Link every major claim to evidence or mark missing support."],
  ["Outline", "Plan title, abstract, contributions, sections, and figure placement."],
  ["Methods", "Write reproducible methods from concrete materials."],
  ["Results", "Report metrics, uncertainty, baselines, and figure/table references."],
  ["Captions", "Generate standalone captions and table notes."],
  ["Review", "Run citation and reviewer checks before final revision."],
];

const state = {
  selectedAgent: agents[0],
};

const agentGrid = document.querySelector("#agentGrid");
const workflowGrid = document.querySelector("#workflowSteps");
const promptOutput = document.querySelector("#promptOutput");
const generateBtn = document.querySelector("#generateBtn");
const runBackendBtn = document.querySelector("#runBackendBtn");
const runQwenBtn = document.querySelector("#runQwenBtn");
const copyBtn = document.querySelector("#copyBtn");
const downloadBtn = document.querySelector("#downloadBtn");
const resetBtn = document.querySelector("#resetBtn");
const backendStatus = document.querySelector("#backendStatus");
const runMeta = document.querySelector("#runMeta");
let lastDownload = null;

async function checkBackendHealth() {
  try {
    const response = await fetch("/api/health");
    if (!response.ok) {
      throw new Error(`Backend returned ${response.status}`);
    }
    const data = await response.json();
    backendStatus.textContent = data.qwenConfigured
      ? `Backend connected, Qwen ready: ${data.defaultQwenModel}`
      : "Backend connected, Qwen key missing";
  } catch (error) {
    backendStatus.textContent = "Backend offline: prompt-only mode";
  }
}

async function loadBackendAgents() {
  try {
    const response = await fetch("/api/agents");
    if (!response.ok) {
      throw new Error(`Backend returned ${response.status}`);
    }
    const data = await response.json();
    if (data.ok && Array.isArray(data.agents)) {
      agents.splice(0, agents.length, ...data.agents);
      state.selectedAgent = agents[0];
      renderAgents();
      generatePrompt();
    }
  } catch (error) {
    backendStatus.textContent = "Backend offline: prompt-only mode";
  }
}

function renderAgents() {
  agentGrid.innerHTML = "";

  agents.forEach((agent) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = `agent-card ${agent.id === state.selectedAgent.id ? "active" : ""}`;
    button.innerHTML = `<strong>${agent.name}</strong><span>${agent.use}</span>`;
    button.addEventListener("click", () => {
      state.selectedAgent = agent;
      renderAgents();
      generatePrompt();
    });
    agentGrid.appendChild(button);
  });
}

function renderWorkflow() {
  workflowGrid.innerHTML = "";

  workflowSteps.forEach(([title, description], index) => {
    const item = document.createElement("div");
    item.className = "workflow-step";
    item.innerHTML = `<b>${index + 1}. ${title}</b><span>${description}</span>`;
    workflowGrid.appendChild(item);
  });
}

function getFormValue(id) {
  return document.querySelector(`#${id}`).value.trim();
}

function getPayload() {
  const agent = state.selectedAgent;
  return {
    agentId: agent.id,
    projectPath: getFormValue("projectPath") || "[PROJECT_PATH]",
    title: getFormValue("title") || "[PAPER_TITLE]",
    topic: getFormValue("topic") || "[PAPER_TOPIC]",
    venue: getFormValue("venue") || "[TARGET_VENUE_OR_STYLE]",
    paperType: getFormValue("paperType") || "[PAPER_TYPE]",
    desiredOutput: getFormValue("desiredOutput") || "Full manuscript draft",
    model: getFormValue("model") || "qwen3.6-max-preview",
    task: getFormValue("task") || "[DESCRIBE_THE_WRITING_TASK]",
    materials: getFormValue("materials") || "[LIST_FILES_FIGURES_TABLES_NOTES_TO_READ]",
  };
}

function generatePrompt() {
  const agent = state.selectedAgent;
  const payload = getPayload();
  const prompt = `Read /home/lry/scientific-paper-writing-agent-system/${agent.file}.
Also read /home/lry/scientific-paper-writing-agent-system/workflows/PAPER_WORKFLOW.md and /home/lry/scientific-paper-writing-agent-system/templates/CLAIM_EVIDENCE_MAP.md.

Operate as the ${agent.name}.

Project or manuscript path:
${payload.projectPath}

Paper title:
${payload.title}

Paper topic:
${payload.topic}

Paper type:
${payload.paperType}

Target venue or writing style:
${payload.venue}

Desired output:
${payload.desiredOutput}

Task:
${payload.task}

Materials, figures, tables, notes, or files to inspect:
${payload.materials}

Output requirements:
1. Write in English unless I explicitly ask otherwise.
2. Do not invent citations, methods, data, metrics, or statistical tests.
3. Mark missing support as [EVIDENCE NEEDED], [CITATION NEEDED], or [METHOD DETAIL NEEDED].
4. Keep claims linked to specific evidence.
5. Generate manuscript-ready content for the requested title/topic.
6. Provide a concise revision checklist after the draft output.
`;

  promptOutput.value = prompt;
  localStorage.setItem("paperAgentPrompt", prompt);
}

async function runBackend() {
  return runBackendRequest(false);
}

async function runQwen() {
  return runBackendRequest(true);
}

async function runBackendRequest(executeModel) {
  runBackendBtn.disabled = true;
  runQwenBtn.disabled = true;
  const activeButton = executeModel ? runQwenBtn : runBackendBtn;
  activeButton.textContent = executeModel ? "Calling Qwen..." : "Running...";
  runMeta.className = "run-meta";
  runMeta.textContent = executeModel
    ? "Calling backend and Qwen model..."
    : "Calling backend /api/run-agent...";

  try {
    const payload = getPayload();
    payload.executeModel = executeModel;
    payload.saveToServer = false;
    const response = await fetch("/api/run-agent", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });
    const data = await response.json();
    if (!response.ok || !data.ok) {
      throw new Error(data.error || `Backend returned ${response.status}`);
    }

    promptOutput.value = data.prompt;
    lastDownload = {
      filename: data.downloadFilename || "paper-agent-output.md",
      content: data.downloadContent || data.prompt,
    };
    if (data.modelOutput) {
      promptOutput.value = `# Qwen Output (${data.model})\n\n${data.modelOutput}\n\n---\n\n# Cursor Prompt\n\n${data.prompt}`;
      lastDownload = {
        filename: data.downloadFilename || "qwen-paper-output.md",
        content: `# Qwen Output (${data.model})\n\n${data.modelOutput}\n`,
      };
      downloadTextFile(lastDownload.filename, lastDownload.content);
    }
    localStorage.setItem("paperAgentPrompt", data.prompt);
    runMeta.className = "run-meta ok";
    runMeta.innerHTML = data.savedPath
      ? `Saved run <strong>${data.runId}</strong><br /><code>${data.savedPath}</code>`
      : `Generated run <strong>${data.runId}</strong>. Result is downloaded by the browser, not saved on the server.`;
  } catch (error) {
    runMeta.className = "run-meta error";
    runMeta.textContent = `Backend run failed: ${error.message}`;
  } finally {
    runBackendBtn.disabled = false;
    runQwenBtn.disabled = false;
    runBackendBtn.textContent = "Run Backend";
    runQwenBtn.textContent = "Run Qwen";
  }
}

function downloadTextFile(filename, content) {
  const blob = new Blob([content], { type: "text/markdown;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

function downloadResult() {
  if (!lastDownload) {
    const fallback = promptOutput.value || "";
    lastDownload = {
      filename: "paper-agent-output.md",
      content: fallback,
    };
  }
  downloadTextFile(lastDownload.filename, lastDownload.content);
}

async function copyPrompt() {
  if (!promptOutput.value.trim()) {
    generatePrompt();
  }

  try {
    await navigator.clipboard.writeText(promptOutput.value);
    copyBtn.textContent = "Copied";
    setTimeout(() => {
      copyBtn.textContent = "Copy Prompt";
    }, 1300);
  } catch (error) {
    promptOutput.select();
    document.execCommand("copy");
  }
}

function resetForm() {
  document.querySelector("#venue").value = "";
  document.querySelector("#title").value = "";
  document.querySelector("#topic").value = "";
  document.querySelector("#paperType").selectedIndex = 0;
  document.querySelector("#desiredOutput").selectedIndex = 0;
  document.querySelector("#model").selectedIndex = 0;
  document.querySelector("#task").value = "";
  document.querySelector("#materials").value = "";
  lastDownload = null;
  state.selectedAgent = agents[0];
  renderAgents();
  generatePrompt();
}

generateBtn.addEventListener("click", generatePrompt);
runBackendBtn.addEventListener("click", runBackend);
runQwenBtn.addEventListener("click", runQwen);
copyBtn.addEventListener("click", copyPrompt);
downloadBtn.addEventListener("click", downloadResult);
resetBtn.addEventListener("click", resetForm);

document.querySelectorAll("input, select, textarea").forEach((field) => {
  if (field.id !== "promptOutput") {
    field.addEventListener("input", generatePrompt);
  }
});

renderAgents();
renderWorkflow();
generatePrompt();
checkBackendHealth();
loadBackendAgents();
