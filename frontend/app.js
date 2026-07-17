const $ = (selector) => document.querySelector(selector);
const state = { projects: [], project: null, run: null, poll: null };

async function api(path, options = {}) {
  const response = await fetch(path, options);
  if (!response.ok) {
    let message = `请求失败 (${response.status})`;
    try { message = (await response.json()).detail || message; } catch (_) {}
    throw new Error(message);
  }
  if (response.status === 204) return null;
  return response.json();
}

function toast(message, error = false) {
  const node = $('#toast');
  node.textContent = message;
  node.className = error ? 'show error' : 'show';
  clearTimeout(node.timer);
  node.timer = setTimeout(() => node.className = '', 3200);
}

function formatBytes(value) {
  if (value < 1024) return `${value} B`;
  if (value < 1024 ** 2) return `${(value / 1024).toFixed(1)} KB`;
  return `${(value / 1024 ** 2).toFixed(1)} MB`;
}

function statusLabel(status) {
  return ({queued:'排队中',running:'运行中',waiting_approval:'等待确认',completed:'已完成',failed:'失败',cancelled:'已终止'})[status] || status;
}

async function boot() {
  bind();
  try {
    const health = await api('/api/health');
    $('#healthDot').className = 'online';
    $('#healthText').textContent = `PaperAI ${health.version}`;
    $('#modelText').textContent = health.modelConfigured ? `模型：${health.model}` : '模型未配置 · 可用 Prompt 模式';
    if (!health.modelConfigured) $('#modeSelect').value = 'prompt';
  } catch (error) {
    $('#healthDot').className = 'offline';
    $('#healthText').textContent = '服务未连接';
    $('#modelText').textContent = error.message;
  }
  await loadProjects();
}

function bind() {
  $('#newProjectBtn').onclick = $('#heroNewBtn').onclick = () => $('#projectDialog').showModal();
  $('#closeDialog').onclick = $('#cancelDialog').onclick = () => $('#projectDialog').close();
  $('#projectForm').onsubmit = createProject;
  $('#fileInput').onchange = uploadFiles;
  $('#runBtn').onclick = createRun;
  $('#runSelect').onchange = (event) => selectRun(event.target.value);
  $('#artifactSelect').onchange = renderArtifact;
  $('#approveBtn').onclick = () => resumeRun(true);
  $('#rejectBtn').onclick = () => resumeRun(false);
}

async function loadProjects(selectId) {
  state.projects = await api('/api/projects');
  const list = $('#projectList');
  list.innerHTML = state.projects.length ? state.projects.map(project => `<button data-id="${project.id}" class="${state.project?.id === project.id ? 'active' : ''}"><span>${escapeHtml(project.title)}</span><small>${project.document_count} 材料 · ${project.run_count} 运行</small></button>`).join('') : '<p class="side-empty">暂无项目</p>';
  list.querySelectorAll('button').forEach(button => button.onclick = () => selectProject(button.dataset.id));
  if (selectId) await selectProject(selectId);
}

async function createProject(event) {
  event.preventDefault();
  const form = new FormData(event.target);
  const payload = Object.fromEntries(form.entries());
  try {
    const project = await api('/api/projects', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload)});
    $('#projectDialog').close();
    event.target.reset();
    await loadProjects(project.id);
    toast('论文项目已创建');
  } catch (error) { toast(error.message, true); }
}

async function selectProject(projectId) {
  clearInterval(state.poll);
  state.project = await api(`/api/projects/${projectId}`);
  state.run = null;
  $('#emptyView').hidden = true;
  $('#workspaceView').hidden = false;
  $('#pageTitle').textContent = state.project.title;
  $('#documentCount').textContent = state.project.documents.length;
  $('#runCount').textContent = state.project.runs.length;
  renderDocuments();
  renderRunSelect();
  $('#exportActions').hidden = true;
  await loadProjects();
  if (state.project.runs.length) await selectRun(state.project.runs[0].id);
}

function renderDocuments() {
  const node = $('#documentList');
  node.innerHTML = state.project.documents.length ? state.project.documents.map(doc => `<div class="document"><div class="file-icon">${doc.filename.split('.').pop().slice(0,4).toUpperCase()}</div><div><strong>${escapeHtml(doc.filename)}</strong><small>${formatBytes(doc.size)} · ${doc.extraction_status === 'ready' ? '文本已提取' : escapeHtml(doc.extraction_status)}</small></div><button class="icon delete-doc" data-id="${doc.id}" title="删除">×</button></div>`).join('') : '<div class="drop-empty">拖入或上传研究材料，Agent 才能基于证据写作。</div>';
  node.querySelectorAll('.delete-doc').forEach(button => button.onclick = () => deleteDocument(button.dataset.id));
}

async function uploadFiles(event) {
  const files = [...event.target.files];
  for (const file of files) {
    const form = new FormData(); form.append('file', file);
    try { await api(`/api/projects/${state.project.id}/documents`, {method:'POST', body:form}); toast(`${file.name} 已完成解析`); }
    catch (error) { toast(`${file.name}: ${error.message}`, true); }
  }
  event.target.value = '';
  await selectProject(state.project.id);
}

async function deleteDocument(documentId) {
  if (!confirm('删除这份研究材料？')) return;
  try { await api(`/api/projects/${state.project.id}/documents/${documentId}`, {method:'DELETE'}); await selectProject(state.project.id); }
  catch (error) { toast(error.message, true); }
}

function renderRunSelect() {
  $('#runSelect').innerHTML = state.project.runs.length ? state.project.runs.map(run => `<option value="${run.id}">${run.workflow} · ${statusLabel(run.status)}</option>`).join('') : '<option value="">暂无运行</option>';
}

async function createRun() {
  const payload = {workflow:$('#workflowSelect').value, execution_mode:$('#modeSelect').value, human_review:$('#reviewCheck').checked};
  $('#runBtn').disabled = true;
  $('#runHint').textContent = '正在创建运行…';
  try {
    const run = await api(`/api/projects/${state.project.id}/runs`, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload)});
    state.project = await api(`/api/projects/${state.project.id}`);
    renderRunSelect();
    await selectRun(run.id);
    toast('多 Agent 工作流已启动');
  } catch (error) { toast(error.message, true); $('#runHint').textContent = error.message; }
  finally { $('#runBtn').disabled = false; }
}

async function selectRun(runId) {
  if (!runId) return;
  clearInterval(state.poll);
  $('#runSelect').value = runId;
  await refreshRun(runId);
  if (!['completed','failed','cancelled'].includes(state.run.status)) state.poll = setInterval(() => refreshRun(runId), 1800);
}

async function refreshRun(runId) {
  try {
    state.run = await api(`/api/runs/${runId}`);
    renderRun();
    if (['completed','failed','cancelled'].includes(state.run.status)) clearInterval(state.poll);
  } catch (error) { clearInterval(state.poll); toast(error.message, true); }
}

function renderRun() {
  $('#runEmpty').hidden = true; $('#runDetail').hidden = false;
  $('#statusBadge').textContent = statusLabel(state.run.status);
  $('#statusBadge').className = `status ${state.run.status}`;
  $('#currentStep').textContent = state.run.current_step ? state.run.current_step.replaceAll('_',' ') : statusLabel(state.run.status);
  $('#progressText').textContent = `${state.run.progress}%`;
  $('#progressBar').style.width = `${state.run.progress}%`;
  $('#progressCount').textContent = `${state.run.progress}%`;
  $('#progressLabel').textContent = statusLabel(state.run.status);
  $('#approvalBox').hidden = state.run.status !== 'waiting_approval';
  $('#runHint').textContent = state.run.error || '';
  $('#exportActions').hidden = !state.run.artifacts.length;
  $('#exportMd').href = `/api/runs/${state.run.id}/export?format=md`;
  $('#exportDocx').href = `/api/runs/${state.run.id}/export?format=docx`;
  $('#exportTex').href = `/api/runs/${state.run.id}/export?format=tex`;
  $('#timeline').innerHTML = state.run.artifacts.map((artifact, index) => `<button data-step="${artifact.step}" class="timeline-item"><span>${String(index + 1).padStart(2,'0')}</span><div><strong>${artifact.step.replaceAll('_',' ')}</strong><small>${artifact.agent_id} agent · 已完成</small></div><i>✓</i></button>`).join('') || '<p class="muted">Agent 正在准备第一个阶段…</p>';
  $('#timeline').querySelectorAll('button').forEach(button => button.onclick = () => { $('#artifactSelect').value = button.dataset.step; renderArtifact(); });
  const selected = $('#artifactSelect').value;
  $('#artifactSelect').innerHTML = '<option value="">选择产物</option>' + state.run.artifacts.map(a => `<option value="${a.step}">${a.step.replaceAll('_',' ')} · ${a.agent_id}</option>`).join('');
  if (state.run.artifacts.some(item => item.step === selected)) $('#artifactSelect').value = selected;
  else if (state.run.artifacts.length) $('#artifactSelect').value = state.run.artifacts.at(-1).step;
  renderArtifact();
}

function renderArtifact() {
  const artifact = state.run?.artifacts.find(item => item.step === $('#artifactSelect').value);
  $('#artifactContent').textContent = artifact?.content || '这里将显示每个 Agent 生成的 Markdown 内容。';
}

async function resumeRun(approved) {
  try {
    await api(`/api/runs/${state.run.id}/resume`, {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({approved, feedback:$('#feedbackInput').value})});
    $('#feedbackInput').value = '';
    toast(approved ? '已批准，Agent 继续工作' : '运行已终止');
    await selectRun(state.run.id);
  } catch (error) { toast(error.message, true); }
}

function escapeHtml(value) {
  const node = document.createElement('div'); node.textContent = value; return node.innerHTML;
}

boot();
