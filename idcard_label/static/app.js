const state = {
  config: null,
  counts: null,
  list: [],
  total: 0,
  scanComplete: true,
  page: 1,
  pageSize: 50,
  current: null,
  activeAnnotation: 0,
  selectedAnnotations: new Set(),
  undoStack: [],
  image: null,
  transform: null,
  dragPoint: null,
  dirty: false,
  loadingItem: false,
  correctionNoticeShown: false,
  archiveSelection: new Set(),
};

const CLASS_COLORS = [
  "#20c997", "#ffb000", "#4dabf7", "#ff6b6b",
  "#b197fc", "#f06595", "#82c91e", "#ffa94d",
  "#22b8cf", "#e599f7", "#ffd43b", "#74c0fc",
];
const KEYPOINT_NAMES = ["K0 TL", "K1 TR", "K2 BR", "K3 BL"];
const CANONICAL_SEMANTICS = "idcard_canonical_v1";
const CORNER_HIT_RADIUS = 24;
const DECISION_BUTTON_IDS = [
  "rejectButton", "reviewButton", "reviewedButton", "saveButton", "acceptButton",
];

const elements = Object.fromEntries([
  "appVersion", "progressText", "modelSelect", "statusFilter", "classFilter", "archiveBatchFilter",
  "queue", "previousPage", "nextPage", "pageText", "itemProgress", "modelRefreshButton", "undoButton", "previousItem", "nextItem", "fileName",
  "imageMeta", "canvasStage", "reviewCanvas", "canvasEditor", "canvasSelectionCount", "canvasClassSwatch", "canvasClassEditor",
  "canvasDeleteTarget", "emptyState", "rejectButton", "reviewButton", "reviewedButton", "saveButton",
  "archiveManagerButton", "acceptButton", "addTargetButton", "targetTitle", "targetList", "reviewNote", "saveReviewButton", "currentStatus", "toast",
  "archiveAllCandidatesButton", "archiveAllCandidatesDialog", "archiveAllCandidatesForm", "archiveAllCandidatesClose",
  "archiveAllCandidatesSummary", "candidateArchiveBatchSelect", "newCandidateArchiveBatchField", "newCandidateArchiveBatchName",
  "archiveAllCandidatesCancel", "archiveAllCandidatesSubmit",
  "batchSelect", "newBatchButton", "settingsButton", "settingsDialog", "settingsForm", "settingsCancelButton", "settingsCloseButton",
  "inputRootPath", "outputRootPath", "modelDirectoryPath", "archiveDialog", "archiveMergeForm", "archiveDialogClose",
  "archiveBatchList", "mergedFolderNameField", "mergedFolderName", "mergeDestinationArchiveField", "mergeDestinationArchive",
  "selectAllArchives", "reviewSelectedArchive", "exportSelectedArchive", "archiveSelectionSummary", "mergeArchivesButton",
  "endOfQueueDialog", "endOfQueueClose",
  "actionDialog", "actionDialogForm", "actionDialogKicker", "actionDialogTitle", "actionDialogDescription",
  "actionDialogClose", "actionDialogSelectField", "actionDialogSelectLabel", "actionDialogSelect", "actionDialogSelectHint",
  "actionDialogField", "actionDialogLabel", "actionDialogInput", "actionDialogHint",
  "actionDialogNotice", "actionDialogCancel", "actionDialogSubmit",
].map((id) => [id, document.getElementById(id)]));

const canvas = elements.reviewCanvas;
const context = canvas.getContext("2d");
let queueLoadSequence = 0;
let actionDialogResolver = null;

async function request(url, options = {}) {
  const response = await fetch(url, {
    ...options,
    headers: { "Content-Type": "application/json", ...(options.headers || {}) },
  });
  const responseText = await response.text();
  let payload = null;
  if (responseText) {
    try {
      payload = JSON.parse(responseText);
    } catch (parseError) {
      if (response.ok) throw new Error("Server returned an invalid JSON response");
    }
  }
  if (!response.ok) {
    const message = payload && typeof payload === "object" && payload.error
      ? payload.error
      : `Request failed: ${response.status}${response.statusText ? ` ${response.statusText}` : ""}`;
    const error = new Error(message);
    error.status = response.status;
    if (payload && typeof payload === "object") Object.assign(error, payload);
    throw error;
  }
  return payload;
}

async function initialize() {
  try {
    state.config = await request("/api/config");
    elements.appVersion.textContent = `v${state.config.app_version}`;
    renderModelOptions(state.config.model_refresh);
    bindEvents();
    if (!state.config.configured) {
      openDirectorySettings(true);
      if (state.config.directory_error) showToast(state.config.directory_error, true);
      return;
    }
    for (const value of state.config.classes) {
      const option = document.createElement("option");
      option.value = value.id;
      option.textContent = `${value.id}: ${value.name}`;
      elements.classFilter.append(option);
      const editorOption = option.cloneNode(true);
      elements.canvasClassEditor.append(editorOption);
    }
    await loadQueue(true);
  } catch (error) {
    showToast(error.message, true);
  }
}

function bindEvents() {
  elements.modelSelect.addEventListener("change", switchRefreshModel);
  elements.statusFilter.addEventListener("change", () => {
    renderArchiveFilter();
    renderDecisionActions();
    resetQueue();
  });
  elements.classFilter.addEventListener("change", resetQueue);
  elements.archiveBatchFilter.addEventListener("change", resetQueue);
  elements.previousPage.addEventListener("click", () => changePage(-1));
  elements.nextPage.addEventListener("click", () => changePage(1));
  elements.modelRefreshButton.addEventListener("click", refreshWithModel);
  elements.undoButton.addEventListener("click", undoLastEdit);
  elements.previousItem.addEventListener("click", () => moveItem(-1));
  elements.nextItem.addEventListener("click", () => moveItem(1));
  elements.rejectButton.addEventListener("click", rejectCurrent);
  elements.reviewButton.addEventListener("click", () => saveCurrent("needs_review", true));
  elements.reviewedButton.addEventListener("click", toggleReviewed);
  elements.archiveAllCandidatesButton.addEventListener("click", openArchiveAllCandidates);
  elements.archiveAllCandidatesClose.addEventListener("click", closeArchiveAllCandidates);
  elements.archiveAllCandidatesCancel.addEventListener("click", closeArchiveAllCandidates);
  elements.candidateArchiveBatchSelect.addEventListener("change", updateArchiveAllCandidatesForm);
  elements.newCandidateArchiveBatchName.addEventListener("input", updateArchiveAllCandidatesForm);
  elements.archiveAllCandidatesForm.addEventListener("submit", archiveAllCandidates);
  elements.archiveManagerButton.addEventListener("click", openArchiveManager);
  elements.archiveDialogClose.addEventListener("click", () => elements.archiveDialog.close());
  elements.selectAllArchives.addEventListener("click", selectAllArchiveBatches);
  elements.reviewSelectedArchive.addEventListener("click", reviewSelectedArchive);
  elements.exportSelectedArchive.addEventListener("click", exportSelectedArchive);
  elements.mergedFolderName.addEventListener("input", updateArchiveSelectionSummary);
  for (const input of document.querySelectorAll('input[name="archiveDestinationMode"]')) {
    input.addEventListener("change", updateArchiveSelectionSummary);
  }
  elements.mergeDestinationArchive.addEventListener("change", updateArchiveSelectionSummary);
  elements.archiveMergeForm.addEventListener("submit", mergeSelectedArchives);
  elements.endOfQueueClose.addEventListener("click", () => elements.endOfQueueDialog.close());
  elements.actionDialogForm.addEventListener("submit", submitActionDialog);
  elements.actionDialogCancel.addEventListener("click", () => closeActionDialog(null));
  elements.actionDialogClose.addEventListener("click", () => closeActionDialog(null));
  elements.actionDialog.addEventListener("cancel", (event) => {
    event.preventDefault();
    closeActionDialog(null);
  });
  elements.actionDialog.addEventListener("close", () => resolveActionDialog(null));
  elements.batchSelect.addEventListener("change", activateReviewBatch);
  elements.newBatchButton.addEventListener("click", createReviewBatch);
  elements.settingsButton.addEventListener("click", () => openDirectorySettings(false));
  elements.settingsCancelButton.addEventListener("click", () => elements.settingsDialog.close());
  elements.settingsCloseButton.addEventListener("click", () => elements.settingsDialog.close());
  elements.settingsDialog.addEventListener("cancel", (event) => {
    if (!state.config?.configured) event.preventDefault();
  });
  elements.settingsForm.addEventListener("submit", saveDirectorySettings);
  elements.saveButton.addEventListener("click", saveCandidate);
  elements.acceptButton.addEventListener("click", approveCandidate);
  elements.addTargetButton.addEventListener("click", addTarget);
  elements.canvasClassEditor.addEventListener("change", () => changeActiveClass(Number(elements.canvasClassEditor.value)));
  elements.canvasDeleteTarget.addEventListener("click", deleteSelectedTargets);
  elements.reviewNote.addEventListener("input", markDirty);
  elements.saveReviewButton.addEventListener("click", saveReview);
  canvas.addEventListener("pointerdown", pointerDown);
  canvas.addEventListener("pointermove", pointerMove);
  canvas.addEventListener("pointerup", pointerUp);
  canvas.addEventListener("pointercancel", pointerUp);
  window.addEventListener("resize", resizeCanvas);
  window.addEventListener("keydown", keyboardShortcut);
}

async function resetQueue() {
  state.page = 1;
  await loadQueue(true);
}

async function loadQueue(selectFirst = false) {
  const loadSequence = ++queueLoadSequence;
  if (elements.statusFilter.value === "archive" && !elements.archiveBatchFilter.value) {
    renderArchiveFilter();
  }
  const archiveBatch = elements.statusFilter.value === "archive"
    ? elements.archiveBatchFilter.value
    : "";
  const params = new URLSearchParams({
    status: elements.statusFilter.value,
    class_id: elements.classFilter.value,
    archive_batch: archiveBatch,
    page: state.page,
    page_size: state.pageSize,
  });
  const [result, config] = await Promise.all([
    request(`/api/items?${params}`),
    request("/api/config"),
  ]);
  if (loadSequence !== queueLoadSequence) return;
  const lastPage = Math.max(1, Math.ceil(result.total / state.pageSize));
  const scanComplete = result.scan_complete !== false;
  if (result.total > 0 && state.page > lastPage && (scanComplete || !result.items.length)) {
    state.page = lastPage;
    return loadQueue(selectFirst);
  }
  state.config = config;
  renderModelOptions(config.model_refresh);
  state.list = result.items;
  state.total = result.total;
  state.scanComplete = scanComplete;
  state.counts = result.counts;
  renderQueue(result.counts);
  renderBatchControls();
  renderArchiveFilter();
  if (selectFirst && state.list.length) await selectItem(state.list[0].id);
  if (!state.list.length) clearCurrent();
}

function renderQueue(counts) {
  elements.queue.replaceChildren();
  for (const item of state.list) {
    const button = document.createElement("button");
    const selected = state.current?.id === item.id;
    button.className = `queue-item${selected ? " selected" : ""}`;
    button.innerHTML = `
      <span class="queue-name" title="${escapeHtml(item.file_name)}">${escapeHtml(item.file_name)}</span>
      <span class="queue-meta"><i class="status-dot ${item.status}"></i>#${item.id} · ${item.annotation_count} targets · ${formatScore(item.min_score)}${item.has_correction ? '<b class="correction-mark" title="Previous correction pending; save this archive item to apply directly">↻</b>' : ""}${item.archived ? `<b class="archive-mark" title="${escapeHtml(item.archive_batch || "Archived")}">${escapeHtml(archiveDate(item))}</b>` : ""}${item.reviewed ? '<b class="review-mark" title="Reviewed">✓</b>' : ""}</span>`;
    button.addEventListener("click", () => selectItem(item.id));
    elements.queue.append(button);
  }
  const pages = Math.max(1, Math.ceil(state.total / state.pageSize));
  elements.pageText.textContent = state.scanComplete ? `${state.page} / ${pages}` : `${state.page} / ${pages}+`;
  elements.previousPage.disabled = state.page <= 1;
  elements.nextPage.disabled = state.page >= pages && state.scanComplete;
  updateItemProgress();
  const batch = activeReviewBatch();
  const batchState = batch ? `${batch.ready_count || 0} ready / ${batch.blocked_count || 0} blocked in batch` : "No active batch";
  elements.progressText.textContent = `${batchState} · ${counts.archived} archived · ${counts.pending} pending`;
  const candidateCount = counts.candidates || 0;
  elements.archiveAllCandidatesButton.hidden = elements.statusFilter.value !== "candidate";
  elements.archiveAllCandidatesButton.disabled = candidateCount === 0;
  elements.archiveAllCandidatesButton.textContent = candidateCount
    ? `Archive all ${candidateCount} candidates`
    : "Archive all candidates";
  renderDecisionActions();
}

function visibleDecisionButtons() {
  if (!state.current) return [];
  const filter = elements.statusFilter.value;
  let visible = [];
  if (filter === "pending") visible = ["rejectButton", "reviewButton", "saveButton"];
  if (filter === "candidate") {
    visible = state.current.reviewed
      ? ["rejectButton", "reviewButton", "reviewedButton"]
      : ["rejectButton", "reviewButton", "acceptButton"];
  }
  if (filter === "archive") visible = ["rejectButton", "reviewButton", "saveButton"];
  if (filter === "rejected") visible = ["reviewButton", "saveButton"];
  const currentAction = {
    rejected: "rejectButton",
    needs_review: "reviewButton",
  }[state.current.status];
  return currentAction ? visible.filter((id) => id !== currentAction) : visible;
}

function renderDecisionActions() {
  const visible = new Set(visibleDecisionButtons());
  for (const id of DECISION_BUTTON_IDS) elements[id].hidden = !visible.has(id);
  const requiresTarget = !state.current?.annotations.length;
  for (const id of ["saveButton", "acceptButton", "reviewedButton"]) {
    elements[id].disabled = requiresTarget;
    elements[id].title = requiresTarget
      ? "Add at least one target before entering the candidate or archive workflow"
      : "";
  }
}

function decisionActionAvailable(id) {
  return Boolean(state.current && !elements[id].hidden && !elements[id].disabled);
}

function updateItemProgress() {
  const index = state.current ? state.list.findIndex((item) => item.id === state.current.id) : -1;
  const position = index >= 0 ? ((state.page - 1) * state.pageSize) + index + 1 : 0;
  const totalText = state.scanComplete ? String(state.total) : `${state.total}+`;
  elements.itemProgress.textContent = `${position} / ${totalText}`;
  elements.itemProgress.title = state.scanComplete
    ? `${position} of ${state.total} items in the current queue`
    : `${position} of at least ${state.total} items; more images remain in the input directory`;
}

function renderBatchControls() {
  const activeId = state.config?.active_review_batch;
  elements.batchSelect.replaceChildren();
  for (const batch of state.config?.review_batches || []) {
    const option = document.createElement("option");
    option.value = batch.id;
    option.disabled = batch.status !== "open";
    option.selected = batch.id === activeId;
    const counts = `${batch.ready_count || 0} ready${batch.blocked_count ? `, ${batch.blocked_count} blocked` : ""}`;
    option.textContent = `${batch.name} · ${counts}${batch.status === "archived" ? " · archived" : ""}`;
    elements.batchSelect.append(option);
  }
  elements.batchSelect.disabled = !(state.config?.review_batches || []).some((batch) => batch.status === "open");
}

function activeReviewBatch() {
  return (state.config?.review_batches || []).find((batch) => batch.id === state.config.active_review_batch) || null;
}

function renderArchiveFilter() {
  const previous = elements.archiveBatchFilter.value;
  const batches = state.config?.archive_batches || [];
  elements.archiveBatchFilter.replaceChildren();
  for (const batch of batches) {
    const label = batch.review_batch_name || batch.archive_batch;
    elements.archiveBatchFilter.append(new Option(`${label} · ${batch.item_count} items`, batch.archive_batch));
  }
  const exists = [...elements.archiveBatchFilter.options].some((option) => option.value === previous);
  elements.archiveBatchFilter.value = exists ? previous : (batches[0]?.archive_batch || "");
  elements.archiveBatchFilter.disabled = batches.length === 0;
  elements.archiveBatchFilter.hidden = elements.statusFilter.value !== "archive";
}

function openArchiveManager() {
  state.archiveSelection = new Set();
  document.querySelector('input[name="archiveDestinationMode"][value="new"]').checked = true;
  elements.mergedFolderName.value = `merged-${localVersionStamp()}`;
  renderArchiveManager();
  elements.archiveDialog.showModal();
}

function renderArchiveManager() {
  elements.archiveBatchList.replaceChildren();
  const batches = state.config?.archive_batches || [];
  if (!batches.length) {
    const empty = document.createElement("div");
    empty.className = "archive-empty";
    empty.textContent = "No archived review batches are available.";
    elements.archiveBatchList.append(empty);
  }
  for (const batch of batches) {
    const row = document.createElement("label");
    row.className = "archive-batch-row";
    const checked = state.archiveSelection.has(batch.archive_batch);
    const label = batch.review_batch_name || batch.archive_batch;
    const path = label === batch.archive_batch ? "" : `<small>${escapeHtml(batch.archive_batch)}</small>`;
    row.innerHTML = `
      <input type="checkbox" value="${escapeHtml(batch.archive_batch)}"${checked ? " checked" : ""}>
      <span class="archive-batch-main">
        <strong>${escapeHtml(label)}</strong>
        ${path}
      </span>
      <span class="archive-batch-stats">${batch.accepted_count} accepted${batch.excluded_count ? ` · ${batch.excluded_count} excluded` : ""}${batch.correction_revisions ? ` · ${batch.correction_revisions} corrections` : ""}</span>`;
    row.querySelector("input").addEventListener("change", (event) => {
      if (event.target.checked) state.archiveSelection.add(batch.archive_batch);
      else state.archiveSelection.delete(batch.archive_batch);
      updateArchiveSelectionSummary();
    });
    elements.archiveBatchList.append(row);
  }
  updateArchiveSelectionSummary();
}

function updateArchiveSelectionSummary() {
  const batches = state.config?.archive_batches || [];
  const selected = batches.filter((batch) => state.archiveSelection.has(batch.archive_batch));
  const destinationBatches = batches.filter((batch) => !state.archiveSelection.has(batch.archive_batch));
  const previousDestination = elements.mergeDestinationArchive.value;
  elements.mergeDestinationArchive.replaceChildren();
  for (const batch of destinationBatches) {
    const option = document.createElement("option");
    option.value = batch.archive_batch;
    option.textContent = batch.review_batch_name
      ? `${batch.review_batch_name} - ${batch.archive_batch}`
      : batch.archive_batch;
    elements.mergeDestinationArchive.append(option);
  }
  if (destinationBatches.some((batch) => batch.archive_batch === previousDestination)) {
    elements.mergeDestinationArchive.value = previousDestination;
  }
  elements.mergeDestinationArchive.disabled = destinationBatches.length === 0;

  const destinationMode = archiveDestinationMode();
  elements.mergedFolderNameField.hidden = destinationMode !== "new";
  elements.mergeDestinationArchiveField.hidden = destinationMode !== "existing";
  elements.mergedFolderName.required = destinationMode === "new";
  elements.mergeDestinationArchive.required = destinationMode === "existing";
  elements.archiveSelectionSummary.textContent = `${selected.length} selected`;
  elements.reviewSelectedArchive.disabled = selected.length !== 1;
  elements.exportSelectedArchive.disabled = selected.length !== 1;
  const destinationArchive = elements.mergeDestinationArchive.value;
  const mergeBatchCount = destinationMode === "existing" ? selected.length + 1 : selected.length;
  const validDestination = destinationMode === "existing"
    ? Boolean(destinationArchive) && !state.archiveSelection.has(destinationArchive)
    : validMergedFolderName();
  elements.mergeArchivesButton.disabled = mergeBatchCount < 2 || !validDestination;
  elements.selectAllArchives.textContent = selected.length === batches.length && batches.length ? "Clear selection" : "Select all";
}

function archiveDestinationMode() {
  return document.querySelector('input[name="archiveDestinationMode"]:checked')?.value || "new";
}

function validMergedFolderName() {
  const value = elements.mergedFolderName.value.trim();
  return Boolean(value) && value !== "." && value !== ".." && !/[\\/]/.test(value);
}

function selectAllArchiveBatches() {
  const batches = state.config?.archive_batches || [];
  if (state.archiveSelection.size === batches.length) state.archiveSelection.clear();
  else state.archiveSelection = new Set(batches.map((batch) => batch.archive_batch));
  renderArchiveManager();
}

async function reviewSelectedArchive() {
  if (state.archiveSelection.size !== 1) return;
  const [archiveBatch] = state.archiveSelection;
  elements.archiveDialog.close();
  elements.statusFilter.value = "archive";
  renderArchiveFilter();
  elements.archiveBatchFilter.value = archiveBatch;
  await resetQueue();
}

async function exportSelectedArchive() {
  if (state.archiveSelection.size !== 1) return;
  const [archiveBatch] = state.archiveSelection;
  const exportSettings = await openActionDialog({
    kicker: "Export archive",
    title: "Export a training dataset",
    description: "Creates or updates a ready-to-train dataset from the selected archive.",
    selectLabel: "Export format",
    selectValue: state.config?.export_format || "yolo_pose",
    selectOptions: [
      { value: "yolo_pose", label: "YOLO Pose" },
      { value: "coco_keypoints", label: "COCO Keypoints" },
    ],
    selectHint: "YOLO exports data.yaml and text labels. COCO exports keypoint JSON annotations.",
    label: "Destination directory",
    value: state.config?.export_root || "",
    placeholder: "/path/to/export",
    hint: "Enter an existing directory or a path that can be created.",
    confirmLabel: "Export archive",
  });
  if (!exportSettings) return;
  const { value: destinationRoot, selected: exportFormat } = exportSettings;
  const originalText = elements.exportSelectedArchive.textContent;
  elements.exportSelectedArchive.disabled = true;
  elements.exportSelectedArchive.textContent = "Exporting...";
  try {
    const result = await request("/api/archives/export", {
      method: "POST",
      body: JSON.stringify({
        archive_batch: archiveBatch,
        destination_root: destinationRoot.trim(),
        export_format: exportFormat,
      }),
    });
    state.config.export_root = result.export_root;
    state.config.export_format = result.export_format;
    const action = result.replaced ? "Updated" : "Exported";
    showToast(`${action} ${result.images} ${result.format_label} images at ${result.path}`);
  } catch (error) {
    showToast(error.message, true);
  } finally {
    elements.exportSelectedArchive.textContent = originalText;
    updateArchiveSelectionSummary();
  }
}

async function mergeSelectedArchives(event) {
  event.preventDefault();
  const destinationMode = archiveDestinationMode();
  const destinationArchive = destinationMode === "existing" ? elements.mergeDestinationArchive.value : "";
  const payload = { archive_batches: [...state.archiveSelection] };
  if (destinationMode === "new") {
    if (state.archiveSelection.size < 2 || !validMergedFolderName()) return;
    payload.folder_name = elements.mergedFolderName.value.trim();
  } else {
    if (!destinationArchive || state.archiveSelection.has(destinationArchive) || state.archiveSelection.size < 1) return;
    payload.destination_archive_batch = destinationArchive;
  }
  const originalText = elements.mergeArchivesButton.textContent;
  elements.mergeArchivesButton.disabled = true;
  elements.mergeArchivesButton.textContent = "Merging...";
  try {
    const result = await request("/api/archives/merge", {
      method: "POST",
      body: JSON.stringify(payload),
    });
    state.config = await request("/api/config");
    renderArchiveFilter();
    elements.archiveDialog.close();
    showToast(`Merged ${result.source_batches.length} archives into ${result.archive_batch}`);
  } catch (error) {
    showToast(error.message, true);
  } finally {
    elements.mergeArchivesButton.textContent = originalText;
    updateArchiveSelectionSummary();
  }
}

function localVersionStamp() {
  const now = new Date();
  const value = (number) => String(number).padStart(2, "0");
  return `${now.getFullYear()}${value(now.getMonth() + 1)}${value(now.getDate())}-${value(now.getHours())}${value(now.getMinutes())}${value(now.getSeconds())}`;
}

async function selectItem(id) {
  if (state.loadingItem || state.current?.id === id) return;
  state.loadingItem = true;
  try {
    if (state.current && state.dirty) await persistCurrent(state.current.status);
    const item = await request(`/api/items/${id}`);
    state.current = item;
    state.activeAnnotation = Math.min(state.activeAnnotation, Math.max(0, item.annotations.length - 1));
    state.selectedAnnotations = item.annotations.length ? new Set([state.activeAnnotation]) : new Set();
    state.undoStack = [];
    state.correctionNoticeShown = false;
    elements.undoButton.disabled = true;
    elements.modelRefreshButton.disabled = false;
    state.dirty = false;
    elements.fileName.textContent = item.file_name;
    elements.fileName.title = item.image_path;
    elements.imageMeta.textContent = `${item.width} × ${item.height} · #${item.id}`;
    elements.reviewNote.value = item.note || "";
    updateReviewControls();
    elements.emptyState.hidden = true;
    renderTargets();
    await loadImage(item.id);
    renderQueue(state.counts || { accepted: 0, pending: 0, needs_review: 0, rejected: 0 });
  } catch (error) {
    showToast(error.message, true);
  } finally {
    state.loadingItem = false;
  }
}

function loadImage(id) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => {
      state.image = image;
      resizeCanvas();
      resolve();
    };
    image.onerror = () => reject(new Error("Could not decode the source image"));
    image.src = `/api/images/${id}?v=${state.current.version}`;
  });
}

function resizeCanvas() {
  const width = Math.max(1, elements.canvasStage.clientWidth);
  const height = Math.max(1, elements.canvasStage.clientHeight);
  const ratio = window.devicePixelRatio || 1;
  canvas.style.width = `${width}px`;
  canvas.style.height = `${height}px`;
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  draw(width, height);
}

function draw(width = elements.canvasStage.clientWidth, height = elements.canvasStage.clientHeight) {
  context.clearRect(0, 0, width, height);
  if (!state.current || !state.image) return;
  const imageWidth = state.image.naturalWidth || state.current.width;
  const imageHeight = state.image.naturalHeight || state.current.height;
  const scale = Math.min(width / imageWidth, height / imageHeight);
  const drawWidth = imageWidth * scale;
  const drawHeight = imageHeight * scale;
  const offsetX = (width - drawWidth) / 2;
  const offsetY = (height - drawHeight) / 2;
  state.transform = {
    scaleX: drawWidth / state.current.width,
    scaleY: drawHeight / state.current.height,
    offsetX,
    offsetY,
  };
  context.drawImage(state.image, offsetX, offsetY, drawWidth, drawHeight);

  state.current.annotations.forEach((annotation, index) => {
    const active = index === state.activeAnnotation;
    const selected = state.selectedAnnotations.has(index);
    const color = classColor(annotation.class_id);
    const points = annotation.polygon.map(toCanvasPoint);
    context.beginPath();
    context.moveTo(points[0].x, points[0].y);
    points.slice(1).forEach((point) => context.lineTo(point.x, point.y));
    context.closePath();
    context.lineWidth = selected ? (active ? 4 : 3) : 2;
    context.strokeStyle = color;
    context.stroke();
    points.forEach((point, pointIndex) => {
      const isOrigin = pointIndex === 0;
      context.beginPath();
      context.arc(point.x, point.y, isOrigin ? 10 : active ? 6 : selected ? 5 : 4, 0, Math.PI * 2);
      const visibility = annotation.keypoint_visibility?.[pointIndex] ?? 2;
      context.fillStyle = isOrigin ? "#ffe066" : visibility === 2 ? "#fff" : visibility === 1 ? "#ffd43b" : "#7d8580";
      context.fill();
      context.lineWidth = isOrigin ? 4 : 2;
      context.strokeStyle = isOrigin ? "#111" : color;
      context.stroke();
      if (isOrigin) {
        context.beginPath();
        context.arc(point.x, point.y, 14, 0, Math.PI * 2);
        context.lineWidth = 2;
        context.strokeStyle = "#ffe066";
        context.stroke();
      }
      const pointLabel = isOrigin ? "K0 START" : `K${pointIndex}`;
      context.font = isOrigin ? "800 12px ui-monospace, monospace" : "700 11px ui-monospace, monospace";
      const pointLabelWidth = context.measureText(pointLabel).width + (isOrigin ? 10 : 0);
      if (isOrigin) {
        context.fillStyle = "#ffe066";
        context.fillRect(point.x + 14, point.y - 20, pointLabelWidth, 18);
        context.fillStyle = "#111";
        context.fillText(pointLabel, point.x + 19, point.y - 7);
      } else {
        context.fillStyle = "#fff";
        context.fillText(pointLabel, point.x + 8, point.y - 8);
      }
    });
    const label = `${index + 1} · ${annotation.class_name}`;
    context.font = "600 12px -apple-system, BlinkMacSystemFont, sans-serif";
    const labelWidth = context.measureText(label).width + 12;
    const minY = Math.min(...points.map((point) => point.y));
    const minX = Math.min(...points.map((point) => point.x));
    const maxX = Math.max(...points.map((point) => point.x));
    const labelX = Math.max(0, Math.min(width - labelWidth, (minX + maxX - labelWidth) / 2));
    const labelY = minY >= 28 ? minY - 25 : minY + 10;
    context.fillStyle = color;
    context.fillRect(labelX, labelY, labelWidth, 21);
    context.fillStyle = classTextColor(annotation.class_id);
    context.fillText(label, labelX + 6, labelY + 15);
  });
  positionCanvasEditor();
}

function renderTargets() {
  elements.targetList.replaceChildren();
  if (!state.current) return;
  renderDecisionActions();
  const selectedCount = state.selectedAnnotations.size;
  elements.targetTitle.textContent = selectedCount ? `Targets · ${selectedCount} selected` : "Targets";
  state.current.annotations.forEach((annotation, index) => {
    const target = document.createElement("div");
    target.className = `target${state.selectedAnnotations.has(index) ? " selected" : ""}${index === state.activeAnnotation ? " active" : ""}`;
    target.innerHTML = `
      <div class="target-head">
        <span class="target-index" style="background:${classColor(annotation.class_id)}">${index + 1}</span>
        <select aria-label="Target ${index + 1} class">${classOptions(annotation.class_id)}</select>
        <button class="delete-target" title="Delete target" aria-label="Delete target">×</button>
      </div>
      <span class="target-score">Teacher score: ${annotation.score == null ? "manual" : annotation.score.toFixed(4)}</span>
      <div class="corner-toolbar">
        <button type="button" class="corner-rotate" data-direction="-1" title="Shift K0-K3 labels one corner counterclockwise (Shift+T)">K &#8630;</button>
        <button type="button" class="corner-rotate" data-direction="1" title="Shift K0-K3 labels one corner clockwise (T)">K &#8631; <kbd>T</kbd></button>
        <button type="button" class="corner-reverse" title="Reverse K0-K3 winding while keeping K0 fixed">&#8644;</button>
      </div>
      <div class="keypoint-list">
        ${(annotation.polygon || []).map((_, pointIndex) => `
          <label class="${pointIndex === 0 ? "keypoint-origin" : ""}"><span>${pointIndex === 0 ? "K0 START · TL" : KEYPOINT_NAMES[pointIndex]}</span>
            <select data-point-index="${pointIndex}" aria-label="${KEYPOINT_NAMES[pointIndex]} visibility">
              <option value="2" ${(annotation.keypoint_visibility?.[pointIndex] ?? 2) === 2 ? "selected" : ""}>Visible</option>
              <option value="1" ${annotation.keypoint_visibility?.[pointIndex] === 1 ? "selected" : ""}>Occluded</option>
              <option value="0" ${annotation.keypoint_visibility?.[pointIndex] === 0 ? "selected" : ""}>Unlabeled</option>
            </select>
          </label>`).join("")}
      </div>`;
    target.addEventListener("click", (event) => {
      if (event.target.closest("button, input, select, label")) return;
      selectAnnotation(index, event.shiftKey);
    });
    target.querySelector("select").addEventListener("change", (event) => {
      event.stopPropagation();
      selectAnnotation(index, false);
      changeActiveClass(Number(event.target.value));
    });
    target.querySelector(".delete-target").addEventListener("click", (event) => {
      event.stopPropagation();
      selectAnnotation(index, event.shiftKey);
      deleteSelectedTargets();
    });
    target.querySelectorAll(".corner-rotate").forEach((button) => button.addEventListener("click", (event) => {
      event.stopPropagation();
      rotateKeypoints(index, Number(button.dataset.direction));
    }));
    target.querySelector(".corner-reverse").addEventListener("click", (event) => {
      event.stopPropagation();
      reverseKeypoints(index);
    });
    target.querySelectorAll(".keypoint-list select").forEach((select) => select.addEventListener("change", (event) => {
      event.stopPropagation();
      pushUndo();
      const pointIndex = Number(select.dataset.pointIndex);
      annotation.keypoint_visibility = annotation.keypoint_visibility || [2, 2, 2, 2];
      annotation.keypoint_visibility[pointIndex] = Number(select.value);
      markDirty();
      draw();
    }));
    elements.targetList.append(target);
  });
}

function addTarget() {
  if (!state.current || !state.config.classes.length) return;
  pushUndo();
  const { width, height } = state.current;
  const selectedClass = state.config.classes.find(
    (value) => value.id === state.config.default_class_id
  ) || state.config.classes[0];
  state.current.annotations.push({
    class_id: selectedClass.id,
    class_name: selectedClass.name,
    score: null,
    polygon: [
      [width * 0.25, height * 0.3], [width * 0.75, height * 0.3],
      [width * 0.75, height * 0.7], [width * 0.25, height * 0.7],
    ],
    keypoint_visibility: [2, 2, 2, 2],
    keypoint_semantics: CANONICAL_SEMANTICS,
  });
  state.activeAnnotation = state.current.annotations.length - 1;
  state.selectedAnnotations = new Set([state.activeAnnotation]);
  markDirty();
  renderTargets();
  draw();
}

function pointerDown(event) {
  if (!state.current || !state.transform) return;
  const pointer = canvasPointer(event);
  let nearest = null;
  state.current.annotations.forEach((annotation, annotationIndex) => {
    annotation.polygon.forEach((point, pointIndex) => {
      const canvasPoint = toCanvasPoint(point);
      const distance = Math.hypot(canvasPoint.x - pointer.x, canvasPoint.y - pointer.y);
      if (distance <= CORNER_HIT_RADIUS && (!nearest || distance < nearest.distance)) {
        nearest = { annotationIndex, pointIndex, distance };
      }
    });
  });
  if (!nearest) {
    for (let index = state.current.annotations.length - 1; index >= 0; index -= 1) {
      const points = state.current.annotations[index].polygon.map(toCanvasPoint);
      if (pointInPolygon(pointer, points)) {
        selectAnnotation(index, event.shiftKey);
        return;
      }
    }
    if (!event.shiftKey) clearAnnotationSelection();
    return;
  }
  if (event.shiftKey) {
    selectAnnotation(nearest.annotationIndex, true);
    return;
  }
  if (!state.selectedAnnotations.has(nearest.annotationIndex)) {
    selectAnnotation(nearest.annotationIndex, false);
  } else {
    state.activeAnnotation = nearest.annotationIndex;
  }
  pushUndo();
  state.dragPoint = nearest;
  canvas.setPointerCapture(event.pointerId);
  renderTargets();
  draw();
}

function pointerMove(event) {
  if (!state.dragPoint || !state.current) return;
  const pointer = canvasPointer(event);
  const { scaleX, scaleY, offsetX, offsetY } = state.transform;
  const x = Math.max(0, Math.min(state.current.width, (pointer.x - offsetX) / scaleX));
  const y = Math.max(0, Math.min(state.current.height, (pointer.y - offsetY) / scaleY));
  state.current.annotations[state.dragPoint.annotationIndex].polygon[state.dragPoint.pointIndex] = [x, y];
  markDirty();
  draw();
}

function pointerUp(event) {
  if (state.dragPoint && canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId);
  state.dragPoint = null;
}

async function saveCurrent(status, advance, reviewedOverride = null, reviewApproved = false) {
  if (!state.current) return;
  const previousIndex = state.list.findIndex((item) => item.id === state.current.id);
  try {
    const updated = await persistCurrent(status, reviewedOverride, reviewApproved);
    const message = state.current.archived ? "Archive updated" : ({
      accepted: updated.reviewed ? "Candidate approved for archive" : "Added to candidate review",
      rejected: "Image rejected",
      needs_review: "Sent to follow-up review",
    }[status] || "Review saved");
    if (status === "accepted" && !updated.pose_ready) {
      showToast(`Review saved, but not ready for archive approval: ${updated.pose_error}`, true);
      await loadQueue(false);
      return;
    }
    showToast(correctionSaveMessage(updated) || message);
    await loadQueue(false);
    if (advance) {
      if (state.list.length) {
        const currentIndex = state.list.findIndex((item) => item.id === state.current.id);
        const nextIndex = currentIndex >= 0
          ? (currentIndex + 1) % state.list.length
          : Math.min(Math.max(0, previousIndex), state.list.length - 1);
        const next = state.list[nextIndex];
        if (next && next.id !== state.current.id) await selectItem(next.id);
      } else showEndOfQueue();
    }
  } catch (error) {
    showToast(error.message, true);
  }
}

async function persistCurrent(status, reviewedOverride = null, reviewApproved = false) {
  const reviewed = reviewedOverride == null ? Boolean(state.current.reviewed) : Boolean(reviewedOverride);
  const payload = {
    version: state.current.version,
    status,
    reviewed,
    review_approved: reviewApproved,
    note: elements.reviewNote.value,
    annotations: state.current.annotations.map((annotation) => ({
      class_id: annotation.class_id,
      class_name: annotation.class_name,
      score: annotation.score,
      source_index: annotation.source_index,
      polygon: annotation.polygon,
      keypoint_visibility: annotation.keypoint_visibility || [2, 2, 2, 2],
      keypoint_semantics: CANONICAL_SEMANTICS,
    })),
  };
  const updated = await request(`/api/items/${state.current.id}`, { method: "PUT", body: JSON.stringify(payload) });
  state.current = updated;
  state.dirty = false;
  updateReviewControls();
  return updated;
}

async function saveReview() {
  if (!state.current) return;
  try {
    const updated = await persistCurrent(state.current.status);
    showToast(updated.archived ? "Archive updated" : (correctionSaveMessage(updated) || "Review content saved"));
    await loadQueue(false);
  } catch (error) {
    showToast(error.message, true);
  }
}

async function saveCandidate() {
  if (!state.current) return;
  try {
    const updated = await persistCurrent("accepted", false);
    correctionSaveMessage(updated);
    if (!updated.pose_ready) {
      await persistCurrent("needs_review", false);
      showToast(`Saved review, but cannot become a candidate: ${updated.pose_error}`, true);
      await loadQueue(false);
      return;
    }
    showToast(updated.archived ? "Archive updated" : "Added to candidate review");
    await loadQueue(false);
    await advanceAfterSave();
  } catch (error) {
    showToast(error.message, true);
  }
}

async function approveCandidate() {
  if (!state.current) return;
  await saveCurrent("accepted", true, true, true);
}

async function rejectCurrent() {
  await saveCurrent("rejected", true, true);
}

async function advanceAfterSave() {
  if (!state.list.length) return showEndOfQueue();
  const currentIndex = state.list.findIndex((item) => item.id === state.current?.id);
  const next = state.list[Math.max(0, currentIndex)] || state.list[0];
  if (next && next.id !== state.current?.id) await selectItem(next.id);
  else if (next && state.list.length > 1) await selectItem(state.list[(currentIndex + 1) % state.list.length].id);
  else showEndOfQueue();
}

async function changePage(offset) {
  if (state.current && state.dirty) await persistCurrent(state.current.status);
  state.page += offset;
  await loadQueue(true);
}

async function moveItem(offset) {
  if (!state.current || !state.list.length) return;
  const index = state.list.findIndex((item) => item.id === state.current.id);
  const next = state.list[index + offset];
  if (next) {
    await selectItem(next.id);
    return;
  }
  const pages = Math.max(1, Math.ceil(state.total / state.pageSize));
  if (offset > 0 && (state.page < pages || !state.scanComplete)) {
    if (state.dirty) await persistCurrent(state.current.status);
    const previousPage = state.page;
    state.page += 1;
    await loadQueue(false);
    if (state.page > previousPage && state.list.length) await selectItem(state.list[0].id);
    else showEndOfQueue(false);
  } else if (offset < 0 && state.page > 1) {
    if (state.dirty) await persistCurrent(state.current.status);
    state.page -= 1;
    await loadQueue(false);
    if (state.list.length) await selectItem(state.list[state.list.length - 1].id);
  } else if (offset > 0) {
    showEndOfQueue(false);
  }
}

function keyboardShortcut(event) {
  if (document.querySelector("dialog[open]")) return;
  const tag = event.target.tagName;
  const commandKey = event.ctrlKey || event.metaKey;
  if (commandKey && event.key.toLowerCase() === "s") {
    event.preventDefault();
    saveReview();
    return;
  }
  if (commandKey && event.key.toLowerCase() === "z" && tag !== "INPUT" && tag !== "TEXTAREA") {
    event.preventDefault();
    undoLastEdit();
    return;
  }
  if (commandKey && event.key.toLowerCase() === "a" && tag !== "INPUT" && tag !== "TEXTAREA") {
    event.preventDefault();
    selectAllAnnotations();
    return;
  }
  if (commandKey && event.key === "Enter") {
    if (decisionActionAvailable("acceptButton")) {
      event.preventDefault();
      approveCandidate();
    }
    return;
  }
  if (commandKey) return;
  if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
  const key = event.key.toLowerCase();
  if (key === "arrowright" || key === "j") { event.preventDefault(); moveItem(1); }
  if (key === "arrowleft" || key === "k") { event.preventDefault(); moveItem(-1); }
  if (key === "g" && decisionActionAvailable("saveButton")) { event.preventDefault(); saveCandidate(); }
  if (key === "r" && decisionActionAvailable("rejectButton")) { event.preventDefault(); rejectCurrent(); }
  if (key === "q" && decisionActionAvailable("reviewButton")) { event.preventDefault(); saveCurrent("needs_review", true); }
  if (key === "a" && decisionActionAvailable("acceptButton")) {
    event.preventDefault();
    approveCandidate();
  }
  if (key === "e" && decisionActionAvailable("reviewedButton")) { event.preventDefault(); toggleReviewed(); }
  if (key === "f") { event.preventDefault(); refreshWithModel(); }
  if (key === "t") {
    event.preventDefault();
    rotateKeypoints(state.activeAnnotation, event.shiftKey ? -1 : 1);
  }
  if (key === "n") { event.preventDefault(); addTarget(); }
  if (key === "x" || key === "delete" || key === "backspace") {
    event.preventDefault();
    deleteSelectedTargets();
  }
  if (/^[0-9]$/.test(key)) { event.preventDefault(); changeActiveClass(Number(key)); }
}

function selectAnnotation(index, additive = false) {
  if (!state.current?.annotations[index]) return;
  if (additive) {
    if (state.selectedAnnotations.has(index)) {
      state.selectedAnnotations.delete(index);
      if (state.activeAnnotation === index) {
        state.activeAnnotation = state.selectedAnnotations.values().next().value ?? 0;
      }
    } else {
      state.selectedAnnotations.add(index);
      state.activeAnnotation = index;
    }
  } else {
    state.activeAnnotation = index;
    state.selectedAnnotations = new Set([index]);
  }
  renderTargets();
  draw();
}

function changeActiveClass(classId) {
  if (!state.current) return;
  const selectedClass = state.config.classes.find((value) => value.id === classId);
  const indexes = selectedAnnotationIndexes();
  if (!indexes.length || !selectedClass) return;
  pushUndo();
  for (const index of indexes) {
    state.current.annotations[index].class_id = classId;
    state.current.annotations[index].class_name = selectedClass.name;
  }
  markDirty();
  renderTargets();
  draw();
}

function deleteSelectedTargets() {
  if (!state.current || !state.current.annotations.length) return;
  const indexes = selectedAnnotationIndexes();
  if (!indexes.length) return;
  pushUndo();
  const nextIndex = Math.min(...indexes);
  for (const index of indexes.sort((a, b) => b - a)) state.current.annotations.splice(index, 1);
  state.activeAnnotation = Math.max(0, Math.min(nextIndex, state.current.annotations.length - 1));
  state.selectedAnnotations = state.current.annotations.length ? new Set([state.activeAnnotation]) : new Set();
  markDirty();
  renderTargets();
  draw();
}

function rotateKeypoints(index, direction) {
  const annotation = state.current?.annotations[index];
  if (!annotation) return;
  pushUndo();
  const shift = direction > 0 ? 3 : 1;
  annotation.polygon = annotation.polygon.map((_, pointIndex, points) => points[(pointIndex + shift) % 4]);
  const visibility = annotation.keypoint_visibility || [2, 2, 2, 2];
  annotation.keypoint_visibility = visibility.map((_, pointIndex, values) => values[(pointIndex + shift) % 4]);
  state.activeAnnotation = index;
  state.selectedAnnotations = new Set([index]);
  markDirty();
  renderTargets();
  draw();
}

function reverseKeypoints(index) {
  const annotation = state.current?.annotations[index];
  if (!annotation) return;
  pushUndo();
  annotation.polygon = [annotation.polygon[0], annotation.polygon[3], annotation.polygon[2], annotation.polygon[1]];
  const visibility = annotation.keypoint_visibility || [2, 2, 2, 2];
  annotation.keypoint_visibility = [visibility[0], visibility[3], visibility[2], visibility[1]];
  state.activeAnnotation = index;
  state.selectedAnnotations = new Set([index]);
  markDirty();
  renderTargets();
  draw();
}

function clearAnnotationSelection() {
  state.selectedAnnotations.clear();
  renderTargets();
  draw();
}

function selectAllAnnotations() {
  if (!state.current?.annotations.length) return;
  state.selectedAnnotations = new Set(state.current.annotations.map((_, index) => index));
  state.activeAnnotation = 0;
  renderTargets();
  draw();
}

function selectedAnnotationIndexes() {
  return [...state.selectedAnnotations].filter((index) => state.current?.annotations[index]);
}

function pushUndo() {
  if (!state.current) return;
  state.undoStack.push({
    annotations: structuredCloneSafe(state.current.annotations),
    activeAnnotation: state.activeAnnotation,
    selectedAnnotations: [...state.selectedAnnotations],
  });
  if (state.undoStack.length > 50) state.undoStack.shift();
  elements.undoButton.disabled = false;
}

function undoLastEdit() {
  if (!state.current || !state.undoStack.length) return;
  const snapshot = state.undoStack.pop();
  state.current.annotations = snapshot.annotations;
  state.activeAnnotation = snapshot.activeAnnotation;
  state.selectedAnnotations = new Set(snapshot.selectedAnnotations);
  markDirty();
  renderTargets();
  draw();
  elements.undoButton.disabled = state.undoStack.length === 0;
  showToast("Last annotation edit undone");
}

async function toggleReviewed() {
  if (!state.current) return;
  await saveCurrent(state.current.status, true, !state.current.reviewed, !state.current.reviewed);
}

async function refreshWithModel() {
  if (!state.current || elements.modelRefreshButton.disabled) return;
  const itemId = state.current.id;
  elements.modelRefreshButton.disabled = true;
  const originalText = elements.modelRefreshButton.textContent;
  elements.modelRefreshButton.textContent = "...";
  try {
    const result = await request(`/api/items/${itemId}/refresh`, { method: "POST", body: "{}" });
    if (state.current?.id !== itemId) return;
    pushUndo();
    state.current.annotations = result.annotations;
    state.activeAnnotation = 0;
    state.selectedAnnotations = result.annotations.length ? new Set([0]) : new Set();
    markDirty();
    renderTargets();
    draw();
    showToast(`Refreshed ${result.detections} target${result.detections === 1 ? "" : "s"} with ${result.model}`);
  } catch (error) {
    showToast(error.message, true);
  } finally {
    elements.modelRefreshButton.textContent = originalText;
    elements.modelRefreshButton.disabled = false;
  }
}

function renderModelOptions(modelRefresh) {
  elements.modelSelect.replaceChildren();
  if (!modelRefresh) {
    elements.modelSelect.hidden = true;
    return;
  }
  elements.modelSelect.hidden = false;
  for (const model of modelRefresh.models) {
    const option = document.createElement("option");
    option.value = model.name;
    option.textContent = model.modified_ns === null ? `${model.name} (missing)` : model.name;
    option.selected = model.name === modelRefresh.selected;
    elements.modelSelect.append(option);
  }
  elements.modelSelect.title = `Local refresh models: ${modelRefresh.directory}`;
}

async function switchRefreshModel() {
  const previous = state.config.model_refresh?.selected;
  elements.modelSelect.disabled = true;
  try {
    const modelRefresh = await request("/api/models", {
      method: "PUT",
      body: JSON.stringify({ model: elements.modelSelect.value }),
    });
    state.config.model_refresh = modelRefresh;
    renderModelOptions(modelRefresh);
    showToast(`Refresh model switched to ${modelRefresh.selected}`);
  } catch (error) {
    if (previous) elements.modelSelect.value = previous;
    showToast(error.message, true);
  } finally {
    elements.modelSelect.disabled = false;
  }
}

function openArchiveAllCandidates() {
  const archiveBatches = state.config?.archive_batches || [];
  elements.candidateArchiveBatchSelect.replaceChildren();
  elements.candidateArchiveBatchSelect.append(new Option("Create a new archive (default)", ""));
  for (const batch of archiveBatches) {
    const label = batch.review_batch_name || batch.archive_batch;
    elements.candidateArchiveBatchSelect.append(new Option(
      `${label} · ${batch.item_count} items · ${batch.archive_batch}`,
      batch.archive_batch,
    ));
  }
  elements.candidateArchiveBatchSelect.append(new Option("Create a new archive folder...", "__new__"));
  elements.candidateArchiveBatchSelect.value = "";
  elements.newCandidateArchiveBatchName.value = `Archive ${localVersionStamp()}`;
  const candidateCount = state.counts?.candidates || state.config?.counts?.candidates || 0;
  elements.archiveAllCandidatesSummary.textContent = `${candidateCount} candidates will be approved and written to one archive.`;
  updateArchiveAllCandidatesForm();
  elements.archiveAllCandidatesDialog.showModal();
}

function closeArchiveAllCandidates() {
  elements.archiveAllCandidatesDialog.close();
}

function updateArchiveAllCandidatesForm() {
  const creating = elements.candidateArchiveBatchSelect.value === "__new__";
  elements.newCandidateArchiveBatchField.hidden = !creating;
  elements.newCandidateArchiveBatchName.required = creating;
  elements.archiveAllCandidatesSubmit.disabled = creating && !elements.newCandidateArchiveBatchName.value.trim();
}

async function archiveAllCandidates(event) {
  event.preventDefault();
  const candidateCount = state.counts?.candidates || state.config?.counts?.candidates || 0;
  if (!candidateCount) return;
  const creating = elements.candidateArchiveBatchSelect.value === "__new__";
  const folderName = elements.newCandidateArchiveBatchName.value.trim();
  if (creating && !folderName) return;
  const originalText = elements.archiveAllCandidatesSubmit.textContent;
  elements.archiveAllCandidatesSubmit.disabled = true;
  elements.archiveAllCandidatesSubmit.textContent = "Archiving...";
  try {
    if (state.current && state.dirty) await persistCurrent(state.current.status);
    let reviewBatch;
    if (creating) {
      const batch = await request("/api/review-batches", {
        method: "POST",
        body: JSON.stringify({ name: folderName }),
      });
      reviewBatch = batch.id;
    }
    const result = await request("/api/archive", {
      method: "POST",
      body: JSON.stringify({
        review_batch: reviewBatch,
        all_candidates: true,
        merge_into_archive: !creating && elements.candidateArchiveBatchSelect.value || undefined,
      }),
    });
    elements.archiveAllCandidatesDialog.close();
    showToast(result.merged_into
      ? `${result.archived} candidates merged into ${result.archive_batch}`
      : `${result.archived} candidates archived to ${result.archive_batch}`);
    await loadQueue(true);
  } catch (error) {
    showToast(error.message, true);
  } finally {
    elements.archiveAllCandidatesSubmit.textContent = originalText;
    updateArchiveAllCandidatesForm();
  }
}

async function createReviewBatch() {
  const suggested = `Review ${new Date().toLocaleString([], { year: "numeric", month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit" })}`;
  const name = await openActionDialog({
    kicker: "Review workflow",
    title: "Create a review batch",
    description: "New accepted items will be isolated in this batch.",
    label: "Batch name",
    value: suggested,
    hint: "Use a name that identifies the review scope or session.",
    confirmLabel: "Create batch",
  });
  if (!name) return;
  try {
    if (state.current && state.dirty) await persistCurrent(state.current.status);
    const batch = await request("/api/review-batches", {
      method: "POST",
      body: JSON.stringify({ name }),
    });
    showToast(`Active review batch: ${batch.name}`);
    await loadQueue(false);
  } catch (error) {
    showToast(error.message, true);
  }
}

async function activateReviewBatch() {
  const batchId = elements.batchSelect.value;
  try {
    if (state.current && state.dirty) await persistCurrent(state.current.status);
    const batch = await request(`/api/review-batches/${encodeURIComponent(batchId)}/activate`, {
      method: "POST",
      body: "{}",
    });
    showToast(`Active review batch: ${batch.name}`);
    await loadQueue(false);
  } catch (error) {
    showToast(error.message, true);
    renderBatchControls();
  }
}

function openDirectorySettings(required = false) {
  elements.inputRootPath.value = state.config?.input_root || "";
  elements.outputRootPath.value = state.config?.output_root || "";
  elements.modelDirectoryPath.value = state.config?.model_directory || state.config?.model_refresh?.directory || "";
  elements.settingsCancelButton.hidden = required;
  elements.settingsCloseButton.hidden = required;
  elements.settingsDialog.showModal();
}

async function saveDirectorySettings(event) {
  event.preventDefault();
  const payload = {
    input_root: elements.inputRootPath.value.trim(),
    output_root: elements.outputRootPath.value.trim(),
  };
  const modelDirectory = elements.modelDirectoryPath.value.trim();
  try {
    if (state.config?.configured && state.current && state.dirty) await persistCurrent(state.current.status);
    let modelRefresh = state.config?.model_refresh;
    if (modelDirectory && modelDirectory !== state.config?.model_refresh?.directory) {
      modelRefresh = await request("/api/models/directory", {
        method: "POST",
        body: JSON.stringify({ model_directory: modelDirectory }),
      });
    }
    const openWorkspace = (confirmCreate) => request("/api/workspace/open", {
      method: "POST",
      body: JSON.stringify({ ...payload, confirm_create: confirmCreate }),
    });
    let config;
    try {
      config = await openWorkspace(false);
    } catch (error) {
      if (!error.requires_confirmation) throw error;
      const confirmed = await openActionDialog({
        kicker: "New workspace",
        title: "Create this review workspace?",
        description: payload.output_root,
        notice: "This creates review.sqlite3, candidates/, and archives/ in the selected output directory.",
        confirmLabel: "Create workspace",
        input: false,
      });
      if (!confirmed) return;
      config = await openWorkspace(true);
    }
    state.config = config;
    if (modelRefresh) state.config.model_refresh = modelRefresh;
    elements.settingsDialog.close();
    window.location.reload();
  } catch (error) {
    showToast(error.message, true);
  }
}

function updateReviewControls() {
  if (!state.current) return;
  const poseState = state.current.pose_ready ? "Pose ready" : `Pose blocked: ${state.current.pose_error || "save review to validate"}`;
  const revisionState = state.current.archived
    ? ` · Archived ${archiveDate(state.current)}`
    : (state.current.supersedes_archive_batch ? ` · Correction of ${state.current.supersedes_archive_batch}` : "");
  elements.currentStatus.textContent = `${statusLabel(state.current.status)} · ${state.current.reviewed ? "Reviewed" : "Unreviewed"} · ${poseState}${revisionState}`;
  elements.reviewedButton.innerHTML = state.current.reviewed ? "Return to candidates <kbd>E</kbd>" : "Approve review <kbd>E</kbd>";
  renderDecisionActions();
}

function positionCanvasEditor() {
  const annotation = state.current?.annotations[state.activeAnnotation];
  if (!annotation || !state.transform || !state.selectedAnnotations.size) {
    elements.canvasEditor.hidden = true;
    return;
  }
  elements.canvasClassEditor.value = String(annotation.class_id);
  elements.canvasClassSwatch.style.background = classColor(annotation.class_id);
  elements.canvasSelectionCount.textContent = String(state.selectedAnnotations.size || 1);
  elements.canvasEditor.hidden = false;
  const editorWidth = 260;
  elements.canvasEditor.style.left = `${Math.max(6, (elements.canvasStage.clientWidth - editorWidth) / 2)}px`;
  elements.canvasEditor.style.top = "8px";
}

function clearCurrent() {
  state.current = null;
  state.image = null;
  state.dirty = false;
  state.selectedAnnotations = new Set();
  state.undoStack = [];
  elements.undoButton.disabled = true;
  elements.modelRefreshButton.disabled = true;
  elements.fileName.textContent = "No matching images";
  elements.imageMeta.textContent = "";
  elements.reviewNote.value = "";
  elements.currentStatus.textContent = "-";
  elements.canvasEditor.hidden = true;
  elements.targetList.replaceChildren();
  elements.emptyState.hidden = false;
  elements.emptyState.textContent = "No items match the current filters";
  renderDecisionActions();
  updateItemProgress();
  resizeCanvas();
}

function showEndOfQueue(clear = true) {
  if (clear) clearCurrent();
  if (!elements.endOfQueueDialog.open) elements.endOfQueueDialog.showModal();
}

function openActionDialog({
  kicker = "Action required",
  title,
  description = "",
  label = "",
  value = "",
  placeholder = "",
  hint = "",
  notice = "",
  confirmLabel = "Continue",
  input = true,
  selectLabel = "",
  selectValue = "",
  selectOptions = [],
  selectHint = "",
}) {
  if (actionDialogResolver) closeActionDialog(null);
  elements.actionDialogKicker.textContent = kicker;
  elements.actionDialogTitle.textContent = title;
  elements.actionDialogDescription.textContent = description;
  elements.actionDialogDescription.hidden = !description;
  elements.actionDialogSelectField.hidden = selectOptions.length === 0;
  elements.actionDialogSelectLabel.textContent = selectLabel;
  elements.actionDialogSelect.replaceChildren();
  for (const optionData of selectOptions) {
    const option = document.createElement("option");
    option.value = optionData.value;
    option.textContent = optionData.label;
    elements.actionDialogSelect.append(option);
  }
  elements.actionDialogSelect.value = selectValue;
  elements.actionDialogSelectHint.textContent = selectHint;
  elements.actionDialogSelectHint.hidden = !selectHint;
  elements.actionDialogField.hidden = !input;
  elements.actionDialogLabel.textContent = label;
  elements.actionDialogInput.value = value;
  elements.actionDialogInput.placeholder = placeholder;
  elements.actionDialogInput.required = input;
  elements.actionDialogHint.textContent = hint;
  elements.actionDialogHint.hidden = !hint;
  elements.actionDialogNotice.textContent = notice;
  elements.actionDialogNotice.hidden = !notice;
  elements.actionDialogSubmit.textContent = confirmLabel;
  elements.actionDialog.showModal();
  requestAnimationFrame(() => {
    if (input) {
      elements.actionDialogInput.focus();
      elements.actionDialogInput.select();
    } else {
      elements.actionDialogSubmit.focus();
    }
  });
  return new Promise((resolve) => {
    actionDialogResolver = resolve;
  });
}

function submitActionDialog(event) {
  event.preventDefault();
  if (!elements.actionDialogField.hidden && !elements.actionDialogInput.reportValidity()) return;
  const value = elements.actionDialogField.hidden ? true : elements.actionDialogInput.value.trim();
  if (!value) return;
  closeActionDialog(elements.actionDialogSelectField.hidden
    ? value
    : { value, selected: elements.actionDialogSelect.value });
}

function closeActionDialog(value) {
  resolveActionDialog(value);
  if (elements.actionDialog.open) elements.actionDialog.close();
}

function resolveActionDialog(value) {
  if (!actionDialogResolver) return;
  const resolve = actionDialogResolver;
  actionDialogResolver = null;
  resolve(value);
}

function markDirty() {
  state.dirty = true;
  if (state.current?.archived && !state.correctionNoticeShown) {
    state.correctionNoticeShown = true;
    showToast("Editing archived item; saving updates the archive immediately");
  }
}

function correctionSaveMessage(updated) {
  if (!updated?.archived && updated?.supersedes_archive_batch) {
    elements.statusFilter.value = updated.status === "accepted"
      ? "candidate"
      : (updated.status === "rejected" ? "rejected" : "pending");
    state.correctionNoticeShown = false;
    return `Correction revision created from ${updated.supersedes_archive_batch}`;
  }
  return "";
}
function structuredCloneSafe(value) { return JSON.parse(JSON.stringify(value)); }
function classColor(classId) {
  const value = Number(classId);
  if (value === 0) return CLASS_COLORS[0];
  if (value === 1) return "#ff0000";
  if (value === 4) return "#ff0000";
  return "#111111";
}
function classTextColor(classId) { return Number(classId) === 0 ? "#111111" : "#ffffff"; }
function toCanvasPoint(point) {
  return {
    x: state.transform.offsetX + point[0] * state.transform.scaleX,
    y: state.transform.offsetY + point[1] * state.transform.scaleY,
  };
}
function canvasPointer(event) { const rect = canvas.getBoundingClientRect(); return { x: event.clientX - rect.left, y: event.clientY - rect.top }; }
function pointInPolygon(point, polygon) {
  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const a = polygon[i];
    const b = polygon[j];
    if (((a.y > point.y) !== (b.y > point.y)) &&
        point.x < ((b.x - a.x) * (point.y - a.y)) / (b.y - a.y) + a.x) inside = !inside;
  }
  return inside;
}
function formatScore(value) { return value == null ? "manual" : `min ${Number(value).toFixed(2)}`; }
function archiveDate(item) { return item.archive_batch?.split("/")[0] || String(item.archived_at || "").slice(0, 10) || "Archived"; }
function statusLabel(value) {
  if (value === "accepted") return state.current?.archived ? "Archive" : "Candidate";
  return { pending: "To inspect", rejected: "Rejected", ignored: "Ignored", needs_review: "Needs review" }[value] || value;
}
function classOptions(selected) { return state.config.classes.map((value) => `<option value="${value.id}"${value.id === selected ? " selected" : ""}>${value.id}: ${escapeHtml(value.name)}</option>`).join(""); }
function escapeHtml(value) { const node = document.createElement("span"); node.textContent = String(value); return node.innerHTML; }
function showToast(message, error = false) {
  elements.toast.textContent = message;
  elements.toast.style.background = error ? "#7d2c29" : "#222a25";
  elements.toast.classList.add("visible");
  clearTimeout(showToast.timer);
  showToast.timer = setTimeout(() => elements.toast.classList.remove("visible"), error ? 5000 : 2800);
}

initialize();
