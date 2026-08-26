/**
 * LedgerLens - Canvas Document Viewer & Snip-First Extraction Tool
 * Handles high-resolution canvas rendering, pixel-exact coordinate translation,
 * pending/confirmed bounding-box overlays, and the click-or-drag snip flow:
 * extract region -> preview (text + confidence + server-computed status) ->
 * Apply writes the cell AND appends to the audit trail.
 */

const Viewer = (() => {
  // ---- Configuration constants (confidence badge thresholds, percent) ----
  const CONFIDENCE_HIGH = 85;  // >= this -> green badge
  const CONFIDENCE_LOW = 50;   // < this -> red badge; between -> amber

  // DOM Elements
  let viewport, stage, docCanvas, overlayCanvas, docCtx, overlayCtx, selectionBox, ocrPopover;
  let zoomDisplay, toolCropBtn, toolPanBtn, toggleBBoxesBtn;
  let snipConfidenceChip, snipSourceChip, snipStatusRow, snipTargetLine;
  let snipApplyBtn, snipCancelBtn, ocrLoading, ocrContent, ocrExtractedText;

  const SNIP_LOADING_HTML = `<div class="spinner"></div><span id="snipLoadingText">Extracting region...</span>`;

  // State
  let currentImage = null;
  let currentPageIndex = 0;
  let currentSessionId = null;
  let currentPageType = "vector"; // "vector" | "scanned" (routing signal from pipeline)
  let currentBBoxes = []; // List of transaction bounding boxes on this page
  let confirmedRowIds = new Set(); // Rows whose values were snip-applied at least once
  let activeHoverBox = null;
  let activeSelectedRowId = null;
  let showBBoxes = true;

  // Zoom & Pan state
  let zoom = 1.0;
  let panMode = false;
  let isPanning = false;
  let panStartX = 0, panStartY = 0;
  let scrollStartX = 0, scrollStartY = 0;

  // Drag-to-select selection state
  let isSelecting = false;
  let selStartX = 0, selStartY = 0;
  let lastSelectionBBox300DPI = null; // in 300 DPI image pixels

  // Current snip preview context
  let currentSnip = null;
  // Shape: { bbox300, fieldHint, originRowId, result, fields, statuses }

  function init() {
    viewport = document.getElementById("viewerViewport");
    stage = document.getElementById("canvasStage");
    docCanvas = document.getElementById("docCanvas");
    overlayCanvas = document.getElementById("overlayCanvas");
    docCtx = docCanvas.getContext("2d");
    overlayCtx = overlayCanvas.getContext("2d");
    selectionBox = document.getElementById("selectionBox");
    ocrPopover = document.getElementById("ocrPopover");
    zoomDisplay = document.getElementById("zoomDisplay");

    toolCropBtn = document.getElementById("toolCropBtn");
    toolPanBtn = document.getElementById("toolPanBtn");
    toggleBBoxesBtn = document.getElementById("toggleBBoxesBtn");

    snipConfidenceChip = document.getElementById("snipConfidenceChip");
    snipSourceChip = document.getElementById("snipSourceChip");
    snipStatusRow = document.getElementById("snipStatusRow");
    snipTargetLine = document.getElementById("snipTargetLine");
    snipApplyBtn = document.getElementById("snipApplyBtn");
    snipCancelBtn = document.getElementById("snipCancelBtn");
    ocrLoading = document.getElementById("ocrLoading");
    ocrContent = document.getElementById("ocrContent");
    ocrExtractedText = document.getElementById("ocrExtractedText");

    bindEvents();
  }

  function bindEvents() {
    // Tool mode toggles
    toolCropBtn.addEventListener("click", () => setMode("crop"));
    toolPanBtn.addEventListener("click", () => setMode("pan"));

    // Zoom buttons
    document.getElementById("zoomInBtn").addEventListener("click", () => adjustZoom(0.15));
    document.getElementById("zoomOutBtn").addEventListener("click", () => adjustZoom(-0.15));
    document.getElementById("zoomFitBtn").addEventListener("click", fitWidth);

    // Toggle bounding boxes overlay
    toggleBBoxesBtn.addEventListener("click", () => {
      showBBoxes = !showBBoxes;
      toggleBBoxesBtn.classList.toggle("active", showBBoxes);
      drawOverlays();
    });

    // Viewport mouse interactions (Pan & Drag-to-Snip)
    viewport.addEventListener("mousedown", onMouseDown);
    window.addEventListener("mousemove", onMouseMove);
    window.addEventListener("mouseup", onMouseUp);

    // Wheel zoom
    viewport.addEventListener("wheel", onWheel, { passive: false });

    // Snip preview actions
    document.getElementById("ocrPopoverClose").addEventListener("click", hidePopover);
    snipCancelBtn.addEventListener("click", hidePopover);
    snipApplyBtn.addEventListener("click", applySnip);
  }

  function setMode(mode) {
    panMode = (mode === "pan");
    toolPanBtn.classList.toggle("active", panMode);
    toolCropBtn.classList.toggle("active", !panMode);
    viewport.classList.toggle("pan-mode", panMode);
  }

  function adjustZoom(delta) {
    const newZoom = Math.min(Math.max(0.2, zoom + delta), 3.5);
    setZoom(newZoom);
  }

  function setZoom(newZoom) {
    zoom = newZoom;
    zoomDisplay.textContent = `${Math.round(zoom * 100)}%`;
    render();
  }

  function fitWidth() {
    if (!currentImage) return;
    const availWidth = viewport.clientWidth - 40;
    if (availWidth > 0 && currentImage.naturalWidth > 0) {
      const fitZoom = availWidth / currentImage.naturalWidth;
      setZoom(Math.min(Math.max(0.2, fitZoom), 2.0));
    }
  }

  /**
   * Loads a 300 DPI page image, its bounding box metadata, and the page's
   * routing type ("vector" | "scanned") which selects native vs OCR snipping.
   */
  function loadPage(pageIndex, imageUrl, sessionId, bboxes = [], pageType = "vector") {
    currentPageIndex = pageIndex;
    currentSessionId = sessionId;
    currentBBoxes = bboxes;
    currentPageType = pageType;
    hidePopover();

    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      currentImage = img;
      document.getElementById("viewerEmptyState").style.display = "none";
      fitWidth();
      render();
    };
    img.src = imageUrl;
  }

  /**
   * Main render method: sizes canvas and redraws background + overlay layers.
   */
  function render() {
    if (!currentImage) return;

    const naturalW = currentImage.naturalWidth;
    const naturalH = currentImage.naturalHeight;

    const displayW = Math.round(naturalW * zoom);
    const displayH = Math.round(naturalH * zoom);

    docCanvas.width = naturalW;
    docCanvas.height = naturalH;
    docCanvas.style.width = `${displayW}px`;
    docCanvas.style.height = `${displayH}px`;

    overlayCanvas.width = displayW;
    overlayCanvas.height = displayH;
    overlayCanvas.style.width = `${displayW}px`;
    overlayCanvas.style.height = `${displayH}px`;

    stage.style.width = `${displayW}px`;
    stage.style.height = `${displayH}px`;

    // Draw document image at full resolution
    docCtx.drawImage(currentImage, 0, 0, naturalW, naturalH);

    drawOverlays();
  }

  /**
   * Renders bounding-box overlays scaled by current zoom factor.
   * Pipeline-detected rows start as dimmed dashed "pending" overlays;
   * rows confirmed via snip Apply switch to solid teal.
   */
  function drawOverlays() {
    if (!overlayCtx || !currentImage) return;
    overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

    if (!showBBoxes || currentBBoxes.length === 0) return;

    overlayCtx.lineWidth = 1.5;

    currentBBoxes.forEach((item) => {
      const isSelected = (item.id === activeSelectedRowId);
      const isConfirmed = confirmedRowIds.has(item.id);

      // Draw main row box
      if (item.bbox) {
        const rx = item.bbox.x * zoom;
        const ry = item.bbox.y * zoom;
        const rw = item.bbox.width * zoom;
        const rh = item.bbox.height * zoom;

        overlayCtx.setLineDash(isConfirmed ? [] : [6, 4]);

        if (isSelected) {
          overlayCtx.strokeStyle = "#38bdf8";
          overlayCtx.fillStyle = "rgba(56, 189, 248, 0.18)";
        } else if (isConfirmed) {
          overlayCtx.strokeStyle = "rgba(15, 118, 110, 0.8)";
          overlayCtx.fillStyle = "rgba(15, 118, 110, 0.07)";
        } else {
          // Pending: dimmed dashed grey
          overlayCtx.strokeStyle = "rgba(148, 163, 184, 0.55)";
          overlayCtx.fillStyle = "rgba(148, 163, 184, 0.04)";
        }
        overlayCtx.fillRect(rx, ry, rw, rh);
        overlayCtx.strokeRect(rx, ry, rw, rh);
        overlayCtx.setLineDash([]);
      }

      // Draw modular column boxes if available (dimmer while pending)
      overlayCtx.globalAlpha = isConfirmed ? 0.9 : 0.4;
      if (item.date_bbox) {
        overlayCtx.strokeStyle = "rgba(59, 130, 246, 0.5)";
        strokeItemBox(item.date_bbox);
      }

      if (item.desc_bbox) {
        overlayCtx.strokeStyle = "rgba(148, 163, 184, 0.45)";
        strokeItemBox(item.desc_bbox);
      }

      if (item.amount_bbox) {
        overlayCtx.strokeStyle = (item.type === "credit") ? "rgba(16, 185, 129, 0.6)" : "rgba(239, 68, 68, 0.6)";
        strokeItemBox(item.amount_bbox);
      }
      overlayCtx.globalAlpha = 1.0;
    });

    // Draw active hover box
    if (activeHoverBox) {
      overlayCtx.strokeStyle = "#f59e0b";
      overlayCtx.lineWidth = 2;
      overlayCtx.strokeRect(
        activeHoverBox.x * zoom,
        activeHoverBox.y * zoom,
        activeHoverBox.width * zoom,
        activeHoverBox.height * zoom
      );
      overlayCtx.lineWidth = 1.5;
    }
  }

  function strokeItemBox(b) {
    overlayCtx.strokeRect(b.x * zoom, b.y * zoom, b.width * zoom, b.height * zoom);
  }

  /**
   * Mouse Down Handler: initiates panning or drag-to-snip selection.
   */
  function onMouseDown(e) {
    if (!currentImage) return;
    if (e.target.closest("#ocrPopover")) return;

    const stageRect = stage.getBoundingClientRect();

    if (panMode || e.button === 1 || e.altKey) {
      isPanning = true;
      panStartX = e.clientX;
      panStartY = e.clientY;
      scrollStartX = viewport.scrollLeft;
      scrollStartY = viewport.scrollTop;
      e.preventDefault();
      return;
    }

    // Drag-to-select selection mode
    if (e.button === 0) {
      // Check if clicked inside stage
      if (
        e.clientX >= stageRect.left &&
        e.clientX <= stageRect.right &&
        e.clientY >= stageRect.top &&
        e.clientY <= stageRect.bottom
      ) {
        isSelecting = true;
        selStartX = e.clientX - stageRect.left;
        selStartY = e.clientY - stageRect.top;

        selectionBox.style.left = `${selStartX}px`;
        selectionBox.style.top = `${selStartY}px`;
        selectionBox.style.width = "0px";
        selectionBox.style.height = "0px";
        selectionBox.style.display = "block";

        hidePopover();
        e.preventDefault();
      }
    }
  }

  /**
   * Mouse Move Handler: updates selection box or pans viewport.
   */
  function onMouseMove(e) {
    if (isPanning) {
      const dx = e.clientX - panStartX;
      const dy = e.clientY - panStartY;
      viewport.scrollLeft = scrollStartX - dx;
      viewport.scrollTop = scrollStartY - dy;
      return;
    }

    if (isSelecting && currentImage) {
      const stageRect = stage.getBoundingClientRect();
      const currentX = Math.max(0, Math.min(stageRect.width, e.clientX - stageRect.left));
      const currentY = Math.max(0, Math.min(stageRect.height, e.clientY - stageRect.top));

      const left = Math.min(selStartX, currentX);
      const top = Math.min(selStartY, currentY);
      const width = Math.abs(currentX - selStartX);
      const height = Math.abs(currentY - selStartY);

      selectionBox.style.left = `${left}px`;
      selectionBox.style.top = `${top}px`;
      selectionBox.style.width = `${width}px`;
      selectionBox.style.height = `${height}px`;
      return;
    }

    // Hover detection over bounding boxes
    if (!isSelecting && !isPanning && showBBoxes && currentImage) {
      const stageRect = stage.getBoundingClientRect();
      if (
        e.clientX >= stageRect.left &&
        e.clientX <= stageRect.right &&
        e.clientY >= stageRect.top &&
        e.clientY <= stageRect.bottom
      ) {
        const mouseCanvasX = (e.clientX - stageRect.left) / zoom;
        const mouseCanvasY = (e.clientY - stageRect.top) / zoom;

        let hit = null;
        for (const item of currentBBoxes) {
          if (item.bbox) {
            const b = item.bbox;
            if (
              mouseCanvasX >= b.x &&
              mouseCanvasX <= b.x + b.width &&
              mouseCanvasY >= b.y &&
              mouseCanvasY <= b.y + b.height
            ) {
              hit = item;
              break;
            }
          }
        }

        if (hit && hit.bbox !== activeHoverBox) {
          activeHoverBox = hit.bbox;
          drawOverlays();
        } else if (!hit && activeHoverBox) {
          activeHoverBox = null;
          drawOverlays();
        }
      }
    }
  }

  /**
   * Mouse Up Handler: finalizes a drag-selection and triggers a region snip.
   */
  function onMouseUp(e) {
    if (isPanning) {
      isPanning = false;
      return;
    }

    if (isSelecting && currentImage) {
      isSelecting = false;
      const stageRect = stage.getBoundingClientRect();
      const currentX = Math.max(0, Math.min(stageRect.width, e.clientX - stageRect.left));
      const currentY = Math.max(0, Math.min(stageRect.height, e.clientY - stageRect.top));

      const leftPx = Math.min(selStartX, currentX);
      const topPx = Math.min(selStartY, currentY);
      const widthPx = Math.abs(currentX - selStartX);
      const heightPx = Math.abs(currentY - selStartY);

      // If selection is tiny, treat as a single click on a pending box
      if (widthPx < 8 || heightPx < 8) {
        selectionBox.style.display = "none";
        handleCanvasClick(leftPx / zoom, topPx / zoom);
        return;
      }

      // Translate display CSS pixels to 300 DPI native image pixels
      const bbox300DPI = {
        x: Math.round(leftPx / zoom),
        y: Math.round(topPx / zoom),
        width: Math.round(widthPx / zoom),
        height: Math.round(heightPx / zoom),
        page: currentPageIndex + 1
      };
      lastSelectionBBox300DPI = bbox300DPI;

      startSnip(bbox300DPI, "all", null);
    }
  }

  /**
   * Single click on canvas: if it lands on a detected row box, select the
   * grid row AND snip that region (sub-box precision for column hits).
   */
  function handleCanvasClick(x300, y300) {
    for (const item of currentBBoxes) {
      if (!item.bbox) continue;
      const b = item.bbox;
      if (
        x300 >= b.x &&
        x300 <= b.x + b.width &&
        y300 >= b.y &&
        y300 <= b.y + b.height
      ) {
        activeSelectedRowId = item.id;
        drawOverlays();
        if (window.Editor && typeof window.Editor.selectRowById === "function") {
          window.Editor.selectRowById(item.id);
        }
        const hit = detectFieldHit(item, x300, y300);
        startSnip(hit.bbox, hit.hint, item.id);
        return;
      }
    }
  }

  /**
   * Determines which column sub-box (if any) contains the click, giving the
   * snip a precise region and a single-field target hint.
   */
  function detectFieldHit(item, x300, y300) {
    const inside = (b) => b && x300 >= b.x && x300 <= b.x + b.width && y300 >= b.y && y300 <= b.y + b.height;
    if (inside(item.date_bbox)) return { hint: "date", bbox: item.date_bbox };
    if (inside(item.desc_bbox)) return { hint: "description", bbox: item.desc_bbox };
    if (inside(item.amount_bbox)) return { hint: "amount", bbox: item.amount_bbox };
    return { hint: "all", bbox: item.bbox };
  }

  function onWheel(e) {
    if (e.ctrlKey || e.metaKey) {
      e.preventDefault();
      const delta = e.deltaY < 0 ? 0.1 : -0.1;
      adjustZoom(delta);
    }
  }

  // ---------------------------------------------------------------------
  // Snip-first extraction flow
  // ---------------------------------------------------------------------

  /**
   * Opens the preview panel near the region and extracts it. Vector pages
   * use the native PDF text stream; scanned pages use TrOCR (backend decides).
   */
  function startSnip(bbox, fieldHint, originRowId) {
    if (!currentSessionId) return;

    const vpRect = viewport.getBoundingClientRect();
    const stageRect = stage.getBoundingClientRect();

    currentSnip = { bbox300: bbox, fieldHint, originRowId, result: null, fields: {}, statuses: {} };

    showPopover(
      bbox.x * zoom + (stageRect.left - vpRect.left),
      bbox.height * zoom + bbox.y * zoom + (stageRect.top - vpRect.top) + 10
    );

    ocrLoading.innerHTML = SNIP_LOADING_HTML;
    const loadingText = document.getElementById("snipLoadingText");
    if (loadingText) {
      loadingText.textContent = (currentPageType === "vector")
        ? "Reading native text stream..."
        : "Scanning region with OCR...";
    }
    ocrLoading.style.display = "flex";
    ocrContent.style.display = "none";

    executeSnip(bbox, fieldHint);
  }

  async function executeSnip(bbox, fieldHint) {
    try {
      const res = await fetch("/api/snip-extract", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: currentSessionId,
          page_index: currentPageIndex,
          bbox: bbox,
          field_hint: fieldHint
        })
      });

      if (!res.ok) throw new Error("Extraction request failed");
      const data = await res.json();
      if (currentSnip) currentSnip.result = data;

      renderSnipPreview(data);
    } catch (err) {
      ocrLoading.innerHTML = `<span style="color:var(--accent-danger);">Extraction failed: ${err.message}</span>`;
    }
  }

  /**
   * Maps an extraction result onto the fields that Apply will write.
   */
  function fieldsFromResult(result, fieldHint) {
    const fields = {};
    const cleaned = (result.cleaned_text || "").trim();

    if ((fieldHint === "all" || fieldHint === "date") && result.parsed_date) {
      fields.date = result.parsed_date;
    }
    if ((fieldHint === "all" || fieldHint === "amount") &&
        result.parsed_amount !== null && result.parsed_amount !== undefined) {
      fields.amount = result.parsed_amount;
    }
    if ((fieldHint === "all" || fieldHint === "description") && cleaned) {
      fields.description = cleaned;
    }
    return fields;
  }

  function confidenceClass(pct) {
    if (pct >= CONFIDENCE_HIGH) return "conf-high";
    if (pct >= CONFIDENCE_LOW) return "conf-mid";
    return "conf-low";
  }

  const STATUS_LABELS = {
    MATCH: "MATCH",
    DIFFERENT: "FILLED",
    LOW_CONFIDENCE: "LOW CONF.",
    CONFLICT: "CONFLICT"
  };

  const STATUS_CLASSES = {
    MATCH: "status-match",
    DIFFERENT: "status-different",
    LOW_CONFIDENCE: "status-lowconf",
    CONFLICT: "status-conflict"
  };

  function renderSnipPreview(data) {
    ocrExtractedText.value = data.cleaned_text || "";
    document.getElementById("ocrParsedDate").textContent = data.parsed_date || "—";
    document.getElementById("ocrParsedAmount").textContent =
      (data.parsed_amount !== null && data.parsed_amount !== undefined) ? `$${data.parsed_amount.toFixed(2)}` : "—";

    // Confidence badge (color-coded via configurable thresholds)
    const pct = Math.round((data.confidence || 0) * 100);
    snipConfidenceChip.className = `chip conf-chip ${confidenceClass(pct)}`;
    snipConfidenceChip.textContent = `Confidence ${pct}%`;

    // Extraction source badge
    const isNative = data.extraction_source === "digital_native";
    snipSourceChip.textContent = isNative ? "Native Text" : "TrOCR OCR";

    currentSnip.fields = fieldsFromResult(data, currentSnip.fieldHint);
    const fieldCount = Object.keys(currentSnip.fields).length;
    snipApplyBtn.disabled = (fieldCount === 0);

    renderTargetLine(fieldCount);
    renderStatusChips(); // placeholder chips until server responds

    ocrLoading.style.display = "none";
    ocrContent.style.display = "block";

    fetchPreviewStatuses();
  }

  /**
   * Resolves which grid row the snip will be applied to.
   * Priority: originating clicked row > active editor row > new row.
   */
  function resolveTarget() {
    if (!window.Editor) return { rowIndex: null };
    if (currentSnip.originRowId) {
      const idx = window.Editor.resolveRowIndex(currentSnip.originRowId);
      if (idx >= 0) return { rowIndex: idx };
    }
    const actIdx = window.Editor.getActiveRowIndex();
    if (actIdx >= 0) return { rowIndex: actIdx };
    return { rowIndex: null };
  }

  function renderTargetLine(fieldCount) {
    const target = resolveTarget();
    const fieldNames = Object.keys(currentSnip.fields);
    const fieldLabel = fieldCount === 0
      ? "nothing parseable detected"
      : fieldNames.join(", ");
    const targetLabel = (target.rowIndex === null)
      ? "a NEW row"
      : `Row ${target.rowIndex + 1}`;
    snipTargetLine.textContent = `Apply target: ${targetLabel} — ${fieldLabel}`;
  }

  function renderStatusChips() {
    snipStatusRow.innerHTML = "";
    const entries = Object.entries(currentSnip.fields);
    if (entries.length === 0) {
      snipStatusRow.innerHTML = `<span class="chip">No text detected</span>`;
      return;
    }
    entries.forEach(([fieldName]) => {
      const known = currentSnip.statuses[fieldName];
      const cls = known ? (STATUS_CLASSES[known] || "") : "";
      const label = known ? (STATUS_LABELS[known] || known) : "…";
      const chip = document.createElement("span");
      chip.className = `chip status-chip ${cls}`;
      chip.dataset.field = fieldName;
      chip.textContent = `${fieldName}: ${label}`;
      snipStatusRow.appendChild(chip);
    });
  }

  /**
   * Asks the SERVER to classify each candidate field (dry_run=true --
   * computed authoritatively server-side, nothing persisted yet) so the
   * user sees MATCH / FILLED / LOW CONF. / CONFLICT before clicking Apply.
   */
  async function fetchPreviewStatuses() {
    const target = resolveTarget();
    const txnIndex = (target.rowIndex !== null) ? target.rowIndex : -1;

    const entries = Object.entries(currentSnip.fields);
    const results = await Promise.all(entries.map(async ([fieldName, val]) => {
      const oldValue = window.Editor.peekFieldValue
        ? window.Editor.peekFieldValue(target.rowIndex, fieldName)
        : "";
      try {
        const res = await fetch("/api/audit-log", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            session_id: currentSessionId,
            transaction_index: txnIndex,
            field_name: fieldName,
            old_value: (oldValue === null || oldValue === undefined) ? "" : String(oldValue),
            new_value: String(val),
            page: currentSnip.bbox300.page,
            bbox: currentSnip.bbox300,
            source: "manual_snip",
            confidence: currentSnip.result ? currentSnip.result.confidence : null,
            dry_run: true
          })
        });
        if (!res.ok) throw new Error("status computation failed");
        const data = await res.json();
        return [fieldName, data.status];
      } catch (err) {
        console.warn("Preview status failed:", err);
        return [fieldName, null];
      }
    }));

    if (!currentSnip) return;
    results.forEach(([fieldName, status]) => { currentSnip.statuses[fieldName] = status; });

    // Only repaint if this preview is still the visible one
    if (ocrPopover.style.display !== "none" && ocrContent.style.display === "block") {
      renderStatusChips();
    }
  }

  /**
   * Apply: writes extracted values into the target cell(s) via the Editor,
   * marks the row confirmed on canvas, and persists one audit row per
   * changed field (source="manual_snip").
   */
  async function applySnip() {
    if (!currentSnip || !currentSnip.result) return;

    const fields = fieldsFromResult(currentSnip.result, currentSnip.fieldHint);
    // Respect manual edits made in the preview text input
    const userInput = (ocrExtractedText.value || "").trim();
    if (fields.description && userInput) fields.description = userInput;

    if (Object.keys(fields).length === 0) {
      hidePopover();
      return;
    }

    const target = resolveTarget();
    let applied = null;
    try {
      applied = window.Editor.applySnipFields(
        target.rowIndex,
        fields,
        { page: currentPageIndex + 1, bbox: currentSnip.bbox300 }
      );
    } catch (err) {
      console.error("Failed to apply snip:", err);
      return;
    }
    if (!applied) return;

    if (applied.rowId) {
      confirmedRowIds.add(applied.rowId);
      drawOverlays();
    }

    // Audit trail: one row per changed field (best-effort, non-blocking UI)
    const snipMeta = {
      sessionId: currentSessionId,
      page: currentSnip.bbox300.page,
      bbox: currentSnip.bbox300,
      confidence: currentSnip.result.confidence
    };
    for (const change of (applied.changes || [])) {
      postAuditEntry(snipMeta, applied.rowIndex, change);
    }

    hidePopover();
  }

  function postAuditEntry(meta, rowIndex, change) {
    fetch("/api/audit-log", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: meta.sessionId,
        transaction_index: rowIndex,
        field_name: change.field_name,
        old_value: change.old_value,
        new_value: change.new_value,
        page: meta.page,
        bbox: meta.bbox,
        source: "manual_snip",
        confidence: meta.confidence,
        dry_run: false
      })
    }).catch((err) => console.warn("Audit log failed:", err));
  }

  function showPopover(posX, posY) {
    const vpRect = viewport.getBoundingClientRect();
    const popW = 320;
    const popH = 260;

    let x = Math.max(10, Math.min(vpRect.width - popW - 20, posX));
    let y = Math.max(10, Math.min(vpRect.height - popH - 20, posY));

    ocrPopover.style.left = `${x}px`;
    ocrPopover.style.top = `${y}px`;
    ocrPopover.style.display = "flex";
  }

  function hidePopover() {
    ocrPopover.style.display = "none";
    selectionBox.style.display = "none";
    lastSelectionBBox300DPI = null;
    currentSnip = null;
  }

  /**
   * External method called by Editor to highlight the active row's bbox on canvas.
   */
  function highlightRow(rowId, bbox) {
    activeSelectedRowId = rowId;
    drawOverlays();

    // Auto-scroll canvas viewport to bring highlighted bbox into view
    if (bbox && currentImage) {
      const targetY = (bbox.y * zoom) - (viewport.clientHeight / 3);
      viewport.scrollTo({ top: Math.max(0, targetY), behavior: "smooth" });
    }
  }

  function updateBBoxes(bboxes) {
    currentBBoxes = bboxes;
    drawOverlays();
  }

  /**
   * Clears all viewer state and restores the initial empty upload screen.
   * Called by Editor when the session is exited.
   */
  function reset() {
    currentImage = null;
    currentSessionId = null;
    currentPageIndex = 0;
    currentPageType = "vector";
    currentBBoxes = [];
    confirmedRowIds.clear();
    activeHoverBox = null;
    activeSelectedRowId = null;
    lastSelectionBBox300DPI = null;
    isSelecting = false;
    isPanning = false;

    zoom = 1.0;
    if (zoomDisplay) zoomDisplay.textContent = "100%";

    if (docCanvas) {
      docCanvas.width = 0;
      docCanvas.height = 0;
      docCanvas.style.width = "";
      docCanvas.style.height = "";
    }
    if (overlayCanvas) {
      overlayCanvas.width = 0;
      overlayCanvas.height = 0;
      overlayCanvas.style.width = "";
      overlayCanvas.style.height = "";
    }
    if (stage) {
      stage.style.width = "";
      stage.style.height = "";
    }

    hidePopover();

    const emptyState = document.getElementById("viewerEmptyState");
    if (emptyState) emptyState.style.display = "flex";
  }

  return {
    init,
    loadPage,
    render,
    setZoom,
    fitWidth,
    highlightRow,
    updateBBoxes,
    hidePopover,
    reset
  };
})();

document.addEventListener("DOMContentLoaded", () => {
  Viewer.init();
});
