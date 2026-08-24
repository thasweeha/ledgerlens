/**
 * LedgerLens - Canvas Document Viewer & 300 DPI Drag-to-Select Re-OCR Tool
 * Handles high-resolution canvas rendering, pixel-exact coordinate translation,
 * visual bounding box overlays, and region crop OCR.
 */

const Viewer = (() => {
  // DOM Elements
  let viewport, stage, docCanvas, overlayCanvas, docCtx, overlayCtx, selectionBox, ocrPopover;
  let zoomDisplay, toolCropBtn, toolPanBtn, toggleBBoxesBtn;

  // State
  let currentImage = null;
  let currentPageIndex = 0;
  let currentSessionId = null;
  let currentBBoxes = []; // List of transaction bounding boxes on this page
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

  // Active OCR result cache
  let lastOCRResult = null;

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

    // Viewport mouse interactions (Pan & Drag-to-Select)
    viewport.addEventListener("mousedown", onMouseDown);
    window.addEventListener("mousemove", onMouseMove);
    window.addEventListener("mouseup", onMouseUp);

    // Wheel zoom
    viewport.addEventListener("wheel", onWheel, { passive: false });

    // Popover close button
    document.getElementById("ocrPopoverClose").addEventListener("click", hidePopover);

    // Popover action buttons
    document.getElementById("ocrSetDateBtn").addEventListener("click", () => applyOCRField("date"));
    document.getElementById("ocrSetDescBtn").addEventListener("click", () => applyOCRField("description"));
    document.getElementById("ocrSetAmtBtn").addEventListener("click", () => applyOCRField("amount"));
    document.getElementById("ocrInsertRowBtn").addEventListener("click", insertOCRAsNewRow);
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
   * Loads a 300 DPI page image and its bounding box metadata.
   */
  function loadPage(pageIndex, imageUrl, sessionId, bboxes = []) {
    currentPageIndex = pageIndex;
    currentSessionId = sessionId;
    currentBBoxes = bboxes;
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
   * Renders visual bounding box overlays scaled by current zoom factor.
   */
  function drawOverlays() {
    if (!overlayCtx || !currentImage) return;
    overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

    if (!showBBoxes || currentBBoxes.length === 0) return;

    overlayCtx.lineWidth = 1.5;

    currentBBoxes.forEach((item) => {
      const isSelected = (item.id === activeSelectedRowId);

      // Draw main row box
      if (item.bbox) {
        const rx = item.bbox.x * zoom;
        const ry = item.bbox.y * zoom;
        const rw = item.bbox.width * zoom;
        const rh = item.bbox.height * zoom;

        if (isSelected) {
          overlayCtx.strokeStyle = "#38bdf8";
          overlayCtx.fillStyle = "rgba(56, 189, 248, 0.18)";
          overlayCtx.fillRect(rx, ry, rw, rh);
          overlayCtx.strokeRect(rx, ry, rw, rh);
        } else {
          overlayCtx.strokeStyle = "rgba(148, 163, 184, 0.4)";
          overlayCtx.fillStyle = "rgba(148, 163, 184, 0.04)";
          overlayCtx.fillRect(rx, ry, rw, rh);
          overlayCtx.strokeRect(rx, ry, rw, rh);
        }
      }

      // Draw modular column boxes if available
      if (item.date_bbox) {
        const dx = item.date_bbox.x * zoom;
        const dy = item.date_bbox.y * zoom;
        const dw = item.date_bbox.width * zoom;
        const dh = item.date_bbox.height * zoom;
        overlayCtx.strokeStyle = "rgba(59, 130, 246, 0.5)";
        overlayCtx.strokeRect(dx, dy, dw, dh);
      }

      if (item.amount_bbox) {
        const ax = item.amount_bbox.x * zoom;
        const ay = item.amount_bbox.y * zoom;
        const aw = item.amount_bbox.width * zoom;
        const ah = item.amount_bbox.height * zoom;
        overlayCtx.strokeStyle = (item.type === "credit") ? "rgba(16, 185, 129, 0.6)" : "rgba(239, 68, 68, 0.6)";
        overlayCtx.strokeRect(ax, ay, aw, ah);
      }
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
    }
  }

  /**
   * Mouse Down Handler: initiates panning or drag-to-select box.
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
   * Mouse Up Handler: finalizes selection and triggers 300 DPI precision Re-OCR.
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

      // If selection is tiny, treat as a single click
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

      // Position Popover near selection
      showPopover(leftPx + stageRect.left - viewport.getBoundingClientRect().left, topPx + heightPx + 10);
      executeReOCR(bbox300DPI);
    }
  }

  /**
   * Handle single click on canvas to select corresponding row in editor grid.
   */
  function handleCanvasClick(x300, y300) {
    for (const item of currentBBoxes) {
      if (item.bbox) {
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
          break;
        }
      }
    }
  }

  function onWheel(e) {
    if (e.ctrlKey || e.metaKey) {
      e.preventDefault();
      const delta = e.deltaY < 0 ? 0.1 : -0.1;
      adjustZoom(delta);
    }
  }

  /**
   * Re-OCR API execution.
   */
  async function executeReOCR(bbox) {
    const loading = document.getElementById("ocrLoading");
    const content = document.getElementById("ocrContent");
    loading.style.display = "flex";
    content.style.display = "none";

    try {
      const res = await fetch("/api/re-ocr", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          session_id: currentSessionId,
          page_index: currentPageIndex,
          bbox: bbox,
          target_field: "all"
        })
      });

      if (!res.ok) throw new Error("Re-OCR request failed");
      const data = await res.json();
      lastOCRResult = data;

      document.getElementById("ocrExtractedText").value = data.cleaned_text || "";
      document.getElementById("ocrParsedDate").textContent = data.parsed_date || "—";
      document.getElementById("ocrParsedAmount").textContent = (data.parsed_amount !== null) ? `$${data.parsed_amount.toFixed(2)}` : "—";

      loading.style.display = "none";
      content.style.display = "block";
    } catch (err) {
      loading.innerHTML = `<span style="color:var(--accent-danger);">OCR Failed: ${err.message}</span>`;
    }
  }

  function showPopover(posX, posY) {
    const vpRect = viewport.getBoundingClientRect();
    const popW = 320;
    const popH = 220;

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
  }

  /**
   * Applies the OCR'd text or parsed value into the active editor row/cell.
   */
  function applyOCRField(field) {
    if (!lastOCRResult) return;
    const manualText = document.getElementById("ocrExtractedText").value;
    let val = manualText;

    if (field === "date" && lastOCRResult.parsed_date) {
      val = lastOCRResult.parsed_date;
    } else if (field === "amount" && lastOCRResult.parsed_amount !== null) {
      val = lastOCRResult.parsed_amount;
    }

    if (window.Editor && typeof window.Editor.updateActiveRowField === "function") {
      window.Editor.updateActiveRowField(field, val);
    }
    hidePopover();
  }

  /**
   * Inserts the full OCR'd selection as a new transaction row.
   */
  function insertOCRAsNewRow() {
    if (!lastOCRResult) return;
    const text = document.getElementById("ocrExtractedText").value;
    const date = lastOCRResult.parsed_date || "";
    const amount = lastOCRResult.parsed_amount !== null ? lastOCRResult.parsed_amount : 0.0;

    const newTx = {
      date: date,
      description: text,
      amount: amount,
      type: amount >= 0 ? "credit" : "debit",
      page: currentPageIndex + 1,
      bbox: lastSelectionBBox300DPI
    };

    if (window.Editor && typeof window.Editor.addNewTransactionRow === "function") {
      window.Editor.addNewTransactionRow(newTx);
    }
    hidePopover();
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
    currentBBoxes = [];
    activeHoverBox = null;
    activeSelectedRowId = null;
    lastOCRResult = null;
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
