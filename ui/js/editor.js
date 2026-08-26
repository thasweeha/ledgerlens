/**
 * LedgerLens - Dynamic Spreadsheet Editor & Real-Time Ledger Reconciliation
 * Provides keyboard grid navigation, inline cell editing, real-time math validation,
 * document upload, multi-tab Excel/JSON export, and bi-directional canvas sync.
 */

const Editor = (() => {
  // Application State
  let currentStatement = null;
  let activePageNumber = 1;
  let activeRowIndex = -1;
  let activeColIndex = 1; // 1: Date, 2: Description, 3: Debit, 4: Credit, 5: Type, 6: Balance
  let isEditingCell = false;

  // Exit / unsaved-changes tracking
  let lastExportFormat = null;
  let isDirty = false;

  // DOM Elements
  let gridTableBody, gridTableContainer, rowCountBadge;
  let openingBalInput, closingBalInput, totalCreditsDisplay, totalDebitsDisplay;
  let calcClosingDisplay, reconStatusBadge, reconDiffDisplay;
  let pageNavControls, currentPageNum, totalPagesNum, firstPageBtn, prevPageBtn, nextPageBtn, lastPageBtn;
  let docInfo, docFilename, docTypeBadge, exportGroup;
  let globalLoading, loadingTitle, loadingSubtitle;
  let gridSearchInput;

  function init() {
    gridTableBody = document.getElementById("gridTableBody");
    gridTableContainer = document.getElementById("gridTableContainer");
    rowCountBadge = document.getElementById("rowCountBadge");

    openingBalInput = document.getElementById("openingBalInput");
    closingBalInput = document.getElementById("closingBalInput");
    totalCreditsDisplay = document.getElementById("totalCreditsDisplay");
    totalDebitsDisplay = document.getElementById("totalDebitsDisplay");
    calcClosingDisplay = document.getElementById("calcClosingDisplay");
    reconStatusBadge = document.getElementById("reconStatusBadge");
    reconDiffDisplay = document.getElementById("reconDiffDisplay");

    pageNavControls = document.getElementById("pageNavControls");
    currentPageNum = document.getElementById("currentPageNum");
    totalPagesNum = document.getElementById("totalPagesNum");
    firstPageBtn = document.getElementById("firstPageBtn");
    prevPageBtn = document.getElementById("prevPageBtn");
    nextPageBtn = document.getElementById("nextPageBtn");
    lastPageBtn = document.getElementById("lastPageBtn");

    docInfo = document.getElementById("docInfo");
    docFilename = document.getElementById("docFilename");
    docTypeBadge = document.getElementById("docTypeBadge");
    exportGroup = document.getElementById("exportGroup");

    globalLoading = document.getElementById("globalLoading");
    loadingTitle = document.getElementById("loadingTitle");
    loadingSubtitle = document.getElementById("loadingSubtitle");
    gridSearchInput = document.getElementById("gridSearchInput");

    bindEvents();
    window.Editor = Editor; // Expose globally for viewer interaction
  }

  function bindEvents() {
    // File upload
    const fileInput = document.getElementById("pdfFileInput");
    fileInput.addEventListener("change", (e) => {
      if (e.target.files.length > 0) {
        uploadPDFFile(e.target.files[0]);
      }
    });

    // Dropzone drag & drop
    const dropZone = document.getElementById("dropZone");
    if (dropZone) {
      dropZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropZone.classList.add("dragover");
      });
      dropZone.addEventListener("dragleave", () => dropZone.classList.remove("dragover"));
      dropZone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropZone.classList.remove("dragover");
        if (e.dataTransfer.files.length > 0 && e.dataTransfer.files[0].name.endsWith(".pdf")) {
          uploadPDFFile(e.dataTransfer.files[0]);
        } else {
          showToast("Please drop a valid PDF file.", "error");
        }
      });
    }

    // Demo statement button
    document.getElementById("demoBtn").addEventListener("click", loadDemoStatement);

    // Page navigation
    firstPageBtn.addEventListener("click", () => switchPage(1));
    prevPageBtn.addEventListener("click", () => switchPage(activePageNumber - 1));
    nextPageBtn.addEventListener("click", () => switchPage(activePageNumber + 1));
    lastPageBtn.addEventListener("click", () => switchPage(currentStatement.page_count));

    // Summary inputs real-time recalculation
    openingBalInput.addEventListener("input", onBalanceInputChange);
    closingBalInput.addEventListener("input", onBalanceInputChange);

    // Grid action buttons
    document.getElementById("addRowBtn").addEventListener("click", () => addNewTransactionRow());
    document.getElementById("deleteRowBtn").addEventListener("click", deleteActiveRow);

    // Search filter
    gridSearchInput.addEventListener("input", filterGridRows);

    // Export buttons
    document.getElementById("exportJsonBtn").addEventListener("click", () => exportStatement("json"));
    document.getElementById("exportXlsxBtn").addEventListener("click", () => exportStatement("xlsx"));

    // Exit button & confirmation modal
    document.getElementById("exitBtn").addEventListener("click", openExitModal);
    document.getElementById("exitSaveBtn").addEventListener("click", saveAndExit);
    document.getElementById("exitDiscardBtn").addEventListener("click", exitWithoutSaving);
    document.getElementById("exitCancelBtn").addEventListener("click", closeExitModal);
    const exitOverlay = document.getElementById("exitConfirmOverlay");
    exitOverlay.addEventListener("mousedown", (e) => {
      if (e.target === exitOverlay) closeExitModal();
    });
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && exitOverlay.style.display === "flex") closeExitModal();
    });

    // Warn before leaving the page with unsaved changes
    window.addEventListener("beforeunload", (e) => {
      if (isDirty) {
        e.preventDefault();
        e.returnValue = "";
      }
    });

    // Keyboard navigation on grid container
    gridTableContainer.addEventListener("keydown", handleGridKeyDown);
  }

  /**
   * Upload PDF to FastAPI backend.
   */
  async function uploadPDFFile(file) {
    showLoading("Ingesting Bank Statement...", "Extracting vector streams & rendering 300 DPI canvas");

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("/api/upload", {
        method: "POST",
        body: formData
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || "Upload failed");
      }

      const statementData = await res.json();
      loadStatement(statementData);
      showToast(`Successfully processed ${file.name} (${statementData.page_count} pages)`, "success");
    } catch (err) {
      showToast(`Error processing PDF: ${err.message}`, "error");
    } finally {
      hideLoading();
    }
  }

  /**
   * Loads parsed statement data into UI state and renders grid.
   */
  function loadStatement(data) {
    currentStatement = data;
    activePageNumber = 1;

    // Header info
    docInfo.style.display = "flex";
    docFilename.textContent = data.filename;
    docTypeBadge.textContent = data.pages && data.pages[0] ? data.pages[0].type.toUpperCase() + " PDF" : "PDF";

    // Multi-page controls
    if (data.page_count > 1) {
      pageNavControls.style.display = "flex";
      currentPageNum.textContent = "1";
      totalPagesNum.textContent = data.page_count;
      updateNavBtnStates();
    } else {
      pageNavControls.style.display = "none";
    }

    // Export group
    exportGroup.style.display = "flex";

    // Set balances
    openingBalInput.value = data.opening_balance.toFixed(2);
    closingBalInput.value = data.closing_balance.toFixed(2);

    // Load first page into Viewer
    if (data.pages && data.pages.length > 0) {
      const p1 = data.pages[0];
      const pageBBoxes = data.transactions
        .filter((t) => t.page === 1)
        .map((t) => ({
          id: t.id,
          bbox: t.bbox,
          date_bbox: t.date_bbox,
          desc_bbox: t.desc_bbox,
          amount_bbox: t.amount_bbox,
          type: t.type
        }));

      Viewer.loadPage(0, p1.image_url, data.session_id, pageBBoxes, p1.type || "vector");
    }

    renderGrid();
    recalculateReconciliation();

    if (data.transactions.length > 0) {
      selectCell(0, 1);
    }

    // A freshly loaded statement is unsaved until exported.
    isDirty = true;
  }

  /**
   * Switches active page in multi-page statement.
   */
  function switchPage(pageNum) {
    if (!currentStatement || pageNum < 1 || pageNum > currentStatement.page_count) return;
    activePageNumber = pageNum;
    currentPageNum.textContent = pageNum;
    updateNavBtnStates();

    const pageIdx = pageNum - 1;
    const p = currentStatement.pages[pageIdx];
    const pageBBoxes = currentStatement.transactions
      .filter((t) => t.page === pageNum)
      .map((t) => ({
        id: t.id,
        bbox: t.bbox,
        date_bbox: t.date_bbox,
        desc_bbox: t.desc_bbox,
        amount_bbox: t.amount_bbox,
        type: t.type
      }));

    Viewer.loadPage(pageIdx, p.image_url, currentStatement.session_id, pageBBoxes, p.type || "vector");
    renderGrid();
  }

  function updateNavBtnStates() {
    if (!currentStatement) return;
    firstPageBtn.disabled = (activePageNumber <= 1);
    prevPageBtn.disabled = (activePageNumber <= 1);
    nextPageBtn.disabled = (activePageNumber >= currentStatement.page_count);
    lastPageBtn.disabled = (activePageNumber >= currentStatement.page_count);
  }

  /**
   * Renders the dynamic spreadsheet grid.
   */
  function renderGrid() {
    if (!currentStatement) return;

    gridTableBody.innerHTML = "";
    const filterText = gridSearchInput.value.toLowerCase().trim();

    const rows = currentStatement.transactions;
    rowCountBadge.textContent = `${rows.length} Rows`;

    if (rows.length === 0) {
      gridTableBody.innerHTML = `
        <tr class="empty-table-row">
          <td colspan="9">No transactions found. Click "+ Add Row" or use Drag-to-OCR on canvas.</td>
        </tr>
      `;
      return;
    }

    rows.forEach((tx, idx) => {
      // Search filter check
      if (filterText) {
        const combined = `${tx.date} ${tx.description} ${tx.amount} ${tx.type}`.toLowerCase();
        if (!combined.includes(filterText)) return;
      }

      const tr = document.createElement("tr");
      tr.dataset.rowIndex = idx;
      tr.dataset.rowId = tx.id;
      if (idx === activeRowIndex) tr.classList.add("selected");

      const amtVal = parseFloat(tx.amount) || 0.0;
      const isCredit = tx.type === "credit";
      const amtFormatted = `$${Math.abs(amtVal).toFixed(2)}`;

      tr.innerHTML = `
        <td style="text-align: center; color: var(--text-muted); font-size: 0.75rem;">${idx + 1}</td>
        <td class="grid-cell" data-col="1">${escapeHtml(tx.date)}</td>
        <td class="grid-cell" data-col="2" title="${escapeHtml(tx.description)}">${escapeHtml(tx.description)}</td>
        <td class="grid-cell cell-amount ${isCredit ? '' : 'debit'}" data-col="3">${isCredit ? '' : amtFormatted}</td>
        <td class="grid-cell cell-amount ${isCredit ? 'credit' : ''}" data-col="4">${isCredit ? amtFormatted : ''}</td>
        <td class="grid-cell" data-col="5" style="text-align: center;">
          <span class="type-pill ${isCredit ? 'credit' : 'debit'}" data-action="toggle-type">${isCredit ? 'Credit' : 'Debit'}</span>
        </td>
        <td class="grid-cell" data-col="6" style="text-align: right; font-family: var(--font-mono);">${tx.balance !== null && tx.balance !== undefined ? `$${Number(tx.balance).toFixed(2)}` : ''}</td>
        <td style="text-align: center; font-size: 0.75rem; color: var(--text-muted);">${tx.page || 1}</td>
        <td style="text-align: center;">
          <button class="row-action-btn" title="Delete Row" data-action="delete-row">
            <svg viewBox="0 0 24 24" width="14" height="14" fill="none" stroke="currentColor" stroke-width="2">
              <line x1="18" y1="6" x2="6" y2="18"></line>
              <line x1="6" y1="6" x2="18" y2="18"></line>
            </svg>
          </button>
        </td>
      `;

      // Row click & cell events
      tr.addEventListener("click", (e) => {
        const target = e.target;
        if (target.dataset.action === "toggle-type" || target.closest('[data-action="toggle-type"]')) {
          toggleTransactionType(idx);
          return;
        }
        if (target.dataset.action === "delete-row" || target.closest('[data-action="delete-row"]')) {
          deleteRowByIndex(idx);
          return;
        }

        const cell = target.closest(".grid-cell");
        if (cell) {
          const col = parseInt(cell.dataset.col, 10);
          selectCell(idx, col);
        } else {
          selectRow(idx);
        }
      });

      // Double click to edit cell
      tr.addEventListener("dblclick", (e) => {
        const cell = e.target.closest(".grid-cell");
        if (cell) {
          const col = parseInt(cell.dataset.col, 10);
          startCellEdit(idx, col, cell);
        }
      });

      gridTableBody.appendChild(tr);
    });

    highlightActiveCell();
  }

  /**
   * Selects an entire row and highlights on canvas viewer.
   */
  function selectRow(rowIdx) {
    if (!currentStatement || rowIdx < 0 || rowIdx >= currentStatement.transactions.length) return;
    activeRowIndex = rowIdx;

    // Update row selections in table
    const allTrs = gridTableBody.querySelectorAll("tr");
    allTrs.forEach((tr, i) => tr.classList.toggle("selected", i === rowIdx));

    // Highlight on canvas
    const tx = currentStatement.transactions[rowIdx];
    if (tx) {
      if (tx.page && tx.page !== activePageNumber) {
        switchPage(tx.page);
      }
      Viewer.highlightRow(tx.id, tx.bbox);
    }
  }

  /**
   * Selects a specific cell and row.
   */
  function selectCell(rowIdx, colIdx) {
    if (isEditingCell) commitCellEdit();
    activeColIndex = Math.max(1, Math.min(6, colIdx));
    selectRow(rowIdx);
    highlightActiveCell();
  }

  function selectRowById(rowId) {
    if (!currentStatement) return;
    const idx = currentStatement.transactions.findIndex((t) => t.id === rowId);
    if (idx !== -1) {
      selectCell(idx, 2); // default focus on description
      const tr = gridTableBody.querySelector(`tr[data-row-id="${rowId}"]`);
      if (tr) tr.scrollIntoView({ block: "nearest", behavior: "smooth" });
    }
  }

  function highlightActiveCell() {
    const allCells = gridTableBody.querySelectorAll(".active-cell");
    allCells.forEach((c) => c.classList.remove("active-cell"));

    if (activeRowIndex >= 0) {
      const tr = gridTableBody.querySelector(`tr[data-row-index="${activeRowIndex}"]`);
      if (tr) {
        const cell = tr.querySelector(`[data-col="${activeColIndex}"]`);
        if (cell) cell.classList.add("active-cell");
      }
    }
  }

  /**
   * Starts inline editing of a cell.
   */
  function startCellEdit(rowIdx, colIdx, cellElement) {
    if (!currentStatement || rowIdx < 0 || rowIdx >= currentStatement.transactions.length) return;
    if (isEditingCell) commitCellEdit();

    isEditingCell = true;
    activeRowIndex = rowIdx;
    activeColIndex = colIdx;

    const tx = currentStatement.transactions[rowIdx];
    let currentValue = "";
    if (colIdx === 1) currentValue = tx.date;
    else if (colIdx === 2) currentValue = tx.description;
    else if (colIdx === 3) currentValue = (tx.type === "debit") ? Math.abs(parseFloat(tx.amount) || 0.0).toString() : "";
    else if (colIdx === 4) currentValue = (tx.type === "credit") ? Math.abs(parseFloat(tx.amount) || 0.0).toString() : "";
    else if (colIdx === 6) currentValue = tx.balance !== null && tx.balance !== undefined ? tx.balance.toString() : "";

    const input = document.createElement("input");
    input.type = (colIdx === 3 || colIdx === 4 || colIdx === 6) ? "number" : "text";
    if (colIdx === 3 || colIdx === 4 || colIdx === 6) input.step = "0.01";
    input.className = "cell-input";
    input.value = currentValue;

    cellElement.innerHTML = "";
    cellElement.appendChild(input);
    input.focus();
    input.select();

    input.addEventListener("blur", () => {
      if (isEditingCell) commitCellEdit();
    });

    input.addEventListener("keydown", (e) => {
      if (e.key === "Enter") {
        e.preventDefault();
        commitCellEdit();
        // Advance to row below
        if (activeRowIndex < currentStatement.transactions.length - 1) {
          selectCell(activeRowIndex + 1, activeColIndex);
        }
      } else if (e.key === "Tab") {
        e.preventDefault();
        commitCellEdit();
        const nextCol = e.shiftKey ? activeColIndex - 1 : activeColIndex + 1;
        selectCell(activeRowIndex, nextCol);
      } else if (e.key === "Escape") {
        e.preventDefault();
        cancelCellEdit();
      }
    });
  }

  function commitCellEdit() {
    if (!isEditingCell || !currentStatement) return;
    const input = gridTableBody.querySelector(".cell-input");
    if (input) {
      const val = input.value.trim();
      const tx = currentStatement.transactions[activeRowIndex];

      if (activeColIndex === 1) {
        tx.date = val;
      } else if (activeColIndex === 2) {
        tx.description = val;
      } else if (activeColIndex === 3 || activeColIndex === 4) {
        const parsed = parseFloat(val);
        if (!isNaN(parsed)) {
          const isDebitCol = (activeColIndex === 3);
          tx.type = isDebitCol ? "debit" : "credit";
          tx.amount = isDebitCol ? -Math.abs(parsed) : Math.abs(parsed);
        }
      } else if (activeColIndex === 6) {
        const parsed = parseFloat(val);
        tx.balance = isNaN(parsed) ? null : parsed;
      }

      isEditingCell = false;
      renderGrid();
      recalculateReconciliation();
      isDirty = true;
    }
    isEditingCell = false;
  }

  function cancelCellEdit() {
    isEditingCell = false;
    renderGrid();
  }

  /**
   * Toggles transaction between Credit and Debit.
   */
  function toggleTransactionType(rowIdx) {
    if (!currentStatement || rowIdx < 0 || rowIdx >= currentStatement.transactions.length) return;
    const tx = currentStatement.transactions[rowIdx];
    tx.type = (tx.type === "credit") ? "debit" : "credit";
    isDirty = true;
    renderGrid();
    recalculateReconciliation();
    updateViewerBBoxes();
  }

  /**
   * Keyboard Grid Navigation Handler.
   */
  function handleGridKeyDown(e) {
    if (isEditingCell) return;
    if (!currentStatement || currentStatement.transactions.length === 0) return;

    if (e.key === "ArrowUp") {
      e.preventDefault();
      if (activeRowIndex > 0) selectCell(activeRowIndex - 1, activeColIndex);
    } else if (e.key === "ArrowDown") {
      e.preventDefault();
      if (activeRowIndex < currentStatement.transactions.length - 1) selectCell(activeRowIndex + 1, activeColIndex);
    } else if (e.key === "ArrowLeft") {
      e.preventDefault();
      if (activeColIndex > 1) selectCell(activeRowIndex, activeColIndex - 1);
    } else if (e.key === "ArrowRight") {
      e.preventDefault();
      if (activeColIndex < 6) selectCell(activeRowIndex, activeColIndex + 1);
    } else if (e.key === "Enter" || e.key === "F2") {
      e.preventDefault();
      const tr = gridTableBody.querySelector(`tr[data-row-index="${activeRowIndex}"]`);
      if (tr) {
        const cell = tr.querySelector(`[data-col="${activeColIndex}"]`);
        if (cell) startCellEdit(activeRowIndex, activeColIndex, cell);
      }
    } else if (e.key === "Tab") {
      e.preventDefault();
      const nextCol = e.shiftKey ? activeColIndex - 1 : activeColIndex + 1;
      selectCell(activeRowIndex, nextCol);
    } else if (e.key === "Delete" || e.key === "Backspace") {
      if (e.altKey || e.metaKey || e.ctrlKey) {
        deleteActiveRow();
      }
    }
  }

  /**
   * Real-Time Ledger Arithmetic Reconciliation Engine:
   * Opening Balance + Sum(Credits) - Sum(Debits) == Closing Balance +/- 0.01
   */
  function recalculateReconciliation() {
    if (!currentStatement) return;

    const opening = parseFloat(openingBalInput.value) || 0.0;
    const closing = parseFloat(closingBalInput.value) || 0.0;
    currentStatement.opening_balance = opening;
    currentStatement.closing_balance = closing;

    let credits = 0.0;
    let debits = 0.0;

    currentStatement.transactions.forEach((tx) => {
      const amt = Math.abs(parseFloat(tx.amount) || 0.0);
      if (tx.type === "credit") {
        credits += amt;
      } else {
        debits += amt;
      }
    });

    credits = Math.round(credits * 100) / 100;
    debits = Math.round(debits * 100) / 100;
    const calculatedClosing = Math.round((opening + credits - debits) * 100) / 100;
    const difference = Math.round(Math.abs(calculatedClosing - closing) * 100) / 100;
    const isReconciled = difference <= 0.01;

    // Update displays
    totalCreditsDisplay.textContent = `$${credits.toFixed(2)}`;
    totalDebitsDisplay.textContent = `$${debits.toFixed(2)}`;
    calcClosingDisplay.textContent = `$${calculatedClosing.toFixed(2)}`;
    reconDiffDisplay.textContent = `Diff: $${difference.toFixed(2)}`;

    if (isReconciled) {
      reconStatusBadge.className = "recon-badge badge-pass";
      reconStatusBadge.textContent = "RECONCILED (PASS)";
    } else {
      reconStatusBadge.className = "recon-badge badge-fail";
      reconStatusBadge.textContent = "UNBALANCED (FAIL)";
    }

    // Sync with statement reconciliation object
    currentStatement.reconciliation = {
      reconciled: isReconciled,
      opening_balance: opening,
      closing_balance: closing,
      total_credits: credits,
      total_debits: debits,
      calculated_closing: calculatedClosing,
      difference: difference,
      transaction_count: currentStatement.transactions.length,
      tolerance: 0.01
    };
  }

  function onBalanceInputChange() {
    isDirty = true;
    recalculateReconciliation();
  }

  /**
   * Adds a new transaction row.
   */
  function addNewTransactionRow(customData = null) {
    if (!currentStatement) {
      // Create empty session if none exists
      currentStatement = {
        session_id: "manual-" + Math.random().toString(36).substr(2, 9),
        filename: "manual_entry.pdf",
        page_count: 1,
        pages: [],
        opening_balance: 0.0,
        closing_balance: 0.0,
        transactions: [],
        reconciliation: { reconciled: true, opening_balance: 0, closing_balance: 0, total_credits: 0, total_debits: 0, calculated_closing: 0, difference: 0, transaction_count: 0 }
      };
      docInfo.style.display = "flex";
      docFilename.textContent = "manual_entry.pdf";
      docTypeBadge.textContent = "MANUAL";
      exportGroup.style.display = "flex";
    }

    const newTx = {
      id: "tx-" + Math.random().toString(36).substr(2, 8),
      date: customData && customData.date ? customData.date : new Date().toISOString().split("T")[0],
      description: customData && customData.description ? customData.description : "New Transaction",
      amount: customData && customData.amount !== undefined ? Math.abs(customData.amount) : 0.0,
      type: customData && customData.type ? customData.type : "credit",
      balance: null,
      page: customData && customData.page ? customData.page : activePageNumber,
      bbox: customData && customData.bbox ? customData.bbox : null
    };

    currentStatement.transactions.push(newTx);
    isDirty = true;
    renderGrid();
    recalculateReconciliation();
    updateViewerBBoxes();
    selectCell(currentStatement.transactions.length - 1, 2);
    showToast("Added new transaction row.", "success");
  }

  /**
   * Deletes active row.
   */
  function deleteActiveRow() {
    if (activeRowIndex >= 0) {
      deleteRowByIndex(activeRowIndex);
    }
  }

  function deleteRowByIndex(idx) {
    if (!currentStatement || idx < 0 || idx >= currentStatement.transactions.length) return;
    currentStatement.transactions.splice(idx, 1);
    activeRowIndex = Math.min(activeRowIndex, currentStatement.transactions.length - 1);
    isDirty = true;
    renderGrid();
    recalculateReconciliation();
    updateViewerBBoxes();
    showToast("Deleted row.", "success");
  }

  /**
   * Updates active row's field from drag-to-OCR popover.
   */
  function updateActiveRowField(field, value) {
    if (!currentStatement || activeRowIndex < 0 || activeRowIndex >= currentStatement.transactions.length) {
      // If no active row, create new row
      const initData = {};
      initData[field] = value;
      addNewTransactionRow(initData);
      return;
    }

    const tx = currentStatement.transactions[activeRowIndex];
    if (field === "date") {
      tx.date = value;
    } else if (field === "description") {
      tx.description = value;
    } else if (field === "amount") {
      const p = parseFloat(value);
      tx.amount = isNaN(p) ? 0.0 : Math.abs(p);
    }

    isDirty = true;
    renderGrid();
    recalculateReconciliation();
    showToast(`Updated active row ${field}.`, "success");
  }

  function updateViewerBBoxes() {
    if (!currentStatement) return;
    const pageBBoxes = currentStatement.transactions
      .filter((t) => t.page === activePageNumber)
      .map((t) => ({
        id: t.id,
        bbox: t.bbox,
        date_bbox: t.date_bbox,
        desc_bbox: t.desc_bbox,
        amount_bbox: t.amount_bbox,
        type: t.type
      }));
    Viewer.updateBBoxes(pageBBoxes);
  }

  /**
   * Export statement to JSON or Excel (.xlsx).
   * Returns true when the download was generated successfully.
   */
  async function exportStatement(format) {
    if (!currentStatement) {
      showToast("No statement loaded to export.", "error");
      return false;
    }

    showLoading(`Generating ${format.toUpperCase()} Report...`, "Applying multi-tab formatting and reconciliation metadata");

    try {
      const res = await fetch("/api/export", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          statement: currentStatement,
          format: format
        })
      });

      if (!res.ok) throw new Error("Export failed");

      const blob = await res.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `${currentStatement.filename.replace(".pdf", "")}_verified.${format}`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      a.remove();

      lastExportFormat = format;
      isDirty = false;
      showToast(`Exported ${format.toUpperCase()} successfully!`, "success");
      return true;
    } catch (err) {
      showToast(`Export error: ${err.message}`, "error");
      return false;
    } finally {
      hideLoading();
    }
  }

  /**
   * Exit flow: confirm, optionally export, then reset to the initial upload screen.
   */
  function openExitModal() {
    if (!currentStatement) {
      showToast("No active session to exit. Load a statement first.", "info");
      return;
    }
    document.getElementById("exitConfirmOverlay").style.display = "flex";
  }

  function closeExitModal() {
    document.getElementById("exitConfirmOverlay").style.display = "none";
  }

  async function saveAndExit() {
    // Export with the format the user last used (default: xlsx); only clear
    // the session once the download has actually been generated.
    const exported = await exportStatement(lastExportFormat || "xlsx");
    closeExitModal();
    if (!exported) return; // keep session intact so work is not lost
    resetSession();
    showToast("Statement saved. Session cleared.", "success");
  }

  function exitWithoutSaving() {
    closeExitModal();
    resetSession();
    showToast("Session cleared.", "info");
  }

  /**
   * Resets all frontend session state and restores the initial upload screen.
   * The backend SESSIONS dict is left to expire naturally (no delete endpoint).
   */
  function resetSession() {
    currentStatement = null;
    activePageNumber = 1;
    activeRowIndex = -1;
    activeColIndex = 1;
    isEditingCell = false;
    lastExportFormat = null;
    isDirty = false;

    // Header chrome back to initial state
    docInfo.style.display = "none";
    pageNavControls.style.display = "none";
    exportGroup.style.display = "none";

    // Reconciliation dashboard back to defaults
    openingBalInput.value = "0.00";
    closingBalInput.value = "0.00";
    totalCreditsDisplay.textContent = "$0.00";
    totalDebitsDisplay.textContent = "$0.00";
    calcClosingDisplay.textContent = "$0.00";
    reconStatusBadge.className = "recon-badge badge-pass";
    reconStatusBadge.textContent = "RECONCILED";
    reconDiffDisplay.textContent = "Diff: $0.00";

    // Grid & search back to initial empty state
    gridSearchInput.value = "";
    rowCountBadge.textContent = "0 Rows";
    gridTableBody.innerHTML = `
      <tr class="empty-table-row">
        <td colspan="9">No statement loaded. Upload a PDF or click "Load Demo".</td>
      </tr>
    `;

    // Allow re-selecting the same file after a reset
    const fileInput = document.getElementById("pdfFileInput");
    if (fileInput) fileInput.value = "";

    Viewer.reset();
  }

  /**
   * Loads a complete interactive demo bank statement.
   */
  function loadDemoStatement() {
    showLoading("Generating Demo Bank Statement...", "Synthesizing 300 DPI vector pages and transaction ledger");

    // Canvas demo preview image generator
    const canvas = document.createElement("canvas");
    canvas.width = 2480; // Standard 300 DPI A4 page width
    canvas.height = 3508; // Standard 300 DPI A4 page height
    const ctx = canvas.getContext("2d");

    // Draw realistic bank statement document
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, 2480, 3508);

    // Header
    ctx.fillStyle = "#1e293b";
    ctx.font = "bold 54px sans-serif";
    ctx.fillText("FIRST HORIZON COMMERCIAL BANK", 150, 220);

    ctx.fillStyle = "#64748b";
    ctx.font = "32px sans-serif";
    ctx.fillText("Account Statement • Period: Jan 01, 2024 - Jan 31, 2024", 150, 280);
    ctx.fillText("Account #: *******4829 • Currency: USD", 150, 330);

    // Divider
    ctx.strokeStyle = "#cbd5e1";
    ctx.lineWidth = 4;
    ctx.beginPath();
    ctx.moveTo(150, 380);
    ctx.lineTo(2330, 380);
    ctx.stroke();

    // Summary box
    ctx.fillStyle = "#f8fafc";
    ctx.fillRect(150, 420, 2180, 160);
    ctx.strokeRect(150, 420, 2180, 160);

    ctx.fillStyle = "#334155";
    ctx.font = "bold 30px sans-serif";
    ctx.fillText("Opening Balance: $10,450.00", 200, 510);
    ctx.fillText("Closing Balance: $14,642.50", 1400, 510);

    // Table Header
    ctx.fillStyle = "#1e293b";
    ctx.fillRect(150, 640, 2180, 70);
    ctx.fillStyle = "#ffffff";
    ctx.font = "bold 28px sans-serif";
    ctx.fillText("Date", 200, 685);
    ctx.fillText("Description", 550, 685);
    ctx.fillText("Amount ($)", 1850, 685);

    const demoTransactions = [
      { date: "2024-01-03", desc: "PAYROLL DIRECT DEPOSIT ACME CORP", amt: 4800.00, type: "credit", y: 780 },
      { date: "2024-01-05", desc: "OFFICE DEPOT SUPPLIES #8841", amt: 142.50, type: "debit", y: 880 },
      { date: "2024-01-10", desc: "AMAZON WEB SERVICES CLOUD HOSTING", amt: 385.00, type: "debit", y: 980 },
      { date: "2024-01-15", desc: "CLIENT WIRE TRANSFER INVOICE #1092", amt: 2250.00, type: "credit", y: 1080 },
      { date: "2024-01-20", desc: "VERIZON WIRELESS TELECOM AUTOPAY", amt: 180.00, type: "debit", y: 1180 },
      { date: "2024-01-25", desc: "GOOGLE WORKSPACE SUBSCRIPTION", amt: 150.00, type: "debit", y: 1280 },
      { date: "2024-01-29", desc: "EQUIPMENT LEASE PAYMENT", amt: 2000.00, type: "debit", y: 1380 }
    ];

    demoTransactions.forEach((tx, i) => {
      ctx.fillStyle = i % 2 === 0 ? "#ffffff" : "#f8fafc";
      ctx.fillRect(150, tx.y - 45, 2180, 70);
      ctx.strokeStyle = "#e2e8f0";
      ctx.lineWidth = 1;
      ctx.strokeRect(150, tx.y - 45, 2180, 70);

      ctx.fillStyle = "#1e293b";
      ctx.font = "28px sans-serif";
      ctx.fillText(tx.date, 200, tx.y);
      ctx.fillText(tx.desc, 550, tx.y);

      ctx.fillStyle = tx.type === "credit" ? "#166534" : "#991b1b";
      ctx.font = "bold 28px monospace";
      const amtStr = tx.type === "credit" ? `+$${tx.amt.toFixed(2)}` : `-$${tx.amt.toFixed(2)}`;
      ctx.fillText(amtStr, 1850, tx.y);
    });

    const demoImgUrl = canvas.toDataURL("image/jpeg", 0.9);

    const demoPayload = {
      session_id: "demo-session-101",
      filename: "statement_demo_jan2024.pdf",
      page_count: 1,
      pages: [
        {
          page_number: 1,
          page_index: 0,
          type: "vector",
          width: 2480,
          height: 3508,
          image_url: demoImgUrl,
          text_preview: "FIRST HORIZON COMMERCIAL BANK Statement Jan 2024"
        }
      ],
      opening_balance: 10450.00,
      closing_balance: 14642.50,
      transactions: demoTransactions.map((tx, idx) => ({
        id: `demo-${idx + 1}`,
        date: tx.date,
        description: tx.desc,
        amount: tx.amt,
        type: tx.type,
        balance: null,
        page: 1,
        bbox: { x: 150, y: tx.y - 45, width: 2180, height: 70, page: 1 },
        date_bbox: { x: 150, y: tx.y - 45, width: 380, height: 70, page: 1 },
        desc_bbox: { x: 530, y: tx.y - 45, width: 1250, height: 70, page: 1 },
        amount_bbox: { x: 1780, y: tx.y - 45, width: 550, height: 70, page: 1 }
      })),
      reconciliation: {
        reconciled: true,
        opening_balance: 10450.00,
        closing_balance: 14642.50,
        total_credits: 7050.00,
        total_debits: 2857.50,
        calculated_closing: 14642.50,
        difference: 0.00,
        transaction_count: demoTransactions.length,
        tolerance: 0.01
      }
    };

    setTimeout(() => {
      loadStatement(demoPayload);
      hideLoading();
      showToast("Demo statement loaded! Try editing cells or dragging on canvas.", "success");
    }, 400);
  }

  function filterGridRows() {
    renderGrid();
  }

  function showLoading(title, subtitle) {
    loadingTitle.textContent = title;
    loadingSubtitle.textContent = subtitle;
    globalLoading.style.display = "flex";
  }

  function hideLoading() {
    globalLoading.style.display = "none";
  }

  function showToast(message, type = "info") {
    const container = document.getElementById("toastContainer");
    const toast = document.createElement("div");
    toast.className = `toast toast-${type}`;
    toast.textContent = message;
    container.appendChild(toast);
    setTimeout(() => {
      toast.style.opacity = "0";
      setTimeout(() => toast.remove(), 250);
    }, 3500);
  }

  function escapeHtml(str) {
    if (!str) return "";
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");
  }

  // ---------------------------------------------------------------------
  // Snip-first integration API (called by Viewer)
  // ---------------------------------------------------------------------

  function getActiveRowIndex() {
    return activeRowIndex;
  }

  function resolveRowIndex(rowId) {
    if (!currentStatement || !rowId) return -1;
    return currentStatement.transactions.findIndex((t) => t.id === rowId);
  }

  function getPageType(pageIndex) {
    if (!currentStatement || !currentStatement.pages || !currentStatement.pages[pageIndex]) return "vector";
    return currentStatement.pages[pageIndex].type || "vector";
  }

  /**
   * Current UI value of one transaction field, for audit old_value reporting.
   */
  function peekFieldValue(rowIndex, fieldName) {
    if (!currentStatement || rowIndex === null || rowIndex === undefined ||
        rowIndex < 0 || rowIndex >= currentStatement.transactions.length) return "";
    const v = currentStatement.transactions[rowIndex][fieldName];
    return (v === null || v === undefined) ? "" : String(v);
  }

  /**
   * Writes snip-extracted fields into a transaction row (or creates a new
   * row when rowIndex is null/out of range). Returns:
   *   { rowIndex, rowId, changes: [{field_name, old_value, new_value}] }
   * so the caller can append matching rows to the backend audit trail.
   */
  function applySnipFields(rowIndex, fields, meta = {}) {
    if (!fields || Object.keys(fields).length === 0) return null;

    let idx = (rowIndex !== null && rowIndex !== undefined) ? rowIndex : -1;

    if (!currentStatement) {
      // Bootstrap an empty manual session shell
      currentStatement = {
        session_id: "manual-" + Math.random().toString(36).substr(2, 9),
        filename: "manual_entry.pdf",
        page_count: 1,
        pages: [],
        opening_balance: 0.0,
        closing_balance: 0.0,
        transactions: [],
        reconciliation: { reconciled: true, opening_balance: 0, closing_balance: 0, total_credits: 0, total_debits: 0, calculated_closing: 0, difference: 0, transaction_count: 0 }
      };
      docInfo.style.display = "flex";
      docFilename.textContent = "manual_entry.pdf";
      docTypeBadge.textContent = "MANUAL";
      exportGroup.style.display = "flex";
    }

    let isNew = false;
    if (idx < 0 || idx >= currentStatement.transactions.length) {
      const amt = (fields.amount !== undefined && fields.amount !== null) ? parseFloat(fields.amount) : 0.0;
      const safeAmt = isNaN(amt) ? 0.0 : amt;
      const newTx = {
        id: "tx-" + Math.random().toString(36).substr(2, 8),
        date: fields.date || new Date().toISOString().split("T")[0],
        description: fields.description || "New Transaction",
        amount: Math.abs(safeAmt),
        type: safeAmt < 0 ? "debit" : "credit",
        balance: null,
        page: meta.page || activePageNumber,
        bbox: meta.bbox || null
      };
      currentStatement.transactions.push(newTx);
      idx = currentStatement.transactions.length - 1;
      isNew = true;
    }

    const tx = currentStatement.transactions[idx];
    const changes = [];

    if (!isNew) {
      if (fields.date !== undefined && fields.date !== null && fields.date !== tx.date) {
        changes.push({ field_name: "date", old_value: tx.date || "", new_value: String(fields.date) });
        tx.date = fields.date;
      }
      if (fields.description !== undefined && fields.description !== null &&
          String(fields.description) !== (tx.description || "")) {
        changes.push({ field_name: "description", old_value: tx.description || "", new_value: String(fields.description) });
        tx.description = String(fields.description);
      }
      if (fields.amount !== undefined && fields.amount !== null) {
        const parsed = parseFloat(fields.amount);
        if (!isNaN(parsed)) {
          const newAmt = Math.abs(parsed);
          const newType = parsed < 0 ? "debit" : "credit";
          const oldAmtStr = (tx.amount === null || tx.amount === undefined) ? "" : String(tx.amount);
          if (oldAmtStr !== String(newAmt)) {
            changes.push({ field_name: "amount", old_value: oldAmtStr, new_value: String(newAmt) });
          }
          tx.amount = newAmt;
          tx.type = newType;
        }
      }
    }

    isDirty = true;
    renderGrid();
    recalculateReconciliation();
    updateViewerBBoxes();
    selectCell(idx, 2);

    if (isNew) {
      showToast("Added transaction row from snip.", "success");
    } else if (changes.length > 0) {
      showToast(`Applied ${changes.length} field(s) to row ${idx + 1}.`, "success");
    }

    return { rowIndex: idx, rowId: tx.id, changes };
  }

  return {
    init,
    loadStatement,
    selectRowById,
    updateActiveRowField,
    addNewTransactionRow,
    recalculateReconciliation,
    getActiveRowIndex,
    resolveRowIndex,
    getPageType,
    peekFieldValue,
    applySnipFields
  };
})();

document.addEventListener("DOMContentLoaded", () => {
  Editor.init();
});
