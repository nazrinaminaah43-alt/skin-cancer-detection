/**
 * Skin Cancer Detection Dashboard - Client Application Logic
 * Integrates image upload, OpenCV preview, REST API communication,
 * dynamic metrics rendering, and prediction history management.
 * 
 * ARCHITECTURE PRINCIPLE:
 * The frontend dashboard has NO direct connection to the ML model.
 * All machine learning inference, OpenCV processing, and data persistence
 * are strictly sequestered behind the Flask REST API Gateway.
 */

// REST API Gateway Base URL (Decoupled client architecture)
const API_BASE_URL = (window.location.protocol === "file:" || !window.location.port)
    ? "http://127.0.0.1:5000"
    : window.location.origin;

document.addEventListener("DOMContentLoaded", () => {
    // DOM Elements
    const dropzone = document.getElementById("dropzone");
    const fileInput = document.getElementById("fileInput");
    const dropzonePlaceholder = document.getElementById("dropzonePlaceholder");
    const previewContainer = document.getElementById("previewContainer");
    const imagePreview = document.getElementById("imagePreview");
    const contourPreview = document.getElementById("contourPreview");
    const toggleContourBtn = document.getElementById("toggleContourBtn");
    const removeImageBtn = document.getElementById("removeImageBtn");
    const uploadBtn = document.getElementById("uploadBtn");
    const predictBtn = document.getElementById("predictBtn");
    const predictBtnText = document.getElementById("predictBtnText");
    const statusAlert = document.getElementById("statusAlert");
    const statusMessage = document.getElementById("statusMessage");
    const samplesContainer = document.getElementById("samplesContainer");

    // Result Card Elements
    const cancerVerdictCallout = document.getElementById("cancerVerdictCallout");
    const verdictIcon = document.getElementById("verdictIcon");
    const verdictTitle = document.getElementById("verdictTitle");
    const verdictDesc = document.getElementById("verdictDesc");
    const resCancerAnswer = document.getElementById("resCancerAnswer");
    const resSkinVerified = document.getElementById("resSkinVerified");
    const resPrediction = document.getElementById("resPrediction");
    const resModel = document.getElementById("resModel");
    const resConfidence = document.getElementById("resConfidence");
    const resultBadge = document.getElementById("resultBadge");
    const confidenceBarWrapper = document.getElementById("confidenceBarWrapper");
    const confidenceBarFill = document.getElementById("confidenceBarFill");
    const probBenignText = document.getElementById("probBenignText");
    const probMalignantText = document.getElementById("probMalignantText");
    const featuresSummaryBox = document.getElementById("featuresSummaryBox");
    const featAsym = document.getElementById("featAsym");
    const featCompact = document.getElementById("featCompact");
    const featIrreg = document.getElementById("featIrreg");
    const featColorVar = document.getElementById("featColorVar");
    const featContrast = document.getElementById("featContrast");
    const featSharpness = document.getElementById("featSharpness");

    // Model Performance Elements
    const metricAccuracy = document.getElementById("metricAccuracy");
    const metricPrecision = document.getElementById("metricPrecision");
    const metricRecall = document.getElementById("metricRecall");
    const metricF1 = document.getElementById("metricF1");
    const cmTN = document.getElementById("cmTN");
    const cmFP = document.getElementById("cmFP");
    const cmFN = document.getElementById("cmFN");
    const cmTP = document.getElementById("cmTP");
    const btnShowMatrixGrid = document.getElementById("btnShowMatrixGrid");
    const btnShowHeatmap = document.getElementById("btnShowHeatmap");
    const cmGridWrapper = document.getElementById("cmGridWrapper");
    const cmHeatmapWrapper = document.getElementById("cmHeatmapWrapper");
    const cmHeatmapImg = document.getElementById("cmHeatmapImg");

    // History Elements
    const historyTableBody = document.getElementById("historyTableBody");
    const historyCountBadge = document.getElementById("historyCountBadge");
    const refreshHistoryBtn = document.getElementById("refreshHistoryBtn");
    const clearHistoryBtn = document.getElementById("clearHistoryBtn");
    const toastContainer = document.getElementById("toastContainer");

    // Clinical Disclaimer Dismissal
    const disclaimerContainer = document.getElementById("disclaimerContainer");
    const dismissDisclaimerBtn = document.getElementById("dismissDisclaimerBtn");

    if (dismissDisclaimerBtn && disclaimerContainer) {
        if (localStorage.getItem("disclaimer_dismissed") === "true") {
            disclaimerContainer.style.display = "none";
        }

        dismissDisclaimerBtn.addEventListener("click", () => {
            disclaimerContainer.style.opacity = "0";
            disclaimerContainer.style.transform = "translateY(-10px)";
            disclaimerContainer.style.transition = "all 0.25s ease";
            setTimeout(() => {
                disclaimerContainer.style.display = "none";
                localStorage.setItem("disclaimer_dismissed", "true");
            }, 250);
        });
    }

    const showDisclaimerBtn = document.getElementById("showDisclaimerBtn");
    if (showDisclaimerBtn && disclaimerContainer) {
        showDisclaimerBtn.addEventListener("click", () => {
            disclaimerContainer.style.display = "block";
            disclaimerContainer.style.opacity = "1";
            disclaimerContainer.style.transform = "translateY(0)";
            localStorage.removeItem("disclaimer_dismissed");
            disclaimerContainer.scrollIntoView({ behavior: "smooth" });
        });
    }

    // State Variables
    let selectedFile = null;
    let selectedSampleId = null;
    let currentOverlayUrl = null;
    let showingContour = false;

    // ==========================================================================
    // 1. Initial Data Fetching: Model Metrics, Curated Samples & History
    // ==========================================================================
    loadModelPerformance();
    loadCuratedSamples();
    loadPredictionHistory();

    // ==========================================================================
    // 2. Drag & Drop and File Selection Handlers
    // ==========================================================================
    uploadBtn.addEventListener("click", () => fileInput.click());
    dropzone.addEventListener("click", (e) => {
        if (e.target !== toggleContourBtn && e.target !== removeImageBtn && !toggleContourBtn.contains(e.target)) {
            fileInput.click();
        }
    });

    fileInput.addEventListener("change", (e) => {
        if (e.target.files && e.target.files[0]) {
            handleFileSelection(e.target.files[0]);
        }
    });

    // Drag-over styling
    ["dragenter", "dragover"].forEach(evt => {
        dropzone.addEventListener(evt, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.add("dragover");
        });
    });

    ["dragleave", "drop"].forEach(evt => {
        dropzone.addEventListener(evt, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropzone.classList.remove("dragover");
        });
    });

    dropzone.addEventListener("drop", (e) => {
        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            handleFileSelection(e.dataTransfer.files[0]);
        }
    });

    function handleFileSelection(file) {
        // Validate format
        const validExtensions = ["image/jpeg", "image/png", "image/webp", "image/bmp"];
        if (!validExtensions.includes(file.type) && !file.name.match(/\.(jpe?g|png|webp|bmp)$/i)) {
            showToast("Invalid format. Please select a JPEG, PNG, or WEBP image.", "error");
            return;
        }

        // Validate size (16MB)
        if (file.size > 16 * 1024 * 1024) {
            showToast("File exceeds 16MB limit.", "error");
            return;
        }

        selectedFile = file;
        selectedSampleId = null;

        // Deselect any active sample chips
        document.querySelectorAll(".sample-chip").forEach(chip => chip.classList.remove("active"));

        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            imagePreview.style.display = "block";
            contourPreview.style.display = "none";
            dropzonePlaceholder.style.display = "none";
            previewContainer.style.display = "flex";
            toggleContourBtn.style.display = "none";
            predictBtn.disabled = false;
            resetResultCard();
            showToast(`Loaded ${file.name}`, "info");
        };
        reader.readAsDataURL(file);
    }

    removeImageBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        resetInput();
    });

    function resetInput() {
        selectedFile = null;
        selectedSampleId = null;
        currentOverlayUrl = null;
        showingContour = false;
        fileInput.value = "";
        imagePreview.src = "";
        contourPreview.src = "";
        dropzonePlaceholder.style.display = "block";
        previewContainer.style.display = "none";
        predictBtn.disabled = true;
        document.querySelectorAll(".sample-chip").forEach(chip => chip.classList.remove("active"));
        resetResultCard();
    }

    // Toggle OpenCV segmented contour overlay
    toggleContourBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        if (!currentOverlayUrl) return;

        showingContour = !showingContour;
        if (showingContour) {
            contourPreview.src = currentOverlayUrl;
            imagePreview.style.display = "none";
            contourPreview.style.display = "block";
            toggleContourBtn.innerHTML = '<span class="chip-dot"></span> View Original Image';
        } else {
            imagePreview.style.display = "block";
            contourPreview.style.display = "none";
            toggleContourBtn.innerHTML = '<span class="chip-dot"></span> View Lesion Boundary';
        }
    });

    // ==========================================================================
    // 3. Model Prediction Pipeline
    // ==========================================================================
    predictBtn.addEventListener("click", async () => {
        if (!selectedFile && !selectedSampleId) {
            showToast("Please upload an image or choose a sample lesion.", "error");
            return;
        }

        // UI Loading State
        predictBtn.disabled = true;
        predictBtnText.textContent = "Analyzing...";
        statusAlert.style.display = "flex";
        statusMessage.textContent = "Validating image format & size...";

        const formData = new FormData();
        if (selectedFile) {
            formData.append("image", selectedFile);
        } else if (selectedSampleId) {
            formData.append("sample_id", selectedSampleId);
        }

        try {
            // Simulated visual step updates for clinical responsiveness
            setTimeout(() => {
                if (statusMessage) statusMessage.textContent = "OpenCV: DullRazor hair removal & CLAHE contrast...";
            }, 300);
            setTimeout(() => {
                if (statusMessage) statusMessage.textContent = "OpenCV: Otsu lesion contour segmentation...";
            }, 600);
            setTimeout(() => {
                if (statusMessage) statusMessage.textContent = "Extracting ABCD features & SVM RBF inference...";
            }, 900);

            const response = await fetch(`${API_BASE_URL}/api/predict`, {
                method: "POST",
                body: formData
            });

            const data = await response.json();

            if (!response.ok || !data.success) {
                throw new Error(data.error || "Prediction request failed.");
            }

            // Populate Result Card with API data
            displayPredictionResult(data);
            showToast(`Analysis complete: ${data.prediction} (${data.confidence_percentage})`, "success");

            // Refresh History Table
            loadPredictionHistory();

        } catch (error) {
            console.error("Prediction error:", error);
            showToast(error.message, "error");
            resPrediction.textContent = "Error";
            resConfidence.textContent = "—";
        } finally {
            predictBtn.disabled = false;
            predictBtnText.textContent = "Predict";
            statusAlert.style.display = "none";
        }
    });

    function displayPredictionResult(data) {
        const isBenign = data.prediction.toUpperCase() === "BENIGN";
        const isCancer = data.is_cancer !== undefined ? data.is_cancer : !isBenign;

        // Prominent Cancer Verdict Callout
        if (cancerVerdictCallout) {
            cancerVerdictCallout.style.display = "flex";
            if (isCancer) {
                cancerVerdictCallout.className = "cancer-verdict-callout verdict-malignant";
                verdictIcon.textContent = "⚠️";
                verdictTitle.textContent = "SKIN CANCER DETECTED";
                verdictDesc.textContent = data.cancer_explanation || "The lesion exhibits features consistent with malignant skin cancer. Prompt dermatological evaluation advised.";
                if (resCancerAnswer) {
                    resCancerAnswer.textContent = "YES — Potential Skin Cancer Detected";
                    resCancerAnswer.style.color = "var(--malignant-color)";
                }
            } else {
                cancerVerdictCallout.className = "cancer-verdict-callout verdict-benign";
                verdictIcon.textContent = "🛡️";
                verdictTitle.textContent = "NOT SKIN CANCER";
                verdictDesc.textContent = data.cancer_explanation || "The lesion displays non-cancerous, benign characteristics.";
                if (resCancerAnswer) {
                    resCancerAnswer.textContent = "NO — Not Skin Cancer (Benign Mole)";
                    resCancerAnswer.style.color = "var(--benign-color)";
                }
            }
        }

        // Skin Image Verification Status
        if (resSkinVerified) {
            const hasSkin = data.is_skin !== false;
            resSkinVerified.innerHTML = hasSkin
                ? `<span style="color: var(--benign-color); font-weight: 700;">✅ Yes (Human Skin Lesion)</span>`
                : `<span style="color: var(--warning-color); font-weight: 700;">⚠️ Low skin tone confidence</span>`;
        }

        // Prediction & Model
        resPrediction.textContent = data.prediction;
        resModel.textContent = `${data.model_used}`;
        resConfidence.textContent = `${data.confidence_percentage}`;

        // Badge styling
        resultBadge.textContent = data.prediction;
        resultBadge.className = `status-pill ${isBenign ? "status-benign" : "status-malignant"}`;

        // Result text coloring
        resPrediction.style.color = isBenign ? "var(--benign-color)" : "var(--malignant-color)";
        resConfidence.style.color = isBenign ? "var(--benign-color)" : "var(--malignant-color)";

        // Probabilities Bar
        if (data.probabilities) {
            confidenceBarWrapper.style.display = "block";
            probBenignText.textContent = `Benign: ${data.probabilities.benign}%`;
            probMalignantText.textContent = `Malignant: ${data.probabilities.malignant}%`;
            
            const fillWidth = isBenign ? data.probabilities.benign : data.probabilities.malignant;
            confidenceBarFill.style.width = `${fillWidth}%`;
            confidenceBarFill.className = `confidence-bar-fill ${isBenign ? "benign" : "malignant"}`;
        }

        // Extracted ABCD Features Summary
        if (data.feature_summary) {
            featuresSummaryBox.style.display = "block";
            featAsym.textContent = data.feature_summary.asymmetry_index !== undefined ? data.feature_summary.asymmetry_index : "—";
            featCompact.textContent = data.feature_summary.border_compactness !== undefined ? data.feature_summary.border_compactness : "—";
            featIrreg.textContent = data.feature_summary.border_irregularity !== undefined ? data.feature_summary.border_irregularity : "—";
            featColorVar.textContent = data.feature_summary.color_variegation !== undefined ? data.feature_summary.color_variegation : "—";
            featContrast.textContent = data.feature_summary.texture_contrast !== undefined ? data.feature_summary.texture_contrast : "—";
            featSharpness.textContent = data.feature_summary.texture_sharpness !== undefined ? data.feature_summary.texture_sharpness : "—";
        }

        // Enable OpenCV segmented contour overlay toggle
        if (data.image_info && data.image_info.segmented_url) {
            currentOverlayUrl = data.image_info.segmented_url.startsWith("http")
                ? data.image_info.segmented_url
                : `${API_BASE_URL}${data.image_info.segmented_url}`;
            toggleContourBtn.style.display = "inline-flex";
            showingContour = false;
            toggleContourBtn.innerHTML = '<span class="chip-dot"></span> View Lesion Boundary';
        }
    }

    function resetResultCard() {
        if (cancerVerdictCallout) cancerVerdictCallout.style.display = "none";
        if (resCancerAnswer) {
            resCancerAnswer.textContent = "—";
            resCancerAnswer.style.color = "var(--text-primary)";
        }
        if (resSkinVerified) resSkinVerified.textContent = "—";
        resPrediction.textContent = "—";
        resPrediction.style.color = "var(--text-primary)";
        resModel.textContent = "SVM (RBF Kernel)";
        resConfidence.textContent = "—";
        resConfidence.style.color = "var(--text-primary)";
        resultBadge.textContent = "Awaiting Input";
        resultBadge.className = "status-pill status-ready";
        confidenceBarWrapper.style.display = "none";
        featuresSummaryBox.style.display = "none";
    }

    // ==========================================================================
    // 4. Model Performance: Dynamically Loaded from Backend
    // ==========================================================================
    async function loadModelPerformance() {
        try {
            const resp = await fetch(`${API_BASE_URL}/api/model-info`);
            const data = await resp.json();

            if (!resp.ok || !data.success) {
                throw new Error("Could not retrieve model metrics");
            }

            const metrics = data.model_info;

            // Populate Performance Numbers dynamically (no hardcoded values!)
            metricAccuracy.textContent = metrics.accuracy_percentage || `${(metrics.accuracy * 100).toFixed(1)}%`;
            metricPrecision.textContent = metrics.precision_percentage || `${(metrics.precision * 100).toFixed(1)}%`;
            metricRecall.textContent = metrics.recall_percentage || `${(metrics.recall * 100).toFixed(1)}%`;
            metricF1.textContent = metrics.f1_score_percentage || `${(metrics.f1_score * 100).toFixed(1)}%`;

            // Populate Confusion Matrix
            if (metrics.confusion_matrix) {
                cmTN.textContent = metrics.confusion_matrix.true_negatives;
                cmFP.textContent = metrics.confusion_matrix.false_positives;
                cmFN.textContent = metrics.confusion_matrix.false_negatives;
                cmTP.textContent = metrics.confusion_matrix.true_positives;
            }

            // Refresh heatmap image with timestamp cache buster
            if (cmHeatmapImg) {
                cmHeatmapImg.src = `${API_BASE_URL}/static/img/confusion_matrix.png?t=${Date.now()}`;
            }

        } catch (err) {
            console.error("Failed to load model metrics:", err);
            metricAccuracy.textContent = "N/A";
            metricPrecision.textContent = "N/A";
            metricRecall.textContent = "N/A";
            metricF1.textContent = "N/A";
        }
    }

    // Toggle Matrix Grid vs Seaborn Plot
    btnShowMatrixGrid.addEventListener("click", () => {
        btnShowMatrixGrid.classList.add("active");
        btnShowHeatmap.classList.remove("active");
        cmGridWrapper.style.display = "block";
        cmHeatmapWrapper.style.display = "none";
    });

    btnShowHeatmap.addEventListener("click", () => {
        btnShowHeatmap.classList.add("active");
        btnShowMatrixGrid.classList.remove("active");
        cmGridWrapper.style.display = "none";
        cmHeatmapWrapper.style.display = "block";
    });

    // ==========================================================================
    // 5. Curated Samples: 1-Click Testing Support
    // ==========================================================================
    async function loadCuratedSamples() {
        try {
            const resp = await fetch(`${API_BASE_URL}/api/samples`);
            const data = await resp.json();

            if (!resp.ok || !data.success || !data.samples || data.samples.length === 0) {
                samplesContainer.innerHTML = '<div class="sample-loading">No sample images found.</div>';
                return;
            }

            samplesContainer.innerHTML = "";
            data.samples.forEach(sample => {
                const chip = document.createElement("button");
                chip.type = "button";
                chip.className = "sample-chip";
                const isBenign = sample.type.toLowerCase() === "benign";
                const sampleImgUrl = sample.relative_url.startsWith("http")
                    ? sample.relative_url
                    : `${API_BASE_URL}${sample.relative_url}`;

                chip.innerHTML = `
                    <img src="${sampleImgUrl}" alt="${sample.title}" class="sample-thumb">
                    <div class="sample-info">
                        <span class="sample-title" title="${sample.title}">${sample.title}</span>
                        <span class="sample-type ${isBenign ? 'benign' : 'malignant'}">${sample.type}</span>
                    </div>
                `;

                chip.addEventListener("click", () => {
                    // Activate this chip
                    document.querySelectorAll(".sample-chip").forEach(c => c.classList.remove("active"));
                    chip.classList.add("active");

                    selectedFile = null;
                    selectedSampleId = sample.filename;

                    imagePreview.src = sampleImgUrl;
                    imagePreview.style.display = "block";
                    contourPreview.style.display = "none";
                    dropzonePlaceholder.style.display = "none";
                    previewContainer.style.display = "flex";
                    toggleContourBtn.style.display = "none";
                    predictBtn.disabled = false;
                    resetResultCard();

                    showToast(`Selected sample: ${sample.title}`, "info");
                });

                samplesContainer.appendChild(chip);
            });

        } catch (err) {
            console.error("Failed to load sample images:", err);
            samplesContainer.innerHTML = '<div class="sample-loading">Failed to load samples.</div>';
        }
    }

    // ==========================================================================
    // 6. Prediction History Table & Management
    // ==========================================================================
    async function loadPredictionHistory() {
        try {
            const resp = await fetch(`${API_BASE_URL}/api/history?limit=30`);
            const data = await resp.json();

            if (!resp.ok || !data.success) {
                throw new Error("Failed to load history.");
            }

            const records = data.history || [];
            historyCountBadge.textContent = `${records.length} Records`;

            if (records.length === 0) {
                historyTableBody.innerHTML = `
                    <tr>
                        <td colspan="8" class="empty-table-cell">No predictions logged yet. Upload an image above to run screening.</td>
                    </tr>
                `;
                return;
            }

            historyTableBody.innerHTML = "";
            records.forEach(rec => {
                const isBenign = rec.prediction.toUpperCase() === "BENIGN";
                const isCancer = rec.is_cancer !== undefined ? rec.is_cancer : !isBenign;
                const thumbUrl = rec.image_url.startsWith("http") ? rec.image_url : `${API_BASE_URL}${rec.image_url}`;
                const tr = document.createElement("tr");

                tr.innerHTML = `
                    <td>
                        <img src="${thumbUrl}" alt="Thumbnail" class="history-thumb" onerror="this.src='${API_BASE_URL}/static/img/placeholder.png'">
                    </td>
                    <td><code style="font-family: var(--font-mono); font-size: 0.78rem;">${escapeHtml(rec.image_identifier)}</code></td>
                    <td>
                        <span class="${isCancer ? 'badge-cancer-yes' : 'badge-cancer-no'}">
                            ${isCancer ? '🔴 SKIN CANCER' : '🟢 NOT CANCER'}
                        </span>
                    </td>
                    <td>
                        <span class="status-pill ${isBenign ? 'status-benign' : 'status-malignant'}">${rec.prediction}</span>
                    </td>
                    <td><strong>${rec.confidence_percentage}</strong></td>
                    <td>${escapeHtml(rec.model_name)}</td>
                    <td style="color: var(--text-muted); font-size: 0.78rem;">${rec.timestamp}</td>
                    <td>
                        <button type="button" class="btn-icon-delete" data-id="${rec.id}" title="Delete Record">
                            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <polyline points="3 6 5 6 21 6"/>
                                <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>
                            </svg>
                        </button>
                    </td>
                `;

                // Wire individual delete button
                const delBtn = tr.querySelector(".btn-icon-delete");
                delBtn.addEventListener("click", () => deleteHistoryItem(rec.id));

                historyTableBody.appendChild(tr);
            });

        } catch (err) {
            console.error("Failed to load history:", err);
        }
    }

    async function deleteHistoryItem(recordId) {
        if (!confirm(`Delete prediction record #${recordId}?`)) return;

        try {
            const resp = await fetch(`${API_BASE_URL}/api/history/${recordId}`, { method: "DELETE" });
            const data = await resp.json();

            if (!resp.ok || !data.success) {
                throw new Error(data.error || "Failed to delete record.");
            }

            showToast("Record deleted successfully", "info");
            loadPredictionHistory();
        } catch (err) {
            showToast(err.message, "error");
        }
    }

    clearHistoryBtn.addEventListener("click", async () => {
        if (!confirm("Are you sure you want to clear all prediction history? This action cannot be undone.")) {
            return;
        }

        try {
            const resp = await fetch(`${API_BASE_URL}/api/history`, { method: "DELETE" });
            const data = await resp.json();

            if (!resp.ok || !data.success) {
                throw new Error(data.error || "Failed to clear history.");
            }

            showToast(data.message || "All history cleared", "info");
            loadPredictionHistory();
        } catch (err) {
            showToast(err.message, "error");
        }
    });

    refreshHistoryBtn.addEventListener("click", () => {
        loadPredictionHistory();
        showToast("History refreshed", "info");
    });

    // ==========================================================================
    // 7. Toast Notification Utility
    // ==========================================================================
    function showToast(message, type = "info") {
        const toast = document.createElement("div");
        toast.className = `toast ${type === "error" ? "toast-error" : type === "success" ? "toast-success" : ""}`;
        
        const iconSvg = type === "error" ? "⚠️" : type === "success" ? "✓" : "ℹ️";
        toast.innerHTML = `<span>${iconSvg}</span> <span>${escapeHtml(message)}</span>`;

        toastContainer.appendChild(toast);

        setTimeout(() => {
            toast.style.opacity = "0";
            toast.style.transform = "translateY(20px)";
            toast.style.transition = "all 0.3s ease";
            setTimeout(() => toast.remove(), 300);
        }, 3500);
    }

    function escapeHtml(text) {
        if (!text) return "";
        return text
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#039;");
    }
});
