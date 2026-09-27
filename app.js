/**
 * app.js — FaceFilter AI Frontend Logic
 * Architecture: Browser webcam → send frames to server → display processed result
 * Works on both localhost AND cloud deployment (Render, Railway, etc.)
 */

// ──────────────────────────────────────────────
// State
// ──────────────────────────────────────────────
let currentFilter = "none";
let isRecording = false;
let mediaRecorder = null;
let recordedChunks = [];
let recTimerInterval = null;
let recSeconds = 0;
let captureCount = 0;
let isProcessing = false;   // Prevent overlapping server calls
let processingLoop = null;

// Filter keyboard shortcuts
const FILTER_KEYS = {
  "1": "none",
  "2": "dog",
  "3": "sunglasses",
  "4": "crown",
  "5": "rainbow",
  "6": "neon",
  "7": "vintage",
  "8": "cyberpunk",
  "9": "mirror",
  "0": "sketch",
  "-": "pixelate",
  "q": "glitch",
  "w": "thermal",
  "e": "nightvision",
  "a": "cartoon",
  "s": "fisheye",
  "d": "disco",
};

// ──────────────────────────────────────────────
// DOM refs
// ──────────────────────────────────────────────
const webcamVideo      = document.getElementById("webcamVideo");
const captureCanvas    = document.getElementById("captureCanvas");
const processedFeed    = document.getElementById("processedFeed");
const filterCards      = document.querySelectorAll(".filter-card");
const btnCapture       = document.getElementById("btnCapture");
const btnRecord        = document.getElementById("btnRecord");
const btnReset         = document.getElementById("btnReset");
const activeFilterBadge= document.getElementById("activeFilterBadge");
const faceCountEl      = document.getElementById("faceCount");
const statusDot        = document.getElementById("statusDot");
const statusText       = document.getElementById("statusText");
const recordTimer      = document.getElementById("recordTimer");
const recTimeEl        = document.getElementById("recTime");
const galleryGrid      = document.getElementById("galleryGrid");
const galleryEmpty     = document.getElementById("galleryEmpty");
const toast            = document.getElementById("toast");

const captureCtx = captureCanvas.getContext("2d");

// ──────────────────────────────────────────────
// Utilities
// ──────────────────────────────────────────────
function showToast(msg, type = "info", duration = 3000) {
  toast.textContent = msg;
  toast.className = `toast show ${type}`;
  clearTimeout(toast._timeout);
  toast._timeout = setTimeout(() => {
    toast.className = "toast";
  }, duration);
}

function setStatus(text, state = "live") {
  statusText.textContent = text;
  statusDot.className = `status-dot ${state}`;
}

function formatTime(s) {
  const m = Math.floor(s / 60).toString().padStart(2, "0");
  const sec = (s % 60).toString().padStart(2, "0");
  return `${m}:${sec}`;
}

// ──────────────────────────────────────────────
// 1. Start browser webcam
// ──────────────────────────────────────────────
async function startWebcam() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
      audio: false,
    });
    webcamVideo.addEventListener("loadedmetadata", () => {
      captureCanvas.width = webcamVideo.videoWidth;
      captureCanvas.height = webcamVideo.videoHeight;
    });
    webcamVideo.srcObject = stream;
    await webcamVideo.play();
    
    // Fallback if loadedmetadata already fired or wasn't caught
    if (webcamVideo.videoWidth) {
      captureCanvas.width = webcamVideo.videoWidth;
      captureCanvas.height = webcamVideo.videoHeight;
    }

    setStatus("Camera active · starting processing…", "live");
    showToast("📷 Camera started!", "success");

    // Start the processing loop
    startProcessingLoop();
  } catch (err) {
    console.error("Webcam error:", err);
    setStatus("Camera denied — check permissions", "error");
    showToast("⚠️ Camera access denied. Please allow camera in your browser.", "error", 5000);
  }
}

// ──────────────────────────────────────────────
// 2. Processing loop: capture frame → send to server → display result
// ──────────────────────────────────────────────
function startProcessingLoop() {
  async function processOnce() {
    if (isProcessing) return; // Skip if previous frame still processing
    if (webcamVideo.readyState < 2) return; // Video not ready

    isProcessing = true;

    try {
      // Draw current webcam frame to hidden canvas
      captureCtx.drawImage(webcamVideo, 0, 0, captureCanvas.width, captureCanvas.height);

      // Convert to base64 JPEG
      const frameData = captureCanvas.toDataURL("image/jpeg", 0.7);

      // Send to server for OpenCV processing
      const res = await fetch("/process_frame", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image: frameData }),
      });

      if (res.ok) {
        const data = await res.json();
        // Display processed frame
        processedFeed.src = data.image;
        // Update face count
        faceCountEl.textContent = data.faces;
        setStatus(`Live · ${data.faces} face(s) detected`, "live");
      }
    } catch (e) {
      // Server might be temporarily unavailable
      setStatus("Processing error — retrying…", "error");
    } finally {
      isProcessing = false;
    }
  }

  // Run at ~12-15 FPS (balanced between smoothness and server load)
  processingLoop = setInterval(processOnce, 70);
}

// ──────────────────────────────────────────────
// 3. Filter selection
// ──────────────────────────────────────────────
async function selectFilter(filterName) {
  try {
    const res = await fetch("/set_filter", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ filter: filterName }),
    });
    if (!res.ok) throw new Error("Server error");

    currentFilter = filterName;

    // Update UI
    filterCards.forEach(card => {
      card.classList.toggle("active", card.dataset.filter === filterName);
    });
    activeFilterBadge.textContent = filterName === "none" ? "NONE" : filterName.toUpperCase();
    activeFilterBadge.style.borderColor = filterName === "none" ? "var(--border)" : "var(--purple)";
    activeFilterBadge.style.color = filterName === "none" ? "var(--text-muted)" : "var(--purple-lt)";

    if (filterName !== "none") {
      showToast(`✨ ${filterName.charAt(0).toUpperCase() + filterName.slice(1)} filter applied!`, "info");
    }
  } catch (e) {
    showToast("⚠️ Could not apply filter", "error");
  }
}

// Attach click handlers to filter cards
filterCards.forEach(card => {
  card.addEventListener("click", () => selectFilter(card.dataset.filter));
});

// ──────────────────────────────────────────────
// 4. Photo capture
// ──────────────────────────────────────────────
function capturePhoto() {
  const src = processedFeed.src;
  if (!src || !src.startsWith("data:")) {
    showToast("⚠️ No processed frame yet", "error");
    return;
  }

  const ts = new Date().toLocaleTimeString();
  captureCount++;

  // Download the processed image
  const a = document.createElement("a");
  a.href = src;
  a.download = `face_filter_${Date.now()}.jpg`;
  a.click();

  // Convert base64 to blob for gallery
  fetch(src)
    .then(r => r.blob())
    .then(blob => {
      const url = URL.createObjectURL(blob);
      addToGallery(url, `Photo ${captureCount} · ${ts}`);
    });

  // Flash effect
  const frame = document.querySelector(".video-frame");
  frame.style.boxShadow = "0 0 0 3px #fff, 0 0 40px rgba(255,255,255,0.5)";
  setTimeout(() => (frame.style.boxShadow = ""), 300);

  showToast("📸 Photo saved!", "success");
}

// ──────────────────────────────────────────────
// 5. Video recording (records the processed output)
// ──────────────────────────────────────────────
function startRecording() {
  const recCanvas = document.createElement("canvas");
  recCanvas.width = 640;
  recCanvas.height = 480;
  const recCtx = recCanvas.getContext("2d");

  // Draw processed frames continuously
  const drawInterval = setInterval(() => {
    try {
      recCtx.drawImage(processedFeed, 0, 0, recCanvas.width, recCanvas.height);
    } catch (e) { /* skip */ }
  }, 33);

  const stream = recCanvas.captureStream(30);
  mediaRecorder = new MediaRecorder(stream, { mimeType: "video/webm;codecs=vp9" });
  recordedChunks = [];

  mediaRecorder.ondataavailable = e => {
    if (e.data && e.data.size > 0) recordedChunks.push(e.data);
  };

  mediaRecorder.onstop = () => {
    clearInterval(drawInterval);
    const blob = new Blob(recordedChunks, { type: "video/webm" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `face_filter_video_${Date.now()}.webm`;
    a.click();
    showToast("🎬 Video saved!", "success");
  };

  mediaRecorder.start(100);
  isRecording = true;

  recSeconds = 0;
  recordTimer.style.display = "flex";
  recTimerInterval = setInterval(() => {
    recSeconds++;
    recTimeEl.textContent = formatTime(recSeconds);
  }, 1000);

  btnRecord.classList.add("recording");
  btnRecord.querySelector(".btn-icon").textContent = "⏹";
  btnRecord.querySelector(".btn-label").textContent = "Stop Recording";
  showToast("🔴 Recording started…", "info");
}

function stopRecording() {
  if (mediaRecorder && mediaRecorder.state !== "inactive") {
    mediaRecorder.stop();
  }
  isRecording = false;
  clearInterval(recTimerInterval);
  recordTimer.style.display = "none";

  btnRecord.classList.remove("recording");
  btnRecord.querySelector(".btn-icon").textContent = "🔴";
  btnRecord.querySelector(".btn-label").textContent = "Record Video";
}

// ──────────────────────────────────────────────
// 6. Gallery
// ──────────────────────────────────────────────
function addToGallery(url, label) {
  if (galleryEmpty) galleryEmpty.style.display = "none";

  const item = document.createElement("div");
  item.className = "gallery-item";

  const img = document.createElement("img");
  img.src = url;
  img.alt = label;

  const overlay = document.createElement("div");
  overlay.className = "gallery-item-overlay";
  const lbl = document.createElement("span");
  lbl.className = "gallery-item-label";
  lbl.textContent = label;
  overlay.appendChild(lbl);

  const dlBtn = document.createElement("button");
  dlBtn.className = "gallery-download-btn";
  dlBtn.textContent = "⬇ Save";
  dlBtn.addEventListener("click", e => {
    e.stopPropagation();
    const a = document.createElement("a");
    a.href = url;
    a.download = `face_filter_${Date.now()}.jpg`;
    a.click();
  });

  item.addEventListener("click", () => window.open(url, "_blank"));

  item.appendChild(img);
  item.appendChild(overlay);
  item.appendChild(dlBtn);
  galleryGrid.prepend(item);
}

// ──────────────────────────────────────────────
// 7. Event listeners
// ──────────────────────────────────────────────
btnCapture.addEventListener("click", capturePhoto);

btnRecord.addEventListener("click", () => {
  if (isRecording) stopRecording();
  else startRecording();
});

btnReset.addEventListener("click", () => selectFilter("none"));

// Keyboard shortcuts
document.addEventListener("keydown", e => {
  if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;

  const key = e.key;

  if (key === " ") {
    e.preventDefault();
    capturePhoto();
  } else if (key.toLowerCase() === "r") {
    if (isRecording) stopRecording();
    else startRecording();
  } else if (key === "Escape") {
    selectFilter("none");
  } else if (FILTER_KEYS[key]) {
    selectFilter(FILTER_KEYS[key]);
    const card = document.querySelector(`[data-filter="${FILTER_KEYS[key]}"]`);
    if (card) {
      card.style.transform = "scale(0.95)";
      setTimeout(() => (card.style.transform = ""), 150);
    }
  }
});

// ──────────────────────────────────────────────
// 8. Initialize
// ──────────────────────────────────────────────
setStatus("Requesting camera access…");
showToast("🎭 FaceFilter AI — requesting camera access…", "info", 4000);
startWebcam();
