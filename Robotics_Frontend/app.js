// ==================== CONFIG ====================
const BASE_URL = "http://127.0.0.1:8000";

// ==================== LANDING PAGE ====================
const landingPage = document.getElementById("landing-page");
const landingStartBtn = document.getElementById("landing-start-btn");

landingStartBtn.addEventListener("click", () => {
  landingPage.classList.add("fade-out");
  setTimeout(() => {
    landingPage.style.display = "none";
    // Scroll gently to the project intro
    const project = document.getElementById("project");
    if (project) project.scrollIntoView({ behavior: "smooth" });
  }, 550);
});

// ==================== NAV ACTIVE ON SCROLL ====================
const sections = ["project", "robot-app"];
const navLinks = document.querySelectorAll(".nav-item");

window.addEventListener("scroll", () => {
  let active = sections[0];

  sections.forEach((id) => {
    const el = document.getElementById(id);
    if (el && window.scrollY + 260 >= el.offsetTop) {
      active = id;
    }
  });

  navLinks.forEach((link) => {
    const href = link.getAttribute("href");
    link.classList.toggle("active", href === "#" + active);
  });
});

// ==================== DETECTION UI ====================
let detectionActive = false;
let statusInterval = null;

// DOM refs
const previewEl = document.getElementById("detector-preview");
const imgEl = document.getElementById("detector-stream");
const phEl = document.getElementById("detector-placeholder");
const startBtn = document.getElementById("detector-start");
const stopBtn = document.getElementById("detector-stop");
const statusText = document.getElementById("detection-status");

const robotStatus = document.getElementById("robot-status");
const currentItemName = document.getElementById("current-item-name");
const currentItemConfidence = document.getElementById("current-item-confidence");
const helperMessage = document.getElementById("helper-message");

// ---- Start detection ----
async function startDetection() {
  // Set up camera stream
  imgEl.src = `${BASE_URL}/video?ts=${Date.now()}`;

  // Ask backend to start detection
  try {
    await fetch(`${BASE_URL}/detection/start`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
    });
  } catch (err) {
    console.error("Error starting detection:", err);
    statusText.textContent = "Backend not reachable. Make sure the robot server is running.";
    statusText.style.color = "#ef4444";
    robotStatus.textContent = "Robot not connected. Please ask someone to restart the system.";
    helperMessage.textContent = "Once the system is restarted, press Start Sorting again.";
    return;
  }

  // Stream appearance
  imgEl.onload = () => {
    imgEl.classList.remove("hidden");
    imgEl.classList.add("stream-visible");
    phEl.classList.add("hidden");
    previewEl.classList.add("streaming");
  };

  imgEl.onerror = () => {
    statusText.textContent = "Cannot connect to camera. Check that the robot server is running.";
    statusText.style.color = "#ef4444";
    startBtn.disabled = false;
    stopBtn.disabled = true;
  };

  startBtn.disabled = true;
  stopBtn.disabled = false;
  detectionActive = true;

  statusText.textContent = "Detection running. Place one grocery item in the camera area.";
  statusText.style.color = "#6b7280";
  robotStatus.textContent = "Robot is ready. Please place one grocery item in the marked area.";
  helperMessage.textContent = "Place a single item under the camera now.";

  startStatusPolling();
}

// ---- Stop detection ----
async function stopDetection() {
  try {
    await fetch(`${BASE_URL}/detection/stop`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
    });
  } catch (err) {
    console.error("Error stopping detection:", err);
  }

  imgEl.src = "";
  imgEl.classList.add("hidden");
  imgEl.classList.remove("stream-visible");
  phEl.classList.remove("hidden");
  previewEl.classList.remove("streaming");

  startBtn.disabled = false;
  stopBtn.disabled = true;
  detectionActive = false;

  statusText.textContent = "Robot paused. Press “Start Sorting” to continue.";
  statusText.style.color = "#6b7280";
  robotStatus.textContent = "Robot is paused.";
  helperMessage.textContent = "When you want to continue, press Start Sorting.";

  stopStatusPolling();
}

startBtn.addEventListener("click", startDetection);
stopBtn.addEventListener("click", stopDetection);

// ==================== STATUS POLLING ====================
function startStatusPolling() {
  if (statusInterval) return;

  statusInterval = setInterval(async () => {
    try {
      const res = await fetch(`${BASE_URL}/detection/status`);
      const data = await res.json();

      if (data.detections && data.detections.length > 0) {
        const top = data.top_detection || data.detections[0];
        const cls = top.cls || "Unknown item";
        const conf = top.conf != null ? Math.round(top.conf * 100) : null;

        const prettyName = cls.charAt(0).toUpperCase() + cls.slice(1);

        currentItemName.textContent = prettyName;
        currentItemConfidence.textContent = conf ? `${conf}% sure` : "Confidence not available";

        statusText.textContent = `Detected: ${prettyName} (${currentItemConfidence.textContent}).`;
        statusText.style.color = "#10b981";
        robotStatus.textContent = "Robot is sorting this item now. Please keep your hands away.";
        helperMessage.textContent =
          "Please wait. Do not place another item until the area is empty again.";
      } else {
        currentItemName.textContent = "None at the moment";
        currentItemConfidence.textContent = "–";

        if (detectionActive) {
          statusText.textContent = "Detection running. Waiting for a grocery item…";
          statusText.style.color = "#6b7280";
          robotStatus.textContent = "Robot is ready and waiting.";
          helperMessage.textContent =
            "Place one grocery item in the marked area under the camera.";
        }
      }
    } catch (err) {
      console.error("Error fetching detection status:", err);
      statusText.textContent = "Could not read robot status. Check the server connection.";
      statusText.style.color = "#ef4444";
      robotStatus.textContent = "Robot connection problem.";
      helperMessage.textContent =
        "If this message stays for long, please ask someone to restart the robot system.";
    }
  }, 2000);
}

function stopStatusPolling() {
  if (statusInterval) {
    clearInterval(statusInterval);
    statusInterval = null;
  }
}

// Safety: stop robot when leaving page
window.addEventListener("beforeunload", () => {
  if (detectionActive) {
    stopDetection();
  }
});

// Quick backend health check on load
window.addEventListener("load", async () => {
  try {
    const res = await fetch(`${BASE_URL}/`);
    if (!res.ok) throw new Error("Not OK");
    console.log("Backend reachable.");
  } catch (err) {
    console.warn("Backend not running yet:", err);
    statusText.textContent = "Backend not running. Start the robot server first.";
    statusText.style.color = "#ef4444";
    robotStatus.textContent = "Robot server is not running yet.";
    helperMessage.textContent =
      "Please start the robot server, then press Start Sorting.";
  }
});
