const healthBadge = document.getElementById("healthBadge");
const metaBadge = document.getElementById("metaBadge");
const chatForm = document.getElementById("chatForm");
const questionInput = document.getElementById("question");
const askBtn = document.getElementById("askBtn");
const answerBox = document.getElementById("answer");
const citationsBox = document.getElementById("citations");
const suggestions = document.getElementById("suggestions");
const uploadForm = document.getElementById("uploadForm");
const pdfFileInput = document.getElementById("pdfFile");
const uploadBtn = document.getElementById("uploadBtn");
const uploadStatus = document.getElementById("uploadStatus");
const documentScope = document.getElementById("documentScope");

let jobPollTimer = null;

async function checkHealth() {
  try {
    const [healthRes, metaRes] = await Promise.all([
      fetch("/api/health"),
      fetch("/api/meta"),
    ]);

    if (!healthRes.ok) {
      throw new Error("Health endpoint unavailable");
    }

    const health = await healthRes.json();
    if (health.vectorstore_ready && health.llm_ready) {
      healthBadge.textContent = "API Ready";
      healthBadge.className = "badge ok";
    } else if (!health.vectorstore_ready) {
      healthBadge.textContent = "Vectorstore Missing";
      healthBadge.className = "badge warn";
    } else if (!health.llm_ready) {
      healthBadge.textContent = "LLM Offline";
      healthBadge.className = "badge warn";
    } else {
      healthBadge.textContent = "API Degraded";
      healthBadge.className = "badge warn";
    }

    if (metaRes.ok) {
      const meta = await metaRes.json();
      metaBadge.textContent = `Model: ${meta.model} | Top-K: ${meta.retrieval_top_k}`;
    }
  } catch (error) {
    healthBadge.textContent = "API Offline";
    healthBadge.className = "badge warn";
    metaBadge.textContent = "Check backend service status.";
  }
}

function renderCitations(citations) {
  if (!Array.isArray(citations) || citations.length === 0) {
    citationsBox.className = "citations empty";
    citationsBox.textContent = "No citations returned.";
    return;
  }

  citationsBox.className = "citations";
  citationsBox.innerHTML = citations
    .map((item) => {
      const pageText = item.page ? `p.${item.page}` : "page n/a";
      const scoreText = item.score !== null && item.score !== undefined ? `score ${item.score}` : "";
      return `
        <article class="citation">
          <div class="head">${item.source} - ${pageText} ${scoreText}</div>
          <div class="snippet">${item.snippet || ""}</div>
        </article>
      `;
    })
    .join("");
}

async function loadDocuments() {
  if (!documentScope) return;

  const selectedValue = documentScope.value;
  try {
    const response = await fetch("/api/documents");
    if (!response.ok) {
      throw new Error("Documents endpoint unavailable");
    }

    const documents = await response.json();
    const options = [
      '<option value="">All indexed documents</option>',
      ...documents.map((document) => {
        const pageText = document.page_count ? ` (${document.page_count} pages)` : "";
        return `<option value="${document.id}">${document.filename}${pageText}</option>`;
      }),
    ];
    documentScope.innerHTML = options.join("");

    if ([...documentScope.options].some((option) => option.value === selectedValue)) {
      documentScope.value = selectedValue;
    }
  } catch (error) {
    documentScope.innerHTML = '<option value="">All indexed documents</option>';
  }
}

function setUploadStatus(message, state = "empty") {
  uploadStatus.className = `job-status ${state}`;
  uploadStatus.textContent = message;
}

function stopJobPolling() {
  if (jobPollTimer) {
    window.clearInterval(jobPollTimer);
    jobPollTimer = null;
  }
}

async function fetchJobStatus(jobId) {
  const response = await fetch(`/api/ingestion/jobs/${jobId}`);
  if (!response.ok) {
    throw new Error("Could not refresh ingestion status");
  }
  return response.json();
}

async function pollJobStatus(jobId) {
  stopJobPolling();

  const refresh = async () => {
    try {
      const job = await fetchJobStatus(jobId);
      const chunks = job.chunks_created !== null && job.chunks_created !== undefined
        ? ` | ${job.chunks_created} chunks`
        : "";
      setUploadStatus(`Job ${job.id}: ${job.status}${chunks}`, job.status);

      if (job.status === "completed" || job.status === "failed") {
        stopJobPolling();
        uploadBtn.disabled = false;
        uploadBtn.textContent = "Upload PDF";
        if (job.status === "completed") {
          checkHealth();
          loadDocuments();
        }
      }
    } catch (error) {
      stopJobPolling();
      uploadBtn.disabled = false;
      uploadBtn.textContent = "Upload PDF";
      setUploadStatus(`Error: ${error.message}`, "failed");
    }
  };

  await refresh();
  jobPollTimer = window.setInterval(refresh, 3000);
}

async function uploadPdf(file) {
  uploadBtn.disabled = true;
  uploadBtn.textContent = "Uploading...";
  setUploadStatus("Uploading PDF...", "running");

  try {
    const formData = new FormData();
    formData.append("file", file);

    const response = await fetch("/api/documents/upload", {
      method: "POST",
      body: formData,
    });

    const body = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(body.detail || "Upload failed");
    }

    setUploadStatus(`Uploaded ${body.filename}. Job ${body.job_id}: ${body.job_status}`, "pending");
    await pollJobStatus(body.job_id);
  } catch (error) {
    uploadBtn.disabled = false;
    uploadBtn.textContent = "Upload PDF";
    setUploadStatus(`Error: ${error.message}`, "failed");
  }
}

async function askQuestion(question) {
  askBtn.disabled = true;
  askBtn.textContent = "Thinking...";
  answerBox.className = "answer";
  answerBox.textContent = "";
  citationsBox.className = "citations empty";
  citationsBox.textContent = "Collecting citations...";

  const selectedDocumentId = documentScope && documentScope.value
    ? Number(documentScope.value)
    : null;

  try {
    const response = await fetch("/api/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        document_id: selectedDocumentId,
      }),
    });

    if (!response.ok) {
      const body = await response.json().catch(() => ({}));
      throw new Error(body.detail || "Chat request failed");
    }

    if (!response.body) {
      throw new Error("Streaming is not supported by this browser.");
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let hasAnswer = false;

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() || "";

      for (const line of lines) {
        if (!line.trim()) continue;
        const event = JSON.parse(line);

        if (event.type === "token") {
          answerBox.textContent += event.content || "";
          hasAnswer = true;
        } else if (event.type === "final") {
          answerBox.textContent = event.answer || answerBox.textContent;
          hasAnswer = true;
        } else if (event.type === "citations") {
          renderCitations(event.citations || []);
        } else if (event.type === "error") {
          throw new Error(event.detail || "Streaming request failed");
        }
      }
    }

    if (buffer.trim()) {
      const event = JSON.parse(buffer);
      if (event.type === "token") {
        answerBox.textContent += event.content || "";
        hasAnswer = true;
      } else if (event.type === "final") {
        answerBox.textContent = event.answer || answerBox.textContent;
        hasAnswer = true;
      } else if (event.type === "citations") {
        renderCitations(event.citations || []);
      } else if (event.type === "error") {
        throw new Error(event.detail || "Streaming request failed");
      }
    }

    if (!hasAnswer) {
      answerBox.textContent = "No answer text returned.";
    }
  } catch (error) {
    answerBox.textContent = `Error: ${error.message}`;
    citationsBox.className = "citations empty";
    citationsBox.textContent = "No citations due to request failure.";
  } finally {
    askBtn.disabled = false;
    askBtn.textContent = "Generate Answer";
  }
}

chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = questionInput.value.trim();
  if (!question) return;
  await askQuestion(question);
});

suggestions.addEventListener("click", (event) => {
  const button = event.target.closest("button[data-q]");
  if (!button) return;
  questionInput.value = button.getAttribute("data-q");
  questionInput.focus();
});

uploadForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const file = pdfFileInput.files && pdfFileInput.files[0];
  if (!file) {
    setUploadStatus("Choose a PDF first.", "failed");
    return;
  }
  if (!file.name.toLowerCase().endsWith(".pdf")) {
    setUploadStatus("Only PDF files are supported.", "failed");
    return;
  }
  await uploadPdf(file);
});

checkHealth();
loadDocuments();
