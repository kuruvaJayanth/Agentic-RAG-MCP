const API_BASE_URL = "http://127.0.0.1:8000";


// ============================================================
// DOM ELEMENTS
// ============================================================

const documentSelect = document.getElementById("document");
const questionInput = document.getElementById("question");
const askButton = document.getElementById("askButton");

const answerContent = document.getElementById("answerContent");
const answerStatus = document.getElementById("answerStatus");

const sourcesContent = document.getElementById("sourcesContent");
const sourceCount = document.getElementById("sourceCount");

const characterCount = document.querySelector(".character-count");


// Upload elements
const uploadArea = document.getElementById("uploadArea");
const pdfInput = document.getElementById("pdfInput");
const chooseFileButton = document.getElementById("chooseFileButton");

const selectedFile = document.getElementById("selectedFile");
const selectedFileName = document.getElementById("selectedFileName");
const selectedFileSize = document.getElementById("selectedFileSize");

const uploadButton = document.getElementById("uploadButton");
const uploadStatus = document.getElementById("uploadStatus");


// ============================================================
// LOAD DOCUMENTS
// ============================================================

async function loadDocuments(selectDocument = "") {

    try {

        const response = await fetch(
            `${API_BASE_URL}/api/documents`
        );

        if (!response.ok) {
            throw new Error("Failed to load documents.");
        }

        const data = await response.json();

        documentSelect.innerHTML = `
            <option value="">
                Select a research document
            </option>
        `;

        data.documents.forEach((documentName) => {

            const option = document.createElement("option");

            option.value = documentName;
            option.textContent = documentName;

            documentSelect.appendChild(option);

        });

        if (selectDocument) {

            documentSelect.value = selectDocument;

        }

    } catch (error) {

        console.error("Document loading error:", error);

    }

}


// ============================================================
// FILE SELECTION
// ============================================================

chooseFileButton.addEventListener(
    "click",
    () => {
        pdfInput.click();
    }
);


uploadArea.addEventListener(
    "click",
    (event) => {

        if (
            event.target !== chooseFileButton
        ) {

            pdfInput.click();

        }

    }
);


// ============================================================
// DRAG & DROP
// ============================================================

uploadArea.addEventListener(
    "dragover",
    (event) => {

        event.preventDefault();

        uploadArea.classList.add("dragging");

    }
);


uploadArea.addEventListener(
    "dragleave",
    () => {

        uploadArea.classList.remove("dragging");

    }
);


uploadArea.addEventListener(
    "drop",
    (event) => {

        event.preventDefault();

        uploadArea.classList.remove("dragging");

        const files = event.dataTransfer.files;

        if (files.length > 0) {

            handleSelectedFile(files[0]);

        }

    }
);


// ============================================================
// FILE INPUT
// ============================================================

pdfInput.addEventListener(
    "change",
    () => {

        if (pdfInput.files.length > 0) {

            handleSelectedFile(
                pdfInput.files[0]
            );

        }

    }
);


// ============================================================
// HANDLE SELECTED FILE
// ============================================================

function handleSelectedFile(file) {

    if (
        file.type !== "application/pdf" &&
        !file.name.toLowerCase().endsWith(".pdf")
    ) {

        showUploadStatus(
            "Please select a PDF file.",
            "error"
        );

        return;

    }


    selectedFile.classList.remove("hidden");

    selectedFileName.textContent =
        file.name;

    selectedFileSize.textContent =
        formatFileSize(file.size);

    uploadStatus.classList.add("hidden");

}


// ============================================================
// FORMAT FILE SIZE
// ============================================================

function formatFileSize(bytes) {

    if (bytes < 1024) {

        return `${bytes} B`;

    }

    if (bytes < 1024 * 1024) {

        return `${(bytes / 1024).toFixed(1)} KB`;

    }

    return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;

}


// ============================================================
// UPLOAD PDF
// ============================================================

uploadButton.addEventListener(
    "click",
    uploadPDF
);


async function uploadPDF() {

    if (
        !pdfInput.files ||
        pdfInput.files.length === 0
    ) {

        showUploadStatus(
            "Please select a PDF first.",
            "error"
        );

        return;

    }


    const file = pdfInput.files[0];


    const formData = new FormData();

    formData.append(
        "file",
        file
    );


    uploadButton.disabled = true;

    uploadButton.textContent =
        "Processing...";


    showUploadStatus(
        "Uploading and indexing your PDF...",
        "loading"
    );


    try {

        const response = await fetch(
            `${API_BASE_URL}/api/upload`,
            {
                method: "POST",
                body: formData
            }
        );


        const data = await response.json();


        if (!response.ok) {

            throw new Error(
                data.detail ||
                "Upload failed."
            );

        }


        showUploadStatus(
            `✓ ${data.message} ${data.chunks} chunks indexed.`,
            "success"
        );


        // Refresh document dropdown
        await loadDocuments(
            data.document
        );


        // Clear file selection
        pdfInput.value = "";


    } catch (error) {

        console.error(
            "Upload error:",
            error
        );

        showUploadStatus(
            error.message ||
            "Failed to upload PDF.",
            "error"
        );

    } finally {

        uploadButton.disabled = false;

        uploadButton.textContent =
            "Upload & Index";

    }

}


// ============================================================
// UPLOAD STATUS
// ============================================================

function showUploadStatus(
    message,
    type
) {

    uploadStatus.textContent =
        message;

    uploadStatus.className =
        `upload-status ${type}`;

    uploadStatus.classList.remove(
        "hidden"
    );

}


// ============================================================
// CHARACTER COUNTER
// ============================================================

questionInput.addEventListener(
    "input",
    () => {

        const length =
            questionInput.value.length;

        characterCount.textContent =
            `${length} / 500`;

    }
);


// ============================================================
// ASK AI
// ============================================================

askButton.addEventListener(
    "click",
    askQuestion
);


async function askQuestion() {

    const question =
        questionInput.value.trim();

    const documentName =
        documentSelect.value;


    if (!documentName) {

        showAnswerError(
            "Please select a document first."
        );

        return;

    }


    if (!question) {

        showAnswerError(
            "Please enter a question."
        );

        return;

    }


    // Loading state
    askButton.disabled = true;

    askButton.innerHTML = `
        <span class="ask-icon">✦</span>
        <span>Researching...</span>
        <span class="ask-arrow">...</span>
    `;


    answerStatus.textContent =
        "Thinking";


    answerContent.className =
        "answer-content";


    answerContent.innerHTML = `
        <div class="loading-state">
            <div class="loading-spinner"></div>
            <p>Searching your document and generating a grounded answer...</p>
        </div>
    `;


    sourceCount.textContent =
        "0 sources";


    sourcesContent.innerHTML = `
        <div class="empty-state small">
            <div class="empty-icon">◈</div>
            <p>Retrieving document sources...</p>
        </div>
    `;


    try {

        const response = await fetch(
            `${API_BASE_URL}/api/ask`,
            {
                method: "POST",

                headers: {
                    "Content-Type":
                        "application/json"
                },

                body: JSON.stringify({

                    question: question,

                    document_name:
                        documentName

                })

            }
        );


        const data =
            await response.json();


        if (!response.ok) {

            throw new Error(
                data.detail ||
                "Request failed."
            );

        }


        displayAnswer(
            data.answer
        );


        displaySources(
            data.sources || []
        );


        answerStatus.textContent =
            "Complete";


    } catch (error) {

        console.error(
            "Ask error:",
            error
        );

        showAnswerError(
            error.message ||
            "Something went wrong."
        );

    } finally {

        askButton.disabled = false;

        askButton.innerHTML = `
            <span class="ask-icon">✦</span>
            <span>Ask AI</span>
            <span class="ask-arrow">→</span>
        `;

    }

}


// ============================================================
// DISPLAY ANSWER
// ============================================================

function displayAnswer(answer) {

    answerContent.className =
        "answer-content";


    answerContent.innerHTML = `
        <div class="answer-text">
            ${escapeHTML(answer)}
        </div>
    `;

}


// ============================================================
// DISPLAY SOURCES
// ============================================================

function displaySources(sources) {

    sourceCount.textContent =
        `${sources.length} source${sources.length === 1 ? "" : "s"}`;


    if (!sources.length) {

        sourcesContent.innerHTML = `
            <div class="empty-state small">
                <div class="empty-icon">◈</div>
                <p>No sources were returned.</p>
            </div>
        `;

        return;

    }


    sourcesContent.innerHTML =
        sources.map(
            (source, index) => `
                <div class="source-item">

                    <div class="source-number">
                        ${String(index + 1).padStart(2, "0")}
                    </div>

                    <div class="source-details">

                        <strong>
                            ${escapeHTML(source.document)}
                        </strong>

                        <span>
                            Page ${source.page}
                        </span>

                    </div>

                </div>
            `
        ).join("");

}


// ============================================================
// ERROR
// ============================================================

function showAnswerError(message) {

    answerStatus.textContent =
        "Error";


    answerContent.className =
        "answer-content";


    answerContent.innerHTML = `
        <div class="error-state">
            <div class="empty-icon">!</div>
            <h4>Something went wrong</h4>
            <p>${escapeHTML(message)}</p>
        </div>
    `;


    sourceCount.textContent =
        "0 sources";


    sourcesContent.innerHTML = `
        <div class="empty-state small">
            <div class="empty-icon">◈</div>
            <p>No sources available.</p>
        </div>
    `;

}


// ============================================================
// HTML ESCAPE
// ============================================================

function escapeHTML(value) {

    return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#039;");

}


// ============================================================
// KEYBOARD SHORTCUT
// ============================================================

questionInput.addEventListener(
    "keydown",
    (event) => {

        if (
            (event.ctrlKey || event.metaKey) &&
            event.key === "Enter"
        ) {

            event.preventDefault();

            askQuestion();

        }

    }
);


// ============================================================
// INITIALIZE
// ============================================================

loadDocuments();