import { useEffect, useMemo, useRef, useState } from "react";
import "./App.css";

const _envUrl = import.meta.env.VITE_PIPELINE_WS_URL;
const _proto =
  typeof window !== "undefined" && window.location?.protocol === "https:"
    ? "wss"
    : "ws";
const _host =
  typeof window !== "undefined" ? window.location.hostname : "localhost";
const PIPELINE_WS_URL = _envUrl ?? `${_proto}://${_host}:8000/scan`;

const PIPELINE_STEPS = [
  {
    id: "region_classification",
    title: "Region classification",
    description: "Identifies the anatomical region in the X-ray.",
  },
  {
    id: "fracture_detection",
    title: "Fracture detection",
    description: "Checks whether a fracture is present.",
  },
  {
    id: "bounding_box",
    title: "Bounding box extraction",
    description: "Marks the localized fracture area when needed.",
  },
  {
    id: "llm",
    title: "Clinical summary",
    description: "Produces the final explanation for the scan.",
  },
];

const STEP_SET = new Set(PIPELINE_STEPS.map((step) => step.id));

const createInitialSteps = () =>
  Object.fromEntries(
    PIPELINE_STEPS.map((step) => [
      step.id,
      { status: "waiting", result: null },
    ]),
  );

const statusLabels = {
  idle: "Ready",
  connecting: "Connecting",
  running: "Running scan",
  complete: "Scan complete",
  error: "Connection error",
};

function formatStepResult(stepId, result) {
  if (result == null) {
    return "No result returned.";
  }

  if (stepId === "llm") {
    if (typeof result === "object" && result.text) {
      return result.text;
    }

    return typeof result === "string"
      ? result
      : JSON.stringify(result, null, 2);
  }

  if (typeof result === "boolean") {
    return result ? "Positive" : "Negative";
  }

  if (typeof result === "object") {
    return JSON.stringify(result, null, 2);
  }

  return String(result);
}

function getNextStep(currentStep, result) {
  if (currentStep === "region_classification") {
    return "fracture_detection";
  }

  if (currentStep === "fracture_detection") {
    return result ? "bounding_box" : "llm";
  }

  if (currentStep === "bounding_box") {
    return "llm";
  }

  return null;
}

function StatusChip({ status }) {
  return (
    <span className={`status-chip status-chip--${status}`}>
      {statusLabels[status] ?? status}
    </span>
  );
}

function PipelineStepCard({ step, data, active }) {
  return (
    <article
      className={`step-card step-card--${data.status}${active ? " step-card--active" : ""}`}
    >
      <div className="step-card__header">
        <div>
          <h3>{step.title}</h3>
          <p>{step.description}</p>
        </div>
        <span className={`step-pill step-pill--${data.status}`}>
          {data.status}
        </span>
      </div>

      <pre className="step-card__result">
        {formatStepResult(step.id, data.result)}
      </pre>
    </article>
  );
}

function ScanEvent({ event }) {
  return (
    <li className="event-item">
      <div className="event-item__meta">
        <span className="event-item__step">{event.label}</span>
        {event.time ? (
          <span className="event-item__time">{event.time}</span>
        ) : null}
      </div>
      <div className="event-item__body">{event.body}</div>
    </li>
  );
}

export default function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [connectionStatus, setConnectionStatus] = useState("idle");
  const [activeStep, setActiveStep] = useState(null);
  const [steps, setSteps] = useState(createInitialSteps);
  const [events, setEvents] = useState([]);
  const [errorMessage, setErrorMessage] = useState("");

  const fileInputRef = useRef(null);
  const socketRef = useRef(null);
  const previewRef = useRef("");

  useEffect(() => {
    return () => {
      if (socketRef.current) {
        socketRef.current.close();
      }

      if (previewRef.current) {
        URL.revokeObjectURL(previewRef.current);
      }
    };
  }, []);

  useEffect(() => {
    if (previewRef.current) {
      URL.revokeObjectURL(previewRef.current);
      previewRef.current = "";
    }

    if (selectedFile) {
      const nextPreview = URL.createObjectURL(selectedFile);
      previewRef.current = nextPreview;
      setPreviewUrl(nextPreview);
    } else {
      setPreviewUrl("");
    }
  }, [selectedFile]);

  const selectedFileInfo = useMemo(() => {
    if (!selectedFile) {
      return null;
    }

    const sizeInMb = (selectedFile.size / (1024 * 1024)).toFixed(2);
    return `${selectedFile.name} · ${sizeInMb} MB`;
  }, [selectedFile]);

  const appendEvent = (label, body) => {
    setEvents((current) => [
      ...current,
      {
        id: `${Date.now()}-${Math.random().toString(16).slice(2)}`,
        label,
        body,
        time: new Date().toLocaleTimeString([], {
          hour: "2-digit",
          minute: "2-digit",
        }),
      },
    ]);
  };

  const resetScanState = () => {
    if (socketRef.current) {
      socketRef.current.close();
      socketRef.current = null;
    }

    setConnectionStatus("idle");
    setActiveStep(null);
    setSteps(createInitialSteps());
    setEvents([]);
    setErrorMessage("");
  };

  const handleFileSelection = (file) => {
    if (!file || !file.type.startsWith("image/")) {
      setErrorMessage("Please choose a PNG, JPEG, or other image file.");
      return;
    }

    setSelectedFile(file);
    setErrorMessage("");
    setEvents([]);
    setSteps(createInitialSteps());
    setActiveStep(null);
    setConnectionStatus("idle");
  };

  const handleFileChange = (event) => {
    const file = event.target.files?.[0];
    handleFileSelection(file);
    event.target.value = "";
  };

  const handleDrop = (event) => {
    event.preventDefault();
    const file = event.dataTransfer.files?.[0];
    handleFileSelection(file);
  };

  const openFilePicker = () => fileInputRef.current?.click();

  const runScan = async () => {
    if (!selectedFile) {
      setErrorMessage("Select an X-ray image before starting the pipeline.");
      return;
    }

    if (socketRef.current) {
      socketRef.current.close();
      socketRef.current = null;
    }

    setErrorMessage("");
    setConnectionStatus("connecting");
    setSteps(createInitialSteps());
    setEvents([]);
    setActiveStep("region_classification");

    const socket = new WebSocket(PIPELINE_WS_URL);
    socketRef.current = socket;

    socket.onopen = async () => {
      try {
        setConnectionStatus("running");
        appendEvent("connection", `Connected to ${PIPELINE_WS_URL}`);

        const imageBuffer = await selectedFile.arrayBuffer();
        socket.send(imageBuffer);
        appendEvent("upload", `Uploaded ${selectedFile.name} to the pipeline.`);
      } catch (uploadError) {
        setConnectionStatus("error");
        setErrorMessage(
          uploadError.message || "Failed to send the image to the pipeline.",
        );
      }
    };

    socket.onmessage = (event) => {
      let payload;

      try {
        payload = JSON.parse(event.data);
      } catch {
        appendEvent("message", event.data);
        return;
      }

      if (payload.step === "error") {
        setConnectionStatus("error");
        setErrorMessage(payload.detail || "The backend returned an error.");
        appendEvent("error", payload.detail || "Unknown backend error");
        socket.close();
        return;
      }

      if (payload.step === "pipeline") {
        setConnectionStatus(
          payload.status === "complete" ? "complete" : "error",
        );
        setActiveStep(null);
        appendEvent("pipeline", `Pipeline ${payload.status}.`);
        socket.close();
        return;
      }

      if (STEP_SET.has(payload.step)) {
        setSteps((current) => {
          const next = { ...current };
          next[payload.step] = {
            status: "done",
            result: payload.result,
          };

          if (payload.step === "fracture_detection" && !payload.result) {
            next.bounding_box = {
              status: "skipped",
              result: "Skipped because fracture detection was negative.",
            };
          }

          return next;
        });

        appendEvent(
          payload.step,
          formatStepResult(payload.step, payload.result),
        );
        setActiveStep(getNextStep(payload.step, payload.result));
      }
    };

    socket.onerror = () => {
      setConnectionStatus("error");
      setErrorMessage(
        "Could not reach the pipeline websocket. Make sure FastAPI is running on port 8000.",
      );
      appendEvent("error", "WebSocket connection failed.");
    };

    socket.onclose = () => {
      socketRef.current = null;
    };
  };

  const onDropZoneKeyDown = (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      openFilePicker();
    }
  };

  return (
    <main className="app-shell">
      <div className="ambient ambient--one" />
      <div className="ambient ambient--two" />

      <header className="topbar">
        <div>
          <p className="eyebrow">X-CV pipeline bridge</p>
          <h1>Connect the X-ray pipeline to the React UI.</h1>
          <p className="lede">
            Upload an X-ray, stream it to the FastAPI websocket, and watch the
            pipeline return region, fracture, and summary results in real time.
          </p>
        </div>

        <div className="topbar__status">
          <StatusChip status={connectionStatus} />
          <p>{PIPELINE_WS_URL}</p>
        </div>
      </header>

      <section className="workspace">
        <aside className="panel panel--upload">
          <div
            className="dropzone"
            role="button"
            tabIndex={0}
            onClick={openFilePicker}
            onKeyDown={onDropZoneKeyDown}
            onDragOver={(event) => event.preventDefault()}
            onDrop={handleDrop}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              onChange={handleFileChange}
              hidden
            />
            <div className="dropzone__badge">Image input</div>
            <h2>Drop an X-ray here</h2>
            <p>
              PNG or JPEG images are sent as binary websocket payloads to
              <code>/scan</code>.
            </p>
            <button type="button" className="button button--ghost">
              Choose file
            </button>
          </div>

          <div className="action-row">
            <button
              type="button"
              className="button button--primary"
              onClick={runScan}
              disabled={!selectedFile}
            >
              Run pipeline
            </button>
            <button
              type="button"
              className="button button--secondary"
              onClick={resetScanState}
            >
              Reset
            </button>
          </div>

          <div className="meta-card">
            <span className="meta-card__label">Selected file</span>
            <strong>{selectedFileInfo ?? "No file selected yet"}</strong>
            <p>
              Point this UI at a local or deployed FastAPI backend by changing
              VITE_PIPELINE_WS_URL.
            </p>
          </div>

          {errorMessage ? (
            <div className="error-banner">{errorMessage}</div>
          ) : null}
        </aside>

        <section className="panel panel--output">
          <div className="output-grid">
            <div className="preview-card">
              <div className="preview-card__header">
                <h2>Preview</h2>
                <span className="preview-card__hint">binary upload</span>
              </div>

              {previewUrl ? (
                <img
                  className="preview-image"
                  src={previewUrl}
                  alt={selectedFile?.name ?? "X-ray preview"}
                />
              ) : (
                <div className="preview-empty">
                  <p>Choose an image to see the preview and start a scan.</p>
                </div>
              )}
            </div>

            <div className="summary-card">
              <div className="preview-card__header">
                <h2>Pipeline summary</h2>
                <span className="preview-card__hint">live status</span>
              </div>

              <div className="summary-grid">
                {PIPELINE_STEPS.map((step) => (
                  <PipelineStepCard
                    key={step.id}
                    step={step}
                    data={steps[step.id]}
                    active={activeStep === step.id}
                  />
                ))}
              </div>
            </div>
          </div>

          <div className="log-card">
            <div className="preview-card__header">
              <h2>Event log</h2>
              <span className="preview-card__hint">ordered stream</span>
            </div>

            {events.length ? (
              <ul className="event-list">
                {events.map((event) => (
                  <ScanEvent key={event.id} event={event} />
                ))}
              </ul>
            ) : (
              <div className="preview-empty preview-empty--compact">
                <p>
                  The websocket output will appear here after you start a scan.
                </p>
              </div>
            )}
          </div>
        </section>
      </section>
    </main>
  );
}
