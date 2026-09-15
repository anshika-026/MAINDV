import { useEffect, useRef, useState } from "react";
import { Camera, Upload, X } from "lucide-react";

// Two independent ways to collect face samples — a live webcam capture and a
// multi-file upload — feeding the same `photos` array. Used by both the
// add-person wizard and the edit-person modal so the interaction (and the
// camera plumbing) only needs to exist once.
export default function FaceEnrollment({ photos, onAddPhoto, onRemovePhoto }) {
  const videoRef = useRef(null);
  const streamRef = useRef(null);
  const fileInputRef = useRef(null);
  const [cameraOn, setCameraOn] = useState(false);
  const [cameraError, setCameraError] = useState("");

  useEffect(() => () => stopCamera(), []);

  async function startCamera() {
    setCameraError("");
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      streamRef.current = stream;
      if (videoRef.current) videoRef.current.srcObject = stream;
      setCameraOn(true);
    } catch {
      setCameraError("Camera access denied or unavailable — try uploading instead.");
    }
  }

  function stopCamera() {
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    setCameraOn(false);
  }

  function capture() {
    const video = videoRef.current;
    if (!video || !video.videoWidth) return;
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d").drawImage(video, 0, 0);
    canvas.toBlob((blob) => {
      if (blob) onAddPhoto(URL.createObjectURL(blob));
    }, "image/jpeg", 0.9);
  }

  function handleFiles(fileList) {
    Array.from(fileList || []).forEach((file) => onAddPhoto(URL.createObjectURL(file)));
  }

  return (
    <div className="space-y-4">
      <div className="w-full h-48 rounded-2xl border-2 border-dashed border-border-300 bg-[#f8f9fc] flex flex-col items-center justify-center gap-2 overflow-hidden">
        {cameraOn ? (
          <video ref={videoRef} autoPlay playsInline muted className="w-full h-full object-cover" />
        ) : (
          <div
            onDragOver={(e) => e.preventDefault()}
            onDrop={(e) => {
              e.preventDefault();
              handleFiles(e.dataTransfer.files);
            }}
            onClick={() => fileInputRef.current?.click()}
            className="w-full h-full flex flex-col items-center justify-center gap-2 cursor-pointer"
          >
            <Camera size={28} className="text-slate-400" />
            <p className="text-sm font-medium text-ink-900">Drag & drop photos here</p>
            <p className="text-xs text-slate-400 px-6 text-center">or capture from your camera / upload below</p>
          </div>
        )}
      </div>

      {cameraError && <p className="text-xs text-danger-600">{cameraError}</p>}

      <div className="flex flex-wrap items-center gap-3">
        {cameraOn ? (
          <>
            <button type="button" onClick={capture} className="btn-primary flex-1 flex items-center justify-center gap-2">
              <Camera size={15} /> Capture photo
            </button>
            <button type="button" onClick={stopCamera} className="btn-secondary flex-1">
              Stop camera
            </button>
          </>
        ) : (
          <button
            type="button"
            onClick={startCamera}
            className="btn-secondary flex-1 flex items-center justify-center gap-2"
          >
            <Camera size={15} /> Capture from camera
          </button>
        )}
        <button
          type="button"
          onClick={() => fileInputRef.current?.click()}
          className="btn-secondary flex-1 flex items-center justify-center gap-2"
        >
          <Upload size={15} /> Upload photo(s)
        </button>
      </div>
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        multiple
        hidden
        onChange={(e) => handleFiles(e.target.files)}
      />

      {photos.length > 0 && (
        <div>
          <p className="text-xs font-medium text-slate-500 mb-2">
            {photos.length} sample{photos.length === 1 ? "" : "s"}
          </p>
          <div className="grid grid-cols-4 gap-2">
            {photos.map((p) => (
              <div
                key={p.id}
                className="relative aspect-square rounded-lg overflow-hidden border border-border-200 bg-[#f8f9fc] flex items-center justify-center"
              >
                {p.url ? (
                  <img src={p.url} alt="Sample" className="w-full h-full object-cover" />
                ) : (
                  <span className="text-[10px] text-slate-400 text-center px-1">No preview</span>
                )}
                <button
                  type="button"
                  onClick={() => onRemovePhoto(p.id)}
                  className="absolute top-1 right-1 w-5 h-5 rounded-full bg-black/60 text-white flex items-center justify-center hover:bg-danger-500"
                  title="Remove photo"
                >
                  <X size={12} />
                </button>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
