import { useEffect, useRef, useState } from "react";
import { ChevronRight, Trash2, Upload } from "lucide-react";
import Modal from "../../components/Modal";
import Avatar from "../../components/Avatar";
import * as api from "../../api/client";
import { useAuth } from "../../context/AuthContext";

const COUNTRY_CODES = ["+91", "+1", "+44", "+971", "+65"];

function splitName(name) {
  const parts = String(name || "").trim().split(/\s+/);
  return { firstName: parts[0] || "", lastName: parts.slice(1).join(" ") || "" };
}

function splitMobile(mobile) {
  const match = String(mobile || "").match(/^(\+\d{1,3})\s*(.*)$/);
  return match ? { code: match[1], number: match[2] } : { code: "+91", number: mobile || "" };
}

function Field({ label, value, onEdit }) {
  return (
    <div className="py-4 first:pt-0 last:pb-0">
      <p className="text-sm font-medium text-ink-900">{label}</p>
      <p className="text-sm text-slate-500 mt-0.5">{value}</p>
      {onEdit && (
        <button onClick={onEdit} className="text-sm text-brand-600 font-medium mt-1.5 flex items-center gap-0.5">
          Edit <ChevronRight size={14} />
        </button>
      )}
    </div>
  );
}

export default function Profile() {
  const { updateUser } = useAuth();
  const [profile, setProfile] = useState(null);
  const [photoUrl, setPhotoUrl] = useState(null);
  const [editing, setEditing] = useState(null); // 'picture' | 'email' | 'name' | 'mobile' | null
  const [draft, setDraft] = useState({});
  const [saving, setSaving] = useState(false);
  const [toast, setToast] = useState("");
  const fileInputRef = useRef(null);

  useEffect(() => {
    api.getProfile().then(setProfile);
  }, []);

  if (!profile) return null;

  function showToast(msg) {
    setToast(msg);
    setTimeout(() => setToast(""), 2000);
  }

  function openEdit(field) {
    if (field === "name") setDraft(splitName(profile.name));
    else if (field === "mobile") setDraft(splitMobile(profile.mobile));
    else if (field === "email") setDraft({ email: profile.email });
    setEditing(field);
  }

  function closeEdit() {
    setEditing(null);
    setDraft({});
  }

  async function saveField(updated) {
    setSaving(true);
    const next = { ...profile, ...updated };
    await api.updateProfile(next);
    setProfile(next);
    updateUser(updated);
    setSaving(false);
    closeEdit();
    showToast("Profile updated");
  }

  function handleEmailSubmit(e) {
    e.preventDefault();
    saveField({ email: draft.email });
  }

  function handleNameSubmit(e) {
    e.preventDefault();
    saveField({ name: `${draft.firstName} ${draft.lastName}`.trim() });
  }

  function handleMobileSubmit(e) {
    e.preventDefault();
    saveField({ mobile: `${draft.code} ${draft.number}`.trim() });
  }

  function handlePictureFiles(fileList) {
    const file = fileList?.[0];
    if (!file) return;
    setPhotoUrl(URL.createObjectURL(file));
  }

  return (
    <div className="card p-6 max-w-xl">
      <h2 className="text-base font-semibold text-ink-900 mb-5">Profile</h2>

      {toast && (
        <div className="text-sm text-success-600 bg-success-50 border border-success-500/20 rounded-lg px-3.5 py-2 mb-4">
          {toast}
        </div>
      )}

      <div className="divide-y divide-[#f1f2f7]">
        <div className="py-4 first:pt-0">
          <p className="text-sm font-medium text-ink-900 mb-2">Profile picture</p>
          {photoUrl ? (
            <img src={photoUrl} alt={profile.name} className="w-12 h-12 rounded-full object-cover" />
          ) : (
            <Avatar name={profile.name} size={48} />
          )}
          <button
            onClick={() => openEdit("picture")}
            className="text-sm text-brand-600 font-medium mt-1.5 flex items-center gap-0.5"
          >
            Edit <ChevronRight size={14} />
          </button>
        </div>

        <Field label="Email address" value={profile.email} onEdit={() => openEdit("email")} />
        <Field label="Name" value={profile.name} onEdit={() => openEdit("name")} />
        <Field label="Mobile" value={profile.mobile} onEdit={() => openEdit("mobile")} />
        <Field label="Role" value={profile.role} />
      </div>

      {/* --- Edit profile picture --- */}
      <Modal open={editing === "picture"} onClose={closeEdit} title="Edit profile picture">
        <div className="space-y-4">
          <div className="flex items-center gap-3">
            {photoUrl ? (
              <img src={photoUrl} alt={profile.name} className="w-12 h-12 rounded-full object-cover" />
            ) : (
              <Avatar name={profile.name} size={48} />
            )}
            <span className="text-sm font-medium text-ink-900">{profile.name}</span>
            {photoUrl && (
              <button
                type="button"
                onClick={() => setPhotoUrl(null)}
                title="Remove picture"
                className="text-slate-400 hover:text-danger-500 ml-auto"
              >
                <Trash2 size={16} />
              </button>
            )}
          </div>

          <div
            onDragOver={(e) => e.preventDefault()}
            onDrop={(e) => {
              e.preventDefault();
              handlePictureFiles(e.dataTransfer.files);
            }}
            onClick={() => fileInputRef.current?.click()}
            className="rounded-xl border-2 border-dashed border-border-300 bg-[#f8f9fc] py-8 text-center cursor-pointer"
          >
            <Upload size={20} className="mx-auto text-slate-400 mb-2" />
            <p className="text-sm font-medium text-ink-900">Drag and drop here or click to upload</p>
            <p className="text-xs text-slate-400 mt-1">You may upload one JPEG or PNG</p>
          </div>
          <input
            ref={fileInputRef}
            type="file"
            accept="image/jpeg,image/png"
            hidden
            onChange={(e) => handlePictureFiles(e.target.files)}
          />

          <div className="flex items-center gap-3 pt-1">
            <button type="button" onClick={closeEdit} className="btn-secondary flex-1">
              Cancel
            </button>
            <button
              type="button"
              onClick={() => {
                showToast("Profile picture updated");
                closeEdit();
              }}
              className="btn-primary flex-1"
            >
              Update
            </button>
          </div>
        </div>
      </Modal>

      {/* --- Edit email address --- */}
      <Modal open={editing === "email"} onClose={closeEdit} title="Edit email address">
        <form onSubmit={handleEmailSubmit} className="space-y-4">
          <div>
            <label className="text-sm font-medium text-ink-900 block mb-1.5">Email address</label>
            <input
              type="email"
              required
              value={draft.email || ""}
              onChange={(e) => setDraft((d) => ({ ...d, email: e.target.value }))}
              className="input-field"
            />
          </div>
          <div className="flex items-center gap-3 pt-1">
            <button type="button" onClick={closeEdit} className="btn-secondary flex-1">
              Cancel
            </button>
            <button type="submit" disabled={saving} className="btn-primary flex-1">
              {saving ? "Updating..." : "Update"}
            </button>
          </div>
        </form>
      </Modal>

      {/* --- Edit name --- */}
      <Modal open={editing === "name"} onClose={closeEdit} title="Edit name">
        <form onSubmit={handleNameSubmit} className="space-y-4">
          <div>
            <label className="text-sm font-medium text-ink-900 block mb-1.5">First name</label>
            <input
              required
              value={draft.firstName || ""}
              onChange={(e) => setDraft((d) => ({ ...d, firstName: e.target.value }))}
              className="input-field"
            />
          </div>
          <div>
            <label className="text-sm font-medium text-ink-900 block mb-1.5">Last name</label>
            <input
              value={draft.lastName || ""}
              onChange={(e) => setDraft((d) => ({ ...d, lastName: e.target.value }))}
              className="input-field"
            />
          </div>
          <div className="flex items-center gap-3 pt-1">
            <button type="button" onClick={closeEdit} className="btn-secondary flex-1">
              Cancel
            </button>
            <button type="submit" disabled={saving} className="btn-primary flex-1">
              {saving ? "Updating..." : "Update"}
            </button>
          </div>
        </form>
      </Modal>

      {/* --- Edit mobile number --- */}
      <Modal open={editing === "mobile"} onClose={closeEdit} title="Edit mobile number">
        <form onSubmit={handleMobileSubmit} className="space-y-4">
          <div>
            <label className="text-sm font-medium text-ink-900 block mb-1.5">Mobile number</label>
            <div className="flex gap-2">
              <select
                value={draft.code || "+91"}
                onChange={(e) => setDraft((d) => ({ ...d, code: e.target.value }))}
                className="input-field w-24"
              >
                {COUNTRY_CODES.map((c) => (
                  <option key={c}>{c}</option>
                ))}
              </select>
              <input
                required
                value={draft.number || ""}
                onChange={(e) => setDraft((d) => ({ ...d, number: e.target.value }))}
                className="input-field flex-1"
              />
            </div>
          </div>
          <div className="flex items-center gap-3 pt-1">
            <button type="button" onClick={closeEdit} className="btn-secondary flex-1">
              Cancel
            </button>
            <button type="submit" disabled={saving} className="btn-primary flex-1">
              {saving ? "Updating..." : "Update"}
            </button>
          </div>
        </form>
      </Modal>
    </div>
  );
}
