import { useEffect, useState } from "react";
import { IdCard, QrCode, Video, SlidersHorizontal, UserCog, Trash2, Pencil } from "lucide-react";
import PageHeader from "../components/PageHeader";
import StatCard from "../components/StatCard";
import DataTable from "../components/DataTable";
import Modal from "../components/Modal";
import PasswordField from "../components/PasswordField";
import * as api from "../api/client";

const CREATE_NEW = "__create_new__";

const STATUS_TONE = {
  active: "badge-success",
  inactive: "badge-neutral",
  suspended: "badge-warning",
  expired: "badge-danger",
};

function portalUrl(companySlug) {
  return companySlug ? `${window.location.origin}/client/${companySlug}/login` : null;
}

function formatExpiry(lic) {
  if (lic.non_expiring) return "No expiry";
  return new Date(lic.expires_at * 1000).toLocaleDateString();
}

function StatusPill({ status }) {
  return <span className={`badge ${STATUS_TONE[status] || "badge-neutral"}`}>{status}</span>;
}

export default function LicenseManagement() {
  const [companies, setCompanies] = useState([]);
  const [licenses, setLicenses] = useState([]);
  const [analytics, setAnalytics] = useState(null);
  const [catalog, setCatalog] = useState([]);
  const [allCameras, setAllCameras] = useState([]);
  const [toast, setToast] = useState("");

  const [addOpen, setAddOpen] = useState(false);
  const [form, setForm] = useState({
    companyId: CREATE_NEW, newCompanyName: "", maxCameras: 4, label: "", featureKeys: [],
    username: "", password: "", expiryDate: "",
  });

  const [editFor, setEditFor] = useState(null); // license row
  const [editForm, setEditForm] = useState({ label: "", maxCameras: 1, expiryDate: "" });

  const [assignFor, setAssignFor] = useState(null); // license row
  const [assignSelected, setAssignSelected] = useState([]);

  const [featuresFor, setFeaturesFor] = useState(null); // license row
  const [featuresSelected, setFeaturesSelected] = useState([]);

  const [qrFor, setQrFor] = useState(null); // license row

  const [credsFor, setCredsFor] = useState(null); // license row
  const [credsForm, setCredsForm] = useState({ username: "", password: "" });
  const [credsSaved, setCredsSaved] = useState(false); // true once Save succeeds — shows the reveal-once view

  // Passwords are stored as a one-way hash (see backend/app/license_db.py)
  // — nobody, including this app, can ever read a previously set password
  // back. The only honest way to "show the current password" is to show
  // it once, right when it's set, before it's hashed and the plaintext is
  // gone for good — that's what credsSaved/revealCreds below are for.
  const [revealCreds, setRevealCreds] = useState(null); // { username, password } | null

  useEffect(() => {
    refresh();
    api.getLicenseFeatureCatalog().then(setCatalog);
    api.getCameras().then(setAllCameras);
  }, []);

  function refresh() {
    api.getCompanies().then(setCompanies);
    api.getLicenses().then((d) => setLicenses(d.licenses));
    api.getLicenseAnalytics().then(setAnalytics);
  }

  function showToast(msg) {
    setToast(msg);
    setTimeout(() => setToast(""), 2500);
  }

  function companyName(companyId) {
    return companies.find((c) => c.id === companyId)?.name || "—";
  }

  function toggleFormFeature(key) {
    setForm((f) => ({
      ...f,
      featureKeys: f.featureKeys.includes(key) ? f.featureKeys.filter((k) => k !== key) : [...f.featureKeys, key],
    }));
  }

  async function handleGenerate(e) {
    e.preventDefault();
    let companyId = form.companyId;
    if (companyId === CREATE_NEW) {
      if (!form.newCompanyName.trim()) return;
      const company = await api.createCompany(form.newCompanyName.trim());
      companyId = company.id;
    }
    try {
      const username = form.username.trim();
      const password = form.password;
      const created = await api.createLicense({
        company_id: companyId,
        max_cameras: Number(form.maxCameras) || 1,
        label: form.label,
        feature_keys: form.featureKeys,
        username,
        password,
        expiry_date: form.expiryDate || null,
      });
      setAddOpen(false);
      setForm({
        companyId: CREATE_NEW, newCompanyName: "", maxCameras: 4, label: "", featureKeys: [],
        username: "", password: "", expiryDate: "",
      });
      refresh();
      // This is the only moment the plaintext password is known anywhere —
      // show it once so it can be copied/shared before it's gone for good.
      setRevealCreds({ username, password, portal: portalUrl(created.company_portal_slug) });
    } catch (err) {
      showToast(err.message);
    }
  }

  function openEdit(license) {
    setEditFor(license);
    setEditForm({
      label: license.label || "",
      maxCameras: license.max_cameras,
      expiryDate: license.non_expiring ? "" : new Date(license.expires_at * 1000).toISOString().slice(0, 10),
    });
  }

  async function saveEdit(e) {
    e.preventDefault();
    try {
      await api.updateLicense(editFor.id, {
        label: editForm.label,
        max_cameras: Number(editForm.maxCameras) || 1,
        // Leaving the date blank means "don't change the expiry" (same
        // convention as Edit Camera's blank-password-means-unchanged) —
        // omitting the key entirely (undefined, dropped by
        // JSON.stringify) rather than sending null, which the backend
        // would otherwise have no way to tell apart from "clear it".
        expiry_date: editForm.expiryDate || undefined,
      });
      setEditFor(null);
      showToast("Client license updated");
      refresh();
    } catch (err) {
      showToast(err.message);
    }
  }

  function openCredentials(license) {
    setCredsFor(license);
    setCredsForm({ username: license.username || "", password: "" });
    setCredsSaved(false);
  }

  function closeCredentials() {
    setCredsFor(null);
    setCredsSaved(false);
  }

  async function saveCredentials(e) {
    e.preventDefault();
    try {
      await api.setLicenseCredentials(credsFor.id, credsForm.username.trim(), credsForm.password);
      setCredsSaved(true); // reveal-once view — see credsSaved's declaration above
      showToast("Portal login updated");
      refresh();
    } catch (err) {
      showToast(err.message);
    }
  }

  async function copyToClipboard(text) {
    try {
      await navigator.clipboard.writeText(text);
      showToast("Copied to clipboard");
    } catch {
      showToast("Couldn't copy — select and copy it manually");
    }
  }

  async function handleStatusChange(license, status) {
    try {
      await api.setLicenseStatus(license.id, status);
      showToast(`Client license ${status}`);
      refresh();
    } catch (err) {
      showToast(err.message);
    }
  }

  async function handleDelete(license) {
    const name = license.username || license.label || companyName(license.company_id);
    if (!window.confirm(`Permanently delete the client license for "${name}"? This cannot be undone.`)) return;
    try {
      await api.deleteLicense(license.id);
      showToast(`Client license for "${name}" deleted`);
      refresh();
    } catch (err) {
      showToast(err.message);
    }
  }

  function openAssign(license) {
    setAssignFor(license);
    api.getLicenseCameras(license.id).then((cams) => setAssignSelected(cams.map((c) => c.id)));
  }

  function toggleAssignCamera(id) {
    setAssignSelected((prev) => (prev.includes(id) ? prev.filter((c) => c !== id) : [...prev, id]));
  }

  async function saveAssign() {
    try {
      const current = await api.getLicenseCameras(assignFor.id);
      const currentIds = current.map((c) => c.id);
      const toAdd = assignSelected.filter((id) => !currentIds.includes(id));
      const toRemove = currentIds.filter((id) => !assignSelected.includes(id));
      if (toAdd.length) await api.assignLicenseCameras(assignFor.id, toAdd);
      if (toRemove.length) await api.unassignLicenseCameras(assignFor.id, toRemove);
      setAssignFor(null);
      showToast("Camera assignment updated");
      refresh();
    } catch (err) {
      showToast(err.message);
    }
  }

  function openFeatures(license) {
    setFeaturesFor(license);
    setFeaturesSelected(license.features);
  }

  function toggleFeaturesSelected(key) {
    setFeaturesSelected((prev) => (prev.includes(key) ? prev.filter((k) => k !== key) : [...prev, key]));
  }

  async function saveFeatures() {
    try {
      await api.setLicenseFeatures(featuresFor.id, featuresSelected);
      setFeaturesFor(null);
      showToast("License features updated");
      refresh();
    } catch (err) {
      showToast(err.message);
    }
  }

  return (
    <div className="space-y-5">
      <PageHeader
        title="Client License Management"
        action={
          <button onClick={() => setAddOpen(true)} className="btn-primary text-sm flex items-center gap-2">
            <IdCard size={15} /> Generate client license
          </button>
        }
      />

      {toast && (
        <div className="text-sm text-success-600 bg-success-50 border border-success-500/20 rounded-lg px-3.5 py-2">
          {toast}
        </div>
      )}

      {analytics && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <StatCard label="Total Client Licenses" value={analytics.total_licenses} subTone="neutral" />
          <StatCard label="Active Client Licenses" value={analytics.active_licenses} />
          <StatCard label="Cameras Assigned" value={`${analytics.cameras_assigned} / ${analytics.total_cameras}`} subTone="neutral" />
          <StatCard label="License Usage" value={`${analytics.license_usage_percent}%`} subTone="neutral" />
        </div>
      )}

      <DataTable
        emptyLabel="No client licenses yet — generate one to get started."
        columns={[
          {
            key: "username",
            label: "Username",
            render: (r) =>
              r.username ? (
                <span className="font-mono text-xs">{r.username}</span>
              ) : (
                <span className="text-xs text-slate-400 italic">Not set</span>
              ),
          },
          { key: "company", label: "Company", render: (r) => companyName(r.company_id) },
          { key: "label", label: "Label", render: (r) => r.label || "—" },
          { key: "cameras", label: "Cameras", render: (r) => `${r.cameras_assigned} / ${r.max_cameras}` },
          {
            key: "features",
            label: "Features",
            render: (r) => (r.features.length ? `${r.features.length} enabled` : "None"),
          },
          { key: "expiry", label: "Expiry", render: (r) => <span className="text-xs text-slate-500">{formatExpiry(r)}</span> },
          {
            key: "status",
            label: "Status",
            // effective_status (computed server-side from status + expiry —
            // see license_db.effective_status), NOT the raw status column:
            // an "active" license whose expiry has passed must show as
            // Expired here even though nobody explicitly suspended it.
            render: (r) => <StatusPill status={r.effective_status} />,
          },
          {
            key: "action",
            label: "Actions",
            render: (r) => (
              <div className="flex items-center gap-3">
                <button onClick={() => openEdit(r)} title="Edit license" className="text-slate-400 hover:text-brand-600">
                  <Pencil size={15} />
                </button>
                <button onClick={() => openAssign(r)} title="Assign cameras" className="text-slate-400 hover:text-brand-600">
                  <Video size={15} />
                </button>
                <button onClick={() => openFeatures(r)} title="Manage features" className="text-slate-400 hover:text-brand-600">
                  <SlidersHorizontal size={15} />
                </button>
                <button
                  onClick={() => openCredentials(r)}
                  title={r.username ? "Reset portal login" : "Set portal login"}
                  className="text-slate-400 hover:text-brand-600"
                >
                  <UserCog size={15} />
                </button>
                <button onClick={() => setQrFor(r)} title="View QR" className="text-slate-400 hover:text-brand-600">
                  <QrCode size={15} />
                </button>
                <button onClick={() => handleDelete(r)} title="Delete license" className="text-slate-400 hover:text-danger-500">
                  <Trash2 size={15} />
                </button>
                <select
                  value={r.status}
                  onChange={(e) => handleStatusChange(r, e.target.value)}
                  className="text-xs border border-border-200 rounded-lg px-1.5 py-1 bg-white"
                >
                  <option value="active">Active</option>
                  <option value="inactive">Inactive</option>
                  <option value="suspended">Suspended</option>
                </select>
              </div>
            ),
          },
        ]}
        rows={licenses}
      />

      {/* --- Generate License --------------------------------------------- */}
      <Modal open={addOpen} onClose={() => setAddOpen(false)} title="Generate Client License" width="max-w-lg">
        <form onSubmit={handleGenerate} className="space-y-4">
          <div>
            <label className="text-sm font-medium text-ink-900 block mb-1.5">Company</label>
            <select
              value={form.companyId}
              onChange={(e) => setForm((f) => ({ ...f, companyId: e.target.value }))}
              className="input-field"
            >
              {companies.map((c) => (
                <option key={c.id} value={c.id}>
                  {c.name}
                </option>
              ))}
              <option value={CREATE_NEW}>+ Create new company</option>
            </select>
          </div>

          {form.companyId === CREATE_NEW && (
            <div>
              <label className="text-sm font-medium text-ink-900 block mb-1.5">New company name</label>
              <input
                required
                value={form.newCompanyName}
                onChange={(e) => setForm((f) => ({ ...f, newCompanyName: e.target.value }))}
                placeholder="e.g. Acme Corp"
                className="input-field"
              />
            </div>
          )}

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="text-sm font-medium text-ink-900 block mb-1.5">Max cameras</label>
              <input
                type="number"
                min={1}
                required
                value={form.maxCameras}
                onChange={(e) => setForm((f) => ({ ...f, maxCameras: e.target.value }))}
                className="input-field"
              />
            </div>
            <div>
              <label className="text-sm font-medium text-ink-900 block mb-1.5">Label (optional)</label>
              <input
                value={form.label}
                onChange={(e) => setForm((f) => ({ ...f, label: e.target.value }))}
                placeholder="e.g. Main site"
                className="input-field"
              />
            </div>
          </div>

          <div>
            <label className="text-sm font-medium text-ink-900 block mb-1.5">Expiry date (optional)</label>
            <input
              type="date"
              value={form.expiryDate}
              onChange={(e) => setForm((f) => ({ ...f, expiryDate: e.target.value }))}
              className="input-field"
            />
            <p className="text-xs text-slate-400 mt-1.5">
              Leave blank for no expiry. Past this date, login is blocked automatically even if status still shows Active.
            </p>
          </div>

          <div className="pt-1">
            <p className="text-sm font-semibold text-ink-900 mb-3">Portal login</p>
            <p className="text-xs text-slate-500 -mt-2 mb-3">
              This is how the client signs in to their portal from any device — see{" "}
              <span className="font-mono">/client-login</span>.
            </p>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="text-sm font-medium text-ink-900 block mb-1.5">
                  Username <span className="text-danger-500">*</span>
                </label>
                <input
                  required
                  value={form.username}
                  onChange={(e) => setForm((f) => ({ ...f, username: e.target.value }))}
                  placeholder="e.g. acme-client"
                  className="input-field"
                />
              </div>
              <div>
                <label className="text-sm font-medium text-ink-900 block mb-1.5">
                  Password <span className="text-danger-500">*</span>
                </label>
                <PasswordField
                  required
                  minLength={6}
                  value={form.password}
                  onChange={(e) => setForm((f) => ({ ...f, password: e.target.value }))}
                  placeholder="At least 6 characters"
                />
              </div>
            </div>
          </div>

          <div>
            <label className="text-sm font-medium text-ink-900 block mb-1.5">Features</label>
            <div className="grid grid-cols-2 gap-x-4 gap-y-2 rounded-xl border border-border-200 p-3">
              {catalog.map((f) => (
                <label key={f.key} className="flex items-center gap-2 text-sm cursor-pointer">
                  <input
                    type="checkbox"
                    checked={form.featureKeys.includes(f.key)}
                    onChange={() => toggleFormFeature(f.key)}
                    className="accent-brand-500"
                  />
                  {f.label}
                </label>
              ))}
            </div>
          </div>

          <div className="flex items-center gap-3 pt-1">
            <button type="button" onClick={() => setAddOpen(false)} className="btn-secondary flex-1">
              Cancel
            </button>
            <button type="submit" className="btn-primary flex-1">
              Generate
            </button>
          </div>
        </form>
      </Modal>

      {/* --- Edit license (label / max cameras / expiry) --------------------- */}
      <Modal open={!!editFor} onClose={() => setEditFor(null)} title="Edit Client License" width="max-w-md">
        {editFor && (
          <form onSubmit={saveEdit} className="space-y-4">
            <div>
              <label className="text-sm font-medium text-ink-900 block mb-1.5">Label</label>
              <input
                value={editForm.label}
                onChange={(e) => setEditForm((f) => ({ ...f, label: e.target.value }))}
                placeholder="e.g. Main site"
                className="input-field"
              />
            </div>
            <div>
              <label className="text-sm font-medium text-ink-900 block mb-1.5">Max cameras</label>
              <input
                type="number"
                min={editFor.cameras_assigned}
                required
                value={editForm.maxCameras}
                onChange={(e) => setEditForm((f) => ({ ...f, maxCameras: e.target.value }))}
                className="input-field"
              />
              {editFor.cameras_assigned > 0 && (
                <p className="text-xs text-slate-400 mt-1.5">
                  Can't go below {editFor.cameras_assigned} — that many camera(s) are already assigned.
                </p>
              )}
            </div>
            <div>
              <label className="text-sm font-medium text-ink-900 block mb-1.5">Expiry date</label>
              <input
                type="date"
                value={editForm.expiryDate}
                onChange={(e) => setEditForm((f) => ({ ...f, expiryDate: e.target.value }))}
                className="input-field"
              />
              <p className="text-xs text-slate-400 mt-1.5">
                Leave blank to keep the current expiry ({formatExpiry(editFor)}).
              </p>
            </div>
            <div className="flex items-center gap-3 pt-1">
              <button type="button" onClick={() => setEditFor(null)} className="btn-secondary flex-1">
                Cancel
              </button>
              <button type="submit" className="btn-primary flex-1">
                Save
              </button>
            </div>
          </form>
        )}
      </Modal>

      {/* --- Assign cameras ------------------------------------------------ */}
      <Modal open={!!assignFor} onClose={() => setAssignFor(null)} title="Assign Cameras" width="max-w-lg">
        {assignFor && (
          <div className="space-y-4">
            <p className="text-sm text-slate-500">
              {assignSelected.length} / {assignFor.max_cameras} cameras selected for{" "}
              <span className="font-mono">{assignFor.username || assignFor.label || companyName(assignFor.company_id)}</span>
            </p>
            <div className="divide-y divide-[#f1f2f7] max-h-72 overflow-y-auto rounded-xl border border-border-200">
              {allCameras.map((cam) => {
                const checked = assignSelected.includes(cam.id);
                const disableNew = !checked && assignSelected.length >= assignFor.max_cameras;
                return (
                  <label
                    key={cam.id}
                    className={`flex items-center gap-3 px-3 py-2.5 text-sm ${disableNew ? "opacity-40" : "cursor-pointer"}`}
                  >
                    <input
                      type="checkbox"
                      checked={checked}
                      disabled={disableNew}
                      onChange={() => toggleAssignCamera(cam.id)}
                      className="accent-brand-500"
                    />
                    <span className="flex-1">{cam.label}</span>
                    <span className="text-xs text-slate-400">{cam.site}</span>
                  </label>
                );
              })}
            </div>
            <div className="flex items-center gap-3 pt-1">
              <button type="button" onClick={() => setAssignFor(null)} className="btn-secondary flex-1">
                Cancel
              </button>
              <button type="button" onClick={saveAssign} className="btn-primary flex-1">
                Save
              </button>
            </div>
          </div>
        )}
      </Modal>

      {/* --- Manage license features ---------------------------------------- */}
      <Modal open={!!featuresFor} onClose={() => setFeaturesFor(null)} title="License Features" width="max-w-md">
        {featuresFor && (
          <div className="space-y-4">
            <p className="text-sm text-slate-500">
              Features enabled for{" "}
              <span className="font-mono">{featuresFor.username || featuresFor.label || companyName(featuresFor.company_id)}</span>
            </p>
            <div className="grid grid-cols-1 gap-2 rounded-xl border border-border-200 p-3">
              {catalog.map((f) => (
                <label key={f.key} className="flex items-center gap-2 text-sm cursor-pointer">
                  <input
                    type="checkbox"
                    checked={featuresSelected.includes(f.key)}
                    onChange={() => toggleFeaturesSelected(f.key)}
                    className="accent-brand-500"
                  />
                  {f.label}
                </label>
              ))}
            </div>
            <div className="flex items-center gap-3 pt-1">
              <button type="button" onClick={() => setFeaturesFor(null)} className="btn-secondary flex-1">
                Cancel
              </button>
              <button type="button" onClick={saveFeatures} className="btn-primary flex-1">
                Save
              </button>
            </div>
          </div>
        )}
      </Modal>

      {/* --- QR code --------------------------------------------------------- */}
      <Modal open={!!qrFor} onClose={() => setQrFor(null)} title="License QR Code" width="max-w-sm">
        {qrFor && (
          <div className="space-y-4 text-center">
            <img src={api.licenseQrUrl(qrFor.id)} alt="License QR code" className="mx-auto w-48 h-48" />
            <p className="font-mono text-sm text-ink-900">{qrFor.license_key}</p>
            <p className="text-xs text-slate-400">Right-click the image to save or print it.</p>
          </div>
        )}
      </Modal>

      {/* --- Portal login (username/password) ------------------------------- */}
      <Modal
        open={!!credsFor}
        onClose={closeCredentials}
        title={credsSaved ? "Portal Login Saved" : credsFor?.username ? "Reset Portal Login" : "Set Portal Login"}
        width="max-w-sm"
      >
        {credsFor && credsSaved && (
          <div className="space-y-4">
            <p className="text-sm text-slate-500 -mt-1">
              Save this password now — it can't be shown again. Only a reset (this same screen) can recover access,
              never a "view current password" option, because it's never stored anywhere in readable form.
            </p>
            <CredentialBox
              username={credsForm.username}
              password={credsForm.password}
              portalUrl={portalUrl(credsFor.company_portal_slug)}
              onCopy={copyToClipboard}
            />
            <button type="button" onClick={closeCredentials} className="btn-primary w-full">
              Done
            </button>
          </div>
        )}

        {credsFor && !credsSaved && (
          <form onSubmit={saveCredentials} className="space-y-4">
            <p className="text-sm text-slate-500 -mt-1">
              For <span className="font-mono">{credsFor.label || companyName(credsFor.company_id)}</span> to sign in at{" "}
              <span className="font-mono">/client-login</span> from any device.
            </p>
            <div>
              <label className="text-sm font-medium text-ink-900 block mb-1.5">Username</label>
              <input
                required
                value={credsForm.username}
                onChange={(e) => setCredsForm((f) => ({ ...f, username: e.target.value }))}
                className="input-field"
              />
            </div>
            <div>
              <label className="text-sm font-medium text-ink-900 block mb-1.5">
                {credsFor.username ? "New password" : "Password"}
              </label>
              <PasswordField
                required
                minLength={6}
                value={credsForm.password}
                onChange={(e) => setCredsForm((f) => ({ ...f, password: e.target.value }))}
                placeholder="At least 6 characters"
              />
              <p className="text-xs text-slate-400 mt-1.5">
                Passwords can't be viewed later, only reset — you'll get one chance to copy this one after saving.
              </p>
            </div>
            <div className="flex items-center gap-3 pt-1">
              <button type="button" onClick={closeCredentials} className="btn-secondary flex-1">
                Cancel
              </button>
              <button type="submit" className="btn-primary flex-1">
                Save
              </button>
            </div>
          </form>
        )}
      </Modal>

      {/* --- Reveal-once: the password just set when generating a license --- */}
      <Modal open={!!revealCreds} onClose={() => setRevealCreds(null)} title="Client License Created" width="max-w-sm">
        {revealCreds && (
          <div className="space-y-4">
            <p className="text-sm text-slate-500 -mt-1">
              Save this password now — it can't be shown again. If it's lost, use the reset-login action on this
              license to set a new one.
            </p>
            <CredentialBox
              username={revealCreds.username}
              password={revealCreds.password}
              portalUrl={revealCreds.portal}
              onCopy={copyToClipboard}
            />
            <button type="button" onClick={() => setRevealCreds(null)} className="btn-primary w-full">
              Done
            </button>
          </div>
        )}
      </Modal>
    </div>
  );
}

function CredentialBox({ username, password, portalUrl: url, onCopy }) {
  return (
    <div className="space-y-2 rounded-xl border border-border-200 bg-[#f8f9fc] p-3.5">
      {url && (
        <div className="flex items-center justify-between gap-3 pb-2 border-b border-border-200">
          <div className="min-w-0">
            <p className="text-xs text-slate-400">Client Portal</p>
            <p className="font-mono text-xs text-ink-900 truncate">{url}</p>
          </div>
          <button type="button" onClick={() => onCopy(url)} className="text-xs font-medium text-brand-600 shrink-0">
            Copy
          </button>
        </div>
      )}
      <div className="flex items-center justify-between gap-3">
        <div className="min-w-0">
          <p className="text-xs text-slate-400">Username</p>
          <p className="font-mono text-sm text-ink-900 truncate">{username}</p>
        </div>
        <button type="button" onClick={() => onCopy(username)} className="text-xs font-medium text-brand-600 shrink-0">
          Copy
        </button>
      </div>
      <div className="flex items-center justify-between gap-3 pt-2 border-t border-border-200">
        <div className="min-w-0">
          <p className="text-xs text-slate-400">Password</p>
          <p className="font-mono text-sm text-ink-900 truncate">{password}</p>
        </div>
        <button type="button" onClick={() => onCopy(password)} className="text-xs font-medium text-brand-600 shrink-0">
          Copy
        </button>
      </div>
    </div>
  );
}
