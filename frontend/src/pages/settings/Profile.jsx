import { useEffect, useState } from "react";
import * as api from "../../api/client";

export default function Profile() {
  const [profile, setProfile] = useState(null);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    api.getProfile().then(setProfile);
  }, []);

  if (!profile) return null;

  async function handleSave(e) {
    e.preventDefault();
    setSaving(true);
    setSaved(false);
    await api.updateProfile(profile);
    setSaving(false);
    setSaved(true);
    setTimeout(() => setSaved(false), 2000);
  }

  return (
    <div className="card p-6">
      <h2 className="text-base font-semibold text-ink-900 mb-5">Profile</h2>
      <form onSubmit={handleSave} className="space-y-4">
        <div className="flex items-center gap-4">
          <span className="w-14 h-14 rounded-full bg-brand-500 text-white flex items-center justify-center text-lg font-semibold">
            {profile.name.slice(0, 1)}
          </span>
          <button type="button" className="btn-secondary text-sm">
            Change picture
          </button>
        </div>
        <div>
          <label className="text-sm font-medium text-ink-900 block mb-1.5">Email address</label>
          <input value={profile.email} disabled className="input-field bg-[#f5f6fa] text-slate-500" />
        </div>
        <div>
          <label className="text-sm font-medium text-ink-900 block mb-1.5">Name</label>
          <input
            value={profile.name}
            onChange={(e) => setProfile((p) => ({ ...p, name: e.target.value }))}
            className="input-field"
          />
        </div>
        <div>
          <label className="text-sm font-medium text-ink-900 block mb-1.5">Mobile</label>
          <input
            value={profile.mobile}
            onChange={(e) => setProfile((p) => ({ ...p, mobile: e.target.value }))}
            className="input-field"
          />
        </div>
        <div>
          <label className="text-sm font-medium text-ink-900 block mb-1.5">Role</label>
          <input value={profile.role} disabled className="input-field bg-[#f5f6fa] text-slate-500" />
        </div>
        <div className="flex items-center gap-3 pt-2">
          <button type="submit" disabled={saving} className="btn-primary">
            {saving ? "Saving..." : "Update"}
          </button>
          {saved && <span className="text-sm text-success-600">Saved</span>}
        </div>
      </form>
    </div>
  );
}
