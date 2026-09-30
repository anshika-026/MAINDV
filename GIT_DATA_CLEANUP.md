# Git Data Cleanup: Biometric and Sensitive Data in History

**Status: REQUIRES HUMAN DECISION.** The current tree has been fixed: the
files below are no longer tracked and `.gitignore` blocks them from coming
back. **History has not been rewritten.** Every file below can still be
recovered from any clone of the repository until the procedure in this
document is carried out and every clone is replaced.

## 1. What was found

Audit run on 2026-09-29 against all local and remote-tracking refs
(`git log --all --diff-filter=A --name-only`).

| Path in history | Count | Branches | Why it is sensitive |
|---|---|---|---|
| `backend/data/face_captures/*.jpg` | 7,432 files added over time (3,716 still tracked on `main` until this branch) | `main`, `unique-footfall`, others | Face crops of real, identifiable people captured from CCTV. **Biometric personal data.** |
| `backend/data/face_training/**` | 5,363 | `unique-footfall` | Face images labeled with employee IDs, which ties identity to biometrics. Includes `classifier.joblib`, a model trained on employees' faces. |
| `backend/data/face_enroll/*` | 100 | `main` (3 still tracked), `unique-footfall` | Enrollment photos of named employees. |
| `backend/data/app.db` | 1 (89 MB) | `unique-footfall` | Full database: face embeddings (biometric templates), employee roster, attendance history, camera hosts **and camera passwords in plaintext**, client license usernames (password hashes), session tokens. |
| `attendance.db`, `backend/attendance.db` | 2 | history | Attendance records of named people. |
| `backend/.env` | 2 versions | `main` | Machine configuration. No secrets found in the tracked versions today, but it is the file where secrets are meant to live, so tracking it invites leaks. |
| `models/*.pt` (`imc_general_best.pt`, `office_fixtures.pt`, `yolo26s.pt`) | 3 | `object-detection` | Not personal data, but large binaries that bloat every clone. Possibly proprietary (custom-trained). |
| `backend/models/emotion_model_v3.keras` | 1 | `main` | Not personal data. Left tracked (3.7 MB) because deploys use it. Move it to model provisioning later if wanted. |

Current `main` also still points to commits containing all of the above.
Removing them from the tree on this branch does **not** remove them from
those commits.

## 2. Decide first (human)

1. **Was the GitHub repo (`anshika-026/MAINDV`, and the older
   `anshika-026/https---github.com-anshika-026-MAIN-DV`) ever public, forked,
   or shared with anyone outside the project team?** Check repository
   settings, the forks list, collaborators, and deploy keys. If yes, this is a
   personal-data exposure: involve whoever owns data protection (India DPDP
   Act 2023 / GDPR if EU subjects), because rewriting history does not
   recall copies that were already taken.
2. **Who has clones?** Every developer laptop, the EC2 server
   (`/home/ubuntu/Deco-vision`), CI caches, and any backups of those.
3. **Pick a window.** The rewrite changes every commit ID. Open PRs and
   unpushed work have to be rebased onto the new history.

## 3. Rotate credentials (do this regardless of the rewrite)

Rewriting history doesn't make a leaked secret safe again.

- **Camera / NVR passwords.** `app.db` in history contains every camera's
  RTSP username and password in plaintext, and the same credential is used
  across all cameras. Change them on the NVR/cameras, then update each
  camera in Camera Management.
- **Client license passwords.** Hashes are in the historical `app.db`. Reset
  them via License Management > credentials (this also signs clients out).
- **Sessions.** Any session token in the historical DB is dead once the new
  auth migration runs (admin sessions are revoked automatically; client
  sessions expire). To be thorough, run `python -m app.manage purge-sessions`
  and reset client credentials.
- **EC2 deploy key (`EC2_PRIVATE_KEY` secret).** Rotate it if the repo was
  ever public or the key was ever shared.
- **`JWT_SECRET`.** Set a new one in production (license QR codes need
  re-issuing afterwards).

## 4. Backup before rewriting

```bash
# A complete mirror of the current remote, including every branch and tag.
git clone --mirror https://github.com/anshika-026/MAINDV.git MAINDV-backup-$(date +%F).git
# Store it encrypted and access-controlled. It contains the biometric data
# being removed. Delete it once the rewrite has been verified and the
# retention decision has been made.
tar czf MAINDV-backup-$(date +%F).tgz MAINDV-backup-$(date +%F).git
```

## 5. Rewrite procedure (git-filter-repo)

`git filter-repo` is the maintained tool (`pip install git-filter-repo`).
Run it on a **fresh mirror clone**, never in a working copy.

```bash
git clone --mirror https://github.com/anshika-026/MAINDV.git MAINDV-clean.git
cd MAINDV-clean.git

git filter-repo --force \
  --path backend/data/ \
  --path backend/data_backup_2026-09-29/ \
  --path backend/.env \
  --path attendance.db \
  --path backend/attendance.db \
  --path-glob '*.db' \
  --path-glob '*.db-wal' \
  --path-glob '*.db-shm' \
  --path-glob '*.joblib' \
  --path-glob '*.joblib.bak-*' \
  --invert-paths

# In the same rewrite: strip AI-assistant attribution trailers from commit
# messages (six commits on main end with "Co-Authored-By: Claude ...").
git filter-repo --force --message-callback '
import re
return re.sub(rb"\n*Co-Authored-By: Claude[^\n]*", b"", message).rstrip() + b"\n"
'

# Optional: also drop large model binaries from history.
# git filter-repo --force --path-glob 'models/*.pt' --path-glob '*.pth' --path-glob '*.onnx' --invert-paths

# Verify nothing sensitive remains in ANY ref:
git log --all --name-only --pretty=format: | grep -E 'backend/data/|\.db$|\.joblib|backend/\.env$' | sort -u
# (must print nothing)

git count-objects -vH   # size should drop sharply
```

## 6. Publish the rewritten history (force-push)

**Force-push implications:**
- Every commit hash changes. Everyone must re-clone; a normal `git pull`
  would merge the old history (and the data) straight back in.
- Open pull requests become unmergeable and must be recreated.
- `main` is protected by the auto-deploy workflow. Disable it (or merge the
  new CI/CD workflow from this branch first) so the force-push doesn't
  trigger a production deploy.
- Branch protection must be temporarily relaxed to allow the force-push,
  then restored.

```bash
cd MAINDV-clean.git
git push --force --mirror origin
```

Then on GitHub:
1. Settings > Danger zone: check for forks. Forks keep the old history; ask
   their owners to delete them.
2. Contact GitHub Support to purge cached views of the removed objects
   (commits can stay reachable by SHA until GitHub runs garbage collection).
   Ask them to "remove cached views and dereferenced objects". Include the
   list of affected commit SHAs.

## 7. After the rewrite

On **every** clone (developer laptops, the EC2 server):

```bash
# Keep local runtime data OUTSIDE the old clone first:
mv MAINDV/backend/data /safe/place/data
mv MAINDV/backend/.env /safe/place/.env
rm -rf MAINDV
git clone https://github.com/anshika-026/MAINDV.git
mv /safe/place/data MAINDV/backend/data
mv /safe/place/.env MAINDV/backend/.env
```

On the EC2 server the application data directory should live **outside the
repository** (for example `/var/lib/decovision`, via `DATA_DIR`; see
DEPLOYMENT.md), so this is a one-time move.

## 8. Team coordination checklist

- [ ] Decide on public-exposure / breach handling (section 2)
- [ ] Rotate camera, client, deploy-key and JWT credentials (section 3)
- [ ] Announce a freeze: no pushes during the rewrite window
- [ ] Take and secure the mirror backup (section 4)
- [ ] Disable the auto-deploy workflow or merge the new CI/CD first
- [ ] Rewrite and verify (section 5)
- [ ] Force-push, restore branch protection (section 6)
- [ ] Everyone re-clones; server re-cloned with data moved outside the repo (section 7)
- [ ] Ask GitHub Support to purge cached objects; check for forks
- [ ] Delete stale branches that only carried data (`unique-footfall` data
      commits, `object-detection` weights) or confirm they were rewritten
- [ ] Destroy the backup mirror once no longer needed (it contains the data)
