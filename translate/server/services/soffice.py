"""LibreOffice-based .docx -> .pdf conversion for exact-layout previews.

Databricks Apps supports no system packages (no apt, no Dockerfile, no init
scripts), so LibreOffice cannot be *installed* — but a portable LibreOffice
tree shipped as a tar.gz in a UC Volume can be downloaded and extracted into
the app's writable temp dir at first use, then run headless as a plain
subprocess. That is what this module does. Discovery order:

1. ``SOFFICE_PATH`` — explicit path to an existing ``soffice`` binary
   (local dev: Windows install, or any pre-provisioned binary).
2. A previously extracted portable tree in the local extract dir (fast path
   after the first request following a restart).
3. ``SOFFICE_ARCHIVE_VOLUME_PATH`` — UC Volume path of a portable
   LibreOffice ``.tar.gz`` (see utils/soffice_packaging/) to download and
   extract once per process lifetime.
4. Well-known system locations (PATH, Program Files) — local dev fallback.

Everything here is synchronous (subprocess + file IO); callers wrap calls in
``asyncio.to_thread`` like the rest of the volume plumbing in translate.py.
If no engine can be found the caller gets ``SofficeUnavailable`` with the
reason — the preview endpoint then falls back to client-side docx-preview.
"""

import hashlib
import logging
import os
import re
import shutil
import subprocess
import tarfile
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_CONVERT_TIMEOUT_S = int(os.getenv('SOFFICE_CONVERT_TIMEOUT_S', '180'))
# Portable-tree extraction can take a while for a ~700 MB tree; one-time cost.
_EXTRACT_TIMEOUT_HINT_S = 600

# soffice is single-document-per-invocation here (isolated profiles), but each
# invocation is CPU/RAM hungry — cap parallel conversions.
_convert_semaphore = threading.Semaphore(int(os.getenv('SOFFICE_MAX_CONCURRENT', '2')))

_discovery_lock = threading.Lock()
_discovered_path: Optional[str] = None
_discovery_error: Optional[str] = None
_engine_version: Optional[str] = None


class SofficeUnavailable(RuntimeError):
    """No usable LibreOffice binary could be found/provisioned."""


class SofficeConversionError(RuntimeError):
    """LibreOffice ran but did not produce the expected PDF."""


def _extract_root() -> Path:
    base = Path(os.getenv('SOFFICE_EXTRACT_DIR', '') or (Path(tempfile.gettempdir()) / 'latlang_soffice'))
    # Version the extract dir by archive basename: uploading a fixed archive
    # under a new filename guarantees a fresh extraction even if /tmp survived
    # the app restart (the old broken tree would otherwise be reused forever).
    archive = os.getenv('SOFFICE_ARCHIVE_VOLUME_PATH', '').strip()
    if archive:
        stem = re.sub(r'[^A-Za-z0-9._-]', '_', archive.rsplit('/', 1)[-1]).removesuffix('.tar.gz')
        return base / stem
    return base


def _cache_dir() -> Path:
    d = Path(tempfile.gettempdir()) / 'latlang_pdf_cache'
    d.mkdir(parents=True, exist_ok=True)
    return d


def _find_binary_in_tree(root: Path) -> Optional[Path]:
    """Locate program/soffice inside an extracted portable tree."""
    for candidate in root.glob('**/program/soffice'):
        if candidate.is_file():
            return candidate
    # Windows portable tree (local testing with an msiexec /a admin image).
    # Prefer soffice.com (console subsystem — sane stdout/exit handling).
    for pattern in ('**/program/soffice.com', '**/program/soffice.exe'):
        for candidate in root.glob(pattern):
            if candidate.is_file():
                return candidate
    return None


def _mark_executable(root: Path) -> None:
    if os.name == 'nt':
        return
    for p in root.glob('**/program/*'):
        try:
            p.chmod(p.stat().st_mode | 0o755)
        except OSError:
            pass


def _extract_archive_from_volume(volume_path: str) -> Path:
    """Download the portable tar.gz from the UC Volume and extract it locally.

    Runs at most once per process (marker file); a crashed half-extract is
    detected by the missing marker and redone from scratch.
    """
    from databricks.sdk import WorkspaceClient

    root = _extract_root()
    marker = root / '.extract_complete'
    if marker.exists():
        binary = _find_binary_in_tree(root)
        if binary:
            return binary
        # Marker without binary — corrupted; start over.
        shutil.rmtree(root, ignore_errors=True)

    if root.exists():
        shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True, exist_ok=True)

    started = time.monotonic()
    logger.info('soffice: downloading portable archive from %s', volume_path)
    w = WorkspaceClient()
    resp = w.files.download(volume_path)
    archive_path = root / 'archive.tar.gz'
    with open(archive_path, 'wb') as f:
        shutil.copyfileobj(resp.contents, f, length=8 * 1024 * 1024)
    logger.info(
        'soffice: archive downloaded (%.0f MB in %.0fs), extracting…',
        archive_path.stat().st_size / 1e6, time.monotonic() - started,
    )

    with tarfile.open(archive_path, 'r:gz') as tar:
        tar.extractall(root)
    archive_path.unlink(missing_ok=True)
    _mark_executable(root)

    binary = _find_binary_in_tree(root)
    if not binary:
        raise SofficeUnavailable(
            f'archive at {volume_path} extracted but contains no program/soffice binary'
        )
    marker.write_text('ok')
    logger.info('soffice: portable tree ready at %s (%.0fs total)', binary, time.monotonic() - started)
    return binary


def _well_known_paths() -> list:
    paths = []
    found = shutil.which('soffice')
    if found:
        paths.append(Path(found))
    if os.name == 'nt':
        for pf in (os.getenv('ProgramFiles'), os.getenv('ProgramFiles(x86)')):
            if pf:
                paths.append(Path(pf) / 'LibreOffice' / 'program' / 'soffice.exe')
    else:
        paths.extend([
            Path('/usr/bin/soffice'),
            Path('/opt/libreoffice/program/soffice'),
        ])
    return [p for p in paths if p.is_file()]


def _probe_version(binary: str) -> Optional[str]:
    """Run `soffice --headless --version`. Also serves as the executability
    check — a present-but-unrunnable binary (noexec mount, missing shared
    libs) fails here with a useful message instead of at first conversion.
    Headless + isolated profile: without them, a pristine Windows soffice.exe
    blocks on first-run profile/dialog handling."""
    try:
        with tempfile.TemporaryDirectory(prefix='soffice_probe_') as tmp:
            profile = Path(tmp) / 'profile'
            profile.mkdir()
            out = subprocess.run(
                [binary, f'-env:UserInstallation={profile.as_uri()}', '--headless', '--version'],
                capture_output=True, text=True, timeout=120,
                env=_subprocess_env(binary),
            )
            if out.returncode == 0:
                # Windows soffice.exe is a GUI-subsystem binary: rc=0 but no
                # console output. Success is rc=0; the version string is a bonus.
                return out.stdout.strip().splitlines()[0] if out.stdout.strip() else 'unknown version'
            raise SofficeUnavailable(
                f'{binary} --version failed (rc={out.returncode}): {(out.stderr or out.stdout).strip()[:500]}'
            )
    except (OSError, subprocess.TimeoutExpired) as e:
        raise SofficeUnavailable(f'{binary} not runnable: {e}')


def _subprocess_env(binary: Optional[str] = None) -> dict:
    env = dict(os.environ)
    # Headless-friendly defaults; harmless when already set.
    env.setdefault('HOME', tempfile.gettempdir())
    env.setdefault('SAL_USE_VCLPLUGIN', 'svp')  # pure-software rendering, no X11
    if binary and os.name != 'nt':
        # The portable tarball carries the Ubuntu system libs LibreOffice
        # needs but slim containers lack (libXinerama & co) inside program/.
        # $ORIGIN rpath covers the executables; LD_LIBRARY_PATH covers
        # anything dlopen'd through a different path.
        program_dir = str(Path(binary).parent)
        env['LD_LIBRARY_PATH'] = program_dir + (
            ':' + env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else ''
        )
    return env


def find_soffice() -> str:
    """Resolve (once per process) the soffice binary path. Raises SofficeUnavailable."""
    global _discovered_path, _discovery_error, _engine_version
    if _discovered_path:
        return _discovered_path
    with _discovery_lock:
        if _discovered_path:
            return _discovered_path

        errors = []

        explicit = os.getenv('SOFFICE_PATH', '').strip()
        if explicit:
            if Path(explicit).is_file():
                candidates = [Path(explicit)]
            else:
                errors.append(f'SOFFICE_PATH={explicit} does not exist')
                candidates = []
        else:
            candidates = []

        if not candidates:
            local = _find_binary_in_tree(_extract_root()) if (_extract_root() / '.extract_complete').exists() else None
            if local:
                candidates = [local]

        if not candidates:
            volume_archive = os.getenv('SOFFICE_ARCHIVE_VOLUME_PATH', '').strip()
            if volume_archive:
                try:
                    candidates = [_extract_archive_from_volume(volume_archive)]
                except Exception as e:
                    errors.append(f'volume archive: {e}')

        if not candidates:
            candidates = _well_known_paths()

        if not candidates:
            errors.append('no soffice binary found (set SOFFICE_PATH or SOFFICE_ARCHIVE_VOLUME_PATH)')
            _discovery_error = '; '.join(errors)
            raise SofficeUnavailable(_discovery_error)

        try:
            _engine_version = _probe_version(str(candidates[0]))
        except SofficeUnavailable as e:
            errors.append(str(e))
            _discovery_error = '; '.join(errors)
            raise SofficeUnavailable(_discovery_error)

        _discovered_path = str(candidates[0])
        _discovery_error = None
        logger.info('soffice: using %s (%s)', _discovered_path, _engine_version)
        return _discovered_path


def soffice_status() -> dict:
    """Non-raising status snapshot for the diagnostic endpoint."""
    try:
        path = find_soffice()
        return {'available': True, 'path': path, 'version': _engine_version}
    except SofficeUnavailable as e:
        return {'available': False, 'error': str(e)}


def convert_docx_to_pdf(docx_bytes: bytes, timeout_s: int = _CONVERT_TIMEOUT_S, suffix: str = '.docx') -> bytes:
    """Convert .docx (or any soffice-readable format — pass the real
    extension via ``suffix``, soffice picks its import filter from it) to
    PDF bytes.

    Results are cached on disk by content hash — the before/after preview of a
    finished job is immutable, so repeat views cost nothing.
    """
    digest = hashlib.sha256(docx_bytes).hexdigest()
    cached = _cache_dir() / f'{digest}.pdf'
    if cached.is_file() and cached.stat().st_size > 0:
        return cached.read_bytes()

    binary = find_soffice()

    with _convert_semaphore:
        with tempfile.TemporaryDirectory(prefix='soffice_job_') as tmp:
            tmp_path = Path(tmp)
            src = tmp_path / f'input{suffix if suffix.startswith(".") else "." + suffix}'
            src.write_bytes(docx_bytes)
            # Isolated profile per invocation — allows concurrent soffice
            # processes and avoids the shared-profile lock file entirely.
            profile = tmp_path / 'profile'
            profile.mkdir()
            cmd = [
                binary,
                f'-env:UserInstallation={profile.as_uri()}',
                '--headless', '--norestore', '--nolockcheck', '--nodefault',
                '--convert-to', 'pdf',
                '--outdir', str(tmp_path),
                str(src),
            ]
            started = time.monotonic()
            try:
                out = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=timeout_s,
                    env=_subprocess_env(binary),
                )
            except subprocess.TimeoutExpired:
                raise SofficeConversionError(f'conversion timed out after {timeout_s}s')

            pdf = tmp_path / 'input.pdf'
            if out.returncode != 0 or not pdf.is_file():
                raise SofficeConversionError(
                    f'soffice rc={out.returncode}: {(out.stderr or out.stdout).strip()[:800]}'
                )
            pdf_bytes = pdf.read_bytes()
            logger.info(
                'soffice: converted %.0f kB docx -> %.0f kB pdf in %.1fs',
                len(docx_bytes) / 1e3, len(pdf_bytes) / 1e3, time.monotonic() - started,
            )

    tmp_cache = cached.with_suffix('.tmp')
    try:
        tmp_cache.write_bytes(pdf_bytes)
        tmp_cache.replace(cached)
    except OSError:
        pass  # cache is best-effort
    return pdf_bytes
