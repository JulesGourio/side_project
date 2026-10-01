"""Build a portable LibreOffice Linux x86_64 tarball for Databricks Apps.

Databricks Apps has no system package manager, so the app provisions
LibreOffice at runtime from a plain tar.gz stored in a UC Volume (see
server/services/soffice.py). This script produces that tar.gz from the
official TDF "Linux x86-64 (deb)" bundle, without needing dpkg/ar — .deb
files are `ar` archives (a trivial fixed-header format parsed here by hand)
wrapping a data.tar.{xz,zst,gz} whose contents Python's tarfile can read.

Steps (all pure Python, runs fine on Windows):
  1. (caller) download LibreOffice_<ver>_Linux_x86-64_deb.tar.gz
  2. extract every bundled .deb's data tarball
  3. merge their ./opt/libreoffice<ver>/ trees into one directory
  4. re-tar with normalized exec permissions (Windows drops them otherwise)

Usage:
  python utils/soffice_packaging/package_libreoffice.py <downloaded.tar.gz> <output.tar.gz>

Then upload to the UC Volume and point the app at it:
  databricks fs cp <output.tar.gz> dbfs:/Volumes/<catalog>/<schema>/<volume>/libreoffice/libreoffice-linux-x64.tar.gz
  app.yaml / deploy: SOFFICE_ARCHIVE_VOLUME_PATH=/Volumes/<...>/libreoffice/libreoffice-linux-x64.tar.gz
"""

import gzip
import io
import lzma
import shutil
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

# System libraries the TDF LibreOffice build links against that slim
# container images (Databricks Apps) don't ship. Their .so files are merged
# into program/ (covered by the executables' $ORIGIN rpath, plus
# LD_LIBRARY_PATH set by server/services/soffice.py). Pulled from the Ubuntu
# jammy (22.04) archive — the Databricks Apps base image generation; these
# X/rendering libs have tiny, ancient glibc requirements, so a slightly
# different container release is fine.
_SYSLIB_PACKAGES = [
    'libx11-6', 'libxcb1', 'libxau6', 'libxdmcp6', 'libxext6', 'libxinerama1',
    'libxrender1', 'libxrandr2', 'libxcursor1', 'libxfixes3', 'libxi6',
    'libxcomposite1', 'libxdamage1', 'libxtst6', 'libsm6', 'libice6',
    'libfontconfig1', 'libfreetype6', 'libpng16-16', 'libexpat1',
    'libcups2', 'libdbus-1-3', 'libglib2.0-0', 'libcairo2', 'libpixman-1-0',
    'libxcb-shm0', 'libxcb-render0', 'libavahi-client3', 'libavahi-common3',
    'libgssapi-krb5-2', 'libkrb5-3', 'libk5crypto3', 'libkrb5support0',
    'libcom-err2', 'libkeyutils1', 'libbrotli1',
    # NSS crypto stack — soffice.bin links libnss3/libssl3 (the NSS one, not
    # OpenSSL) directly; TDF does not bundle it.
    'libnss3', 'libnspr4', 'libsqlite3-0',
    # Full transitive closure from utils/soffice_packaging/scan_needed_libs.py
    # (pure-Python DT_NEEDED scan of the tree), minus GUI-plugin-only deps
    # (Qt5/KF5/GTK3/GStreamer — never loaded with SAL_USE_VCLPLUGIN=svp):
    'libssl3', 'libcrypt1', 'libbsd0', 'libmd0', 'libx11-xcb1', 'libffi8',
    'libmount1', 'libblkid1', 'libselinux1', 'libpcre3', 'libuuid1',
    'libgnutls30', 'libidn2-0', 'libunistring2', 'libtasn1-6', 'libnettle8',
    'libhogweed6', 'libp11-kit0', 'libgmp10',
    'libsystemd0', 'liblzma5', 'libzstd1', 'liblz4-1', 'libcap2',
    'libgcrypt20', 'libgpg-error0',
]
_UBUNTU_MIRROR = 'http://archive.ubuntu.com/ubuntu'


def iter_ar_members(data: bytes):
    """Minimal ar(1) reader: yields (name, payload) for each member."""
    assert data[:8] == b'!<arch>\n', 'not an ar archive'
    off = 8
    while off < len(data):
        header = data[off:off + 60]
        if len(header) < 60:
            break
        name = header[0:16].decode('ascii').strip().rstrip('/')
        size = int(header[48:58].decode('ascii').strip())
        payload = data[off + 60:off + 60 + size]
        yield name, payload
        off += 60 + size + (size % 2)  # members are 2-byte aligned


def extract_deb(deb_bytes: bytes, dest: Path) -> None:
    for name, payload in iter_ar_members(deb_bytes):
        if not name.startswith('data.tar'):
            continue
        if name.endswith('.xz'):
            stream = io.BytesIO(lzma.decompress(payload))
        elif name.endswith('.gz') or name.endswith('.tar'):
            stream = io.BytesIO(payload)
        elif name.endswith('.zst'):
            import zstandard  # only needed if TDF ever switches to zstd
            stream = io.BytesIO(zstandard.ZstdDecompressor().decompress(payload, max_output_size=2**33))
        else:
            raise RuntimeError(f'unsupported data member {name}')
        with tarfile.open(fileobj=stream) as tar:
            tar.extractall(dest)
        return
    raise RuntimeError('no data.tar member found in .deb')


def _load_ubuntu_index() -> dict:
    """package name -> pool Filename, from the jammy + jammy-updates indexes
    (updates last so security-patched versions win)."""
    filenames: dict = {}
    for dist in ('jammy', 'jammy-updates'):
        url = f'{_UBUNTU_MIRROR}/dists/{dist}/main/binary-amd64/Packages.gz'
        print(f'fetching package index {url} …')
        with urllib.request.urlopen(url, timeout=120) as resp:
            data = gzip.decompress(resp.read()).decode('utf-8', errors='replace')
        name = None
        for line in data.splitlines():
            if line.startswith('Package: '):
                name = line[9:].strip()
            elif line.startswith('Filename: ') and name:
                filenames[name] = line[10:].strip()
    return filenames


def add_system_libs(rootfs_program_dir: Path) -> None:
    """Download the _SYSLIB_PACKAGES debs and merge their shared objects into
    the LibreOffice program/ directory."""
    index = _load_ubuntu_index()
    missing = [p for p in _SYSLIB_PACKAGES if p not in index]
    if missing:
        raise SystemExit(f'packages not found in Ubuntu index: {missing}')

    with tempfile.TemporaryDirectory(prefix='syslibs_') as tmp:
        tmp_path = Path(tmp)
        for pkg in _SYSLIB_PACKAGES:
            url = f'{_UBUNTU_MIRROR}/{index[pkg]}'
            print(f'  syslib {pkg}: {url.rsplit("/", 1)[-1]}')
            with urllib.request.urlopen(url, timeout=120) as resp:
                deb_bytes = resp.read()
            extract_deb(deb_bytes, tmp_path)

        count = 0
        for so in tmp_path.glob('**/*.so*'):
            if not so.is_file() or so.is_symlink():
                # tarfile extraction preserves symlinks as symlinks — resolve
                # by copying the target under the link's own name below.
                pass
            dest = rootfs_program_dir / so.name
            if dest.exists():
                continue
            if so.is_symlink():
                target = so.resolve()
                if target.is_file():
                    shutil.copy2(target, dest)
                    count += 1
            else:
                shutil.copy2(so, dest)
                count += 1
        print(f'merged {count} system libraries into program/')


def main(bundle_path: str, output_path: str) -> None:
    bundle = Path(bundle_path)
    output = Path(output_path)
    with tempfile.TemporaryDirectory(prefix='lo_pkg_') as tmp:
        tmp_path = Path(tmp)
        debs_dir = tmp_path / 'debs'
        rootfs = tmp_path / 'rootfs'
        debs_dir.mkdir()
        rootfs.mkdir()

        print(f'extracting bundle {bundle} …')
        with tarfile.open(bundle) as tar:
            tar.extractall(debs_dir)

        debs = sorted(debs_dir.glob('**/*.deb'))
        if not debs:
            raise SystemExit('no .deb files found in bundle')
        print(f'unpacking {len(debs)} debs …')
        for deb in debs:
            extract_deb(deb.read_bytes(), rootfs)

        # The debs install into ./opt/libreofficeXX.Y — keep only that tree.
        opt_trees = list((rootfs / 'opt').glob('libreoffice*'))
        if len(opt_trees) != 1:
            raise SystemExit(f'expected exactly one opt/libreoffice* tree, got {opt_trees}')
        lo_root = opt_trees[0]
        soffice = lo_root / 'program' / 'soffice'
        if not soffice.is_file():
            raise SystemExit('program/soffice missing from merged tree')

        add_system_libs(lo_root / 'program')

        print(f'building {output} …')
        output.parent.mkdir(parents=True, exist_ok=True)

        def normalize(info: tarfile.TarInfo) -> tarfile.TarInfo:
            # Windows loses POSIX modes; grant exec broadly — a doc-conversion
            # sandbox binary tree, not a security boundary.
            info.mode = 0o755 if info.isdir() or info.isfile() else info.mode
            info.uname = info.gname = ''
            info.uid = info.gid = 0
            return info

        with tarfile.open(output, 'w:gz', compresslevel=6) as tar:
            tar.add(lo_root, arcname=lo_root.name, filter=normalize)

        print(f'done: {output} ({output.stat().st_size / 1e6:.0f} MB)')
        print('upload with:')
        print(f'  databricks fs cp "{output}" "dbfs:/Volumes/<catalog>/<schema>/<volume>/libreoffice/{output.name}"')


if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2])
