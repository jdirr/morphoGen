#!/bin/sh
# The macOS package of the JupyterLab environment of morphoGen: the folder that
# build_environment.py makes (a Python of its own, morphoGen, JupyterLab) inside an
# application, "morphoGen JupyterLab.app", that opens JupyterLab on the user's notebooks in a
# window of Terminal; the package (.pkg) installs it in the Applications folder of the user
# ("Install for me only", no administrator rights) or of all the users. On macOS, from the
# root of the repository, once the environment is built:
#
#     sh packaging/jupyterlab/build_pkg.sh build/jupyterlab dist/morphoGen-JupyterLab-1.0.2-macos-arm64.pkg
#
# The package is for the architecture of the Mac that builds it (arm64 or x86_64), as the
# environment is. See packaging/jupyterlab/README.md.
set -eu

environment=$(cd "$1" && pwd)
package=$2
version=$(cat "$environment/VERSION")
arch=$(uname -m)
here=$(cd "$(dirname "$0")" && pwd)
root=$(cd "$here/../.." && pwd)
identifier=io.github.jdirr.morphogen.jupyterlab
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT

app="$work/root/morphoGen JupyterLab.app"
mkdir -p "$app/Contents/MacOS" "$app/Contents/Resources"
cp -R "$environment/" "$app/Contents/Resources/"

# the icon, from that of the package
iconset="$work/morphogen.iconset"
mkdir "$iconset"
for size in 16 32 128 256 512; do
    sips -z "$size" "$size" "$root/src/morphogen/gui/icon.png" --out "$iconset/icon_${size}x${size}.png" >/dev/null
    double=$((2 * size))
    sips -z "$double" "$double" "$root/src/morphogen/gui/icon.png" --out "$iconset/icon_${size}x${size}@2x.png" >/dev/null
done
iconutil -c icns "$iconset" -o "$app/Contents/Resources/morphogen.icns"

# the application opens a window of Terminal that runs the launcher: closing it stops JupyterLab
cat > "$app/Contents/Resources/start.command" <<'EOF'
#!/bin/sh
here=$(cd "$(dirname "$0")" && pwd)
exec "$here/python/bin/python3" "$here/launcher.py"
EOF
cat > "$app/Contents/MacOS/launcher" <<'EOF'
#!/bin/sh
here=$(cd "$(dirname "$0")/../Resources" && pwd)
exec open -a Terminal "$here/start.command"
EOF
chmod 755 "$app/Contents/Resources/start.command" "$app/Contents/MacOS/launcher"
cat > "$app/Contents/Info.plist" <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>CFBundleName</key><string>morphoGen JupyterLab</string>
    <key>CFBundleDisplayName</key><string>morphoGen JupyterLab</string>
    <key>CFBundleIdentifier</key><string>$identifier</string>
    <key>CFBundleVersion</key><string>$version</string>
    <key>CFBundleShortVersionString</key><string>$version</string>
    <key>CFBundlePackageType</key><string>APPL</string>
    <key>CFBundleExecutable</key><string>launcher</string>
    <key>CFBundleIconFile</key><string>morphogen.icns</string>
    <key>LSMinimumSystemVersion</key><string>11.0</string>
    <key>NSHumanReadableCopyright</key><string>Copyright (C) 2026 Justin Dirrenberger, GPL-3.0-or-later</string>
</dict>
</plist>
EOF
plutil -lint "$app/Contents/Info.plist" >/dev/null

# the component: installed where it is asked (not moved to another copy of the application)
pkgbuild --analyze --root "$work/root" "$work/component.plist" >/dev/null
plutil -replace 0.BundleIsRelocatable -bool NO "$work/component.plist"
pkgbuild --root "$work/root" --component-plist "$work/component.plist" --identifier "$identifier" \
    --version "$version" --install-location /Applications "$work/component.pkg" >/dev/null

# the product: for the user's Applications folder or for all the users, on this architecture
cat > "$work/distribution.xml" <<EOF
<?xml version="1.0" encoding="utf-8"?>
<installer-gui-script minSpecVersion="2">
    <title>morphoGen JupyterLab $version</title>
    <domains enable_anywhere="false" enable_currentUserHome="true" enable_localSystem="true"/>
    <options customize="never" require-scripts="false" hostArchitectures="$arch"/>
    <volume-check><allowed-os-versions><os-version min="11.0"/></allowed-os-versions></volume-check>
    <choices-outline><line choice="default"/></choices-outline>
    <choice id="default" title="morphoGen JupyterLab"><pkg-ref id="$identifier"/></choice>
    <pkg-ref id="$identifier" version="$version" onConclusion="none">component.pkg</pkg-ref>
</installer-gui-script>
EOF
mkdir -p "$(dirname "$package")"
productbuild --distribution "$work/distribution.xml" --package-path "$work" "$package" >/dev/null
echo "$package: morphoGen JupyterLab $version for macOS $arch"
