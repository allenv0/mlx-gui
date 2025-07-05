#!/bin/bash

# MLX-GUI DMG Creator
# Creates a distributable DMG installer for MLX-GUI

set -e

echo "💿 Creating MLX-GUI DMG installer..."

# Check if app bundle exists
if [ ! -d "dist/MLX-GUI.app" ]; then
    echo "❌ Error: MLX-GUI.app not found. Run ./build_app.sh first."
    exit 1
fi

# Variables
APP_NAME="MLX-GUI"
DMG_NAME="${APP_NAME}-0.1.0"
TEMP_DMG="temp_${DMG_NAME}.dmg"
FINAL_DMG="${DMG_NAME}.dmg"
VOLUME_NAME="MLX-GUI Installer"
SOURCE_DIR="dist"
MOUNT_POINT="/Volumes/${VOLUME_NAME}"

# Clean up any existing DMG
echo "🧹 Cleaning up previous DMGs..."
rm -f "${TEMP_DMG}" "${FINAL_DMG}"

# Create temporary DMG
echo "📦 Creating temporary DMG..."
hdiutil create -srcfolder "${SOURCE_DIR}" -fs HFS+ -volname "${VOLUME_NAME}" "${TEMP_DMG}"

# Mount the DMG
echo "🔗 Mounting DMG..."
hdiutil attach "${TEMP_DMG}" -noautoopen -quiet

# Wait for mount
sleep 2

# Create Applications alias
echo "🔗 Creating Applications shortcut..."
ln -sf /Applications "${MOUNT_POINT}/Applications"

# Set DMG window properties (optional styling)
echo "🎨 Setting DMG window properties..."
osascript << EOD
tell application "Finder"
    tell disk "${VOLUME_NAME}"
        open
        set current view of container window to icon view
        set toolbar visible of container window to false
        set statusbar visible of container window to false
        set the bounds of container window to {400, 100, 900, 400}
        set viewOptions to the icon view options of container window
        set arrangement of viewOptions to not arranged
        set icon size of viewOptions to 72
        set background picture of viewOptions to file ".background:background.png"
        make new alias file at container window to POSIX file "/Applications" with properties {name:"Applications"}
        set position of item "${APP_NAME}.app" of container window to {150, 200}
        set position of item "Applications" of container window to {350, 200}
        close
        open
        update without registering applications
        delay 2
    end tell
end tell
EOD

# Unmount
echo "📤 Unmounting DMG..."
hdiutil detach "${MOUNT_POINT}" -quiet

# Convert to compressed, read-only DMG
echo "🗜️  Converting to final DMG..."
hdiutil convert "${TEMP_DMG}" -format UDZO -o "${FINAL_DMG}"

# Clean up
rm -f "${TEMP_DMG}"

# Check result
if [ -f "${FINAL_DMG}" ]; then
    echo "✅ DMG created successfully!"
    echo "📍 Location: ${FINAL_DMG}"
    echo "📊 Size: $(du -sh "${FINAL_DMG}" | cut -f1)"
    echo ""
    echo "🎉 Your DMG installer is ready!"
    echo "   • Users can drag MLX-GUI.app to Applications"
    echo "   • Double-click to mount and install"
    echo "   • Ready for distribution"
else
    echo "❌ DMG creation failed!"
    exit 1
fi 