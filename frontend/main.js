const { app, BrowserWindow, ipcMain, dialog } = require("electron");
const fs = require("fs");
const path = require("path");

let mainWindow;

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1000,
    height: 700,
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: false, // ✅ required for ipcRenderer
      nodeIntegration: true,   // ✅ allows require("electron") in renderer.js
    },
  });

  mainWindow.loadFile("index.html");
}

app.whenReady().then(createWindow);
ipcMain.on("export-image", async (event, imageUrl) => {
  const { filePath } = await dialog.showSaveDialog(mainWindow, {
    title: "Save Processed Image",
    defaultPath: "processed.png",
    filters: [
      { name: "Images", extensions: ["png", "jpg", "jpeg"] },
    ],
  });

  if (filePath) {
    try {
      // imageUrl is now base64 (data:image/png;base64,...)
      const data = imageUrl.split(",")[1];
      const buffer = Buffer.from(data, "base64");
      fs.writeFileSync(filePath, buffer);
      console.log("✅ Image saved to", filePath);
    } catch (err) {
      console.error("❌ Save failed:", err);
    }
  }
});

