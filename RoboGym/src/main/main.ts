import { app, shell, BrowserWindow, ipcMain, dialog } from 'electron'
import path, { join } from 'path'
import { electronApp, optimizer, is } from '@electron-toolkit/utils'
import icon from '../../resources/icon.png?asset'
import { spawn } from 'child_process'
import kill from 'tree-kill';
let pyProc;
let pythonReady = false;
function createWindow(): void {

  const splashWindow = new BrowserWindow({
    width: 850,
    height: 500,
    transparent: true,
    frame: false,
    alwaysOnTop: true,
    center: true,
    icon: path.join(__dirname, '../../src/Assets/RogoGymLogo.png'), 
    resizable: false,
    show: true,
  });

  splashWindow.loadFile(path.join(__dirname, '../../src/Assets/splashScreen/splash2.html'));

  const mainWindow = new BrowserWindow({
    width: 900,
    height: 670,
    show: false,
    autoHideMenuBar: true,
    ...(process.platform === 'linux' ? { icon } : {}),
    webPreferences: {
      preload: join(__dirname, '../preload/index.js'),
      sandbox: false
    },
    icon: path.join(__dirname, '../../src/Assets/RogoGymLogo.png'), 
    title:"RoboGym"
  })

  mainWindow.webContents.setWindowOpenHandler((details) => {
    shell.openExternal(details.url)
    return { action: 'deny' }
  })

  // HMR for renderer base on electron-vite cli.
  // Load the remote URL for development or the local html file for production.
  if (is.dev && process.env['ELECTRON_RENDERER_URL']) {
    mainWindow.loadURL(process.env['ELECTRON_RENDERER_URL'])
  } else {
    mainWindow.loadFile(join(__dirname, '../renderer/index.html'))
  }

  console.log("Yarab")
  const pythonFlask = path.join(__dirname,"..","..","..", "robogym_structure", "flask_Apis.py")
  console.log(pythonFlask)
  pyProc = spawn('python', [pythonFlask]);

  pyProc.stdout.on('data', (data) => {
    const output = data.toString();
    console.log(`[Python] ${output}`);
  
    if (!pythonReady && output.includes("Debugger PIN")) {
      pythonReady = true;
      splashWindow.destroy();
      mainWindow.show();
    }
  });
  
  pyProc.stderr.on('data', (data) => {
    const output = data.toString();
    console.error(`[Python Error] ${output}`);
  
    if (!pythonReady && output.includes("Debugger PIN")) {
      pythonReady = true;
      splashWindow.destroy();
      mainWindow.show();
    }
  });
  mainWindow.once('ready-to-show', () => {
    // setTimeout(() => {
    //   splashWindow.destroy();
    //   mainWindow.show();
    // }, 4000); 
  });
}

// This method will be called when Electron has finished
// initialization and is ready to create browser windows.
// Some APIs can only be used after this event occurs.
app.whenReady().then(() => {
  // Set app user model id for windows
  electronApp.setAppUserModelId('com.electron')

  // Default open or close DevTools by F12 in development
  // and ignore CommandOrControl + R in production.
  // see https://github.com/alex8088/electron-toolkit/tree/master/packages/utils
  app.on('browser-window-created', (_, window) => {
    optimizer.watchWindowShortcuts(window)
  })

  // IPC test
  ipcMain.on('ping', () => console.log('pong'))
  
  createWindow()

  
  app.on('activate', function () {
    // On macOS it's common to re-create a window in the app when the
    // dock icon is clicked and there are no other windows open.
    if (BrowserWindow.getAllWindows().length === 0) createWindow()
  })

  // const pythonScriptPath = path.join(__dirname, '..', '..',"..", 'robogym structure', 'flask_api.py');
 

ipcMain.handle('open-file-dialog', async () => {
  const result = await dialog.showOpenDialog({
    properties: ['openFile'],
    
    filters: [{ name: 'ZIP Files', extensions: ['zip'] }],
  });

  if (result.canceled || result.filePaths.length === 0) return null;
  return result.filePaths[0];
});

ipcMain.handle('save-file-dialog', async (event, defaultName: string) => {
  const result = await dialog.showSaveDialog({
    defaultPath: defaultName,
    filters: [{ name: 'ZIP Files', extensions: ['zip'] }],
  });

  if (result.canceled || !result.filePath) return null;
  return result.filePath;
});

  
})

app.on('quit', () => {
  if (pyProc && pyProc.pid) {
    console.log("Killing Python process tree...");
    kill(pyProc.pid, 'SIGTERM');
  }
});
// Quit when all windows are closed, except on macOS. There, it's common
// for applications and their menu bar to stay active until the user quits
// explicitly with Cmd + Q.
app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit()
  }
})

