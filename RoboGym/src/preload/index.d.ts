import { ElectronAPI } from '@electron-toolkit/preload'

declare global {
  interface Window {
    electron: ElectronAPI
    api: unknown
    electronAPI: {
      initiateWebSocketConnection: (url: string) => void
      startPyBullet: () => void
      openFileDialog: () => Promise<string | null>
      saveFileDialog: (defaultName: string) => Promise<string | null>
    }
  }
}

export {}
