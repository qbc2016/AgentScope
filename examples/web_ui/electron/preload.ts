import { contextBridge, ipcRenderer } from 'electron';

contextBridge.exposeInMainWorld('electronAPI', {
	getBackendUrl: (): string => ipcRenderer.sendSync('get-backend-url'),
	getUserId: (): string => 'local-user',
});
