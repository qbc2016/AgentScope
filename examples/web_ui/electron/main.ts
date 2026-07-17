import { app, BrowserWindow, ipcMain } from 'electron';
import { spawn, ChildProcess, execSync } from 'child_process';
import * as path from 'path';
import * as http from 'http';
import * as net from 'net';
import * as fs from 'fs';

let mainWindow: BrowserWindow | null = null;
let backendProcess: ChildProcess | null = null;
let backendPort = 0;

function getFreePort(): Promise<number> {
	return new Promise((resolve, reject) => {
		const srv = net.createServer();
		srv.listen(0, '127.0.0.1', () => {
			const addr = srv.address();
			if (addr && typeof addr !== 'string') {
				const port = addr.port;
				srv.close(() => resolve(port));
			} else {
				srv.close(() => reject(new Error('Failed to get port')));
			}
		});
		srv.on('error', reject);
	});
}

function findPython(): string {
	if (process.env.PYTHON_CMD) return process.env.PYTHON_CMD;
	const candidates = ['python3', 'python'];
	for (const cmd of candidates) {
		try {
			execSync(`${cmd} --version`, { stdio: 'ignore' });
			return cmd;
		} catch {
			// try next
		}
	}
	throw new Error('Python not found. Set PYTHON_CMD env var.');
}

function getDesktopMainPath(): string {
	const devPath = path.resolve(__dirname, '../../desktop/main.py');
	const prodPath = path.join(process.resourcesPath || '', 'desktop_main.py');
	if (fs.existsSync(devPath)) return devPath;
	if (fs.existsSync(prodPath)) return prodPath;
	throw new Error(`desktop_main.py not found at ${devPath} or ${prodPath}`);
}

async function startBackend(): Promise<void> {
	backendPort = await getFreePort();
	const pythonCmd = findPython();
	const scriptPath = getDesktopMainPath();
	const cmd = `${pythonCmd} "${scriptPath}" --port ${backendPort}`;

	console.log(`Starting backend: ${cmd}`);

	backendProcess = spawn(cmd, {
		env: { ...process.env, AGENTSCOPE_MODE: 'desktop' },
		stdio: ['ignore', 'pipe', 'pipe'],
		shell: true,
	});

	backendProcess.stdout?.on('data', (data: Buffer) => {
		console.log(`[backend] ${data.toString().trim()}`);
	});

	backendProcess.stderr?.on('data', (data: Buffer) => {
		console.error(`[backend] ${data.toString().trim()}`);
	});

	backendProcess.on('exit', (code) => {
		console.log(`Backend exited with code ${code}`);
		backendProcess = null;
	});

	await waitForBackend(backendPort, 30000);
}

function waitForBackend(port: number, timeoutMs: number): Promise<void> {
	const start = Date.now();
	return new Promise((resolve, reject) => {
		const check = () => {
			if (Date.now() - start > timeoutMs) {
				reject(new Error(`Backend did not start within ${timeoutMs}ms`));
				return;
			}
			http.get(`http://127.0.0.1:${port}/docs`, (res) => {
				if (res.statusCode === 200) {
					resolve();
				} else {
					setTimeout(check, 500);
				}
			}).on('error', () => setTimeout(check, 500));
		};
		check();
	});
}

function createWindow(): void {
	const isDev = !app.isPackaged;

	mainWindow = new BrowserWindow({
		width: 1400,
		height: 900,
		minWidth: 800,
		minHeight: 600,
		title: 'AgentScope',
		webPreferences: {
			preload: path.join(__dirname, 'preload.js'),
			contextIsolation: true,
			nodeIntegration: false,
			webSecurity: isDev,
		},
	});

	mainWindow.webContents.on('render-process-gone', (_event, details) => {
		console.error(`Renderer crashed: ${details.reason}, exitCode=${details.exitCode}`);
	});

	mainWindow.webContents.on('unresponsive', () => {
		console.error('Renderer became unresponsive');
	});

	if (isDev) {
		mainWindow.loadURL('http://localhost:5173');
		mainWindow.webContents.openDevTools();
	} else {
		const indexPath = path.join(__dirname, 'frontend-dist/index.html');
		mainWindow.loadFile(indexPath);
	}

	mainWindow.on('closed', () => {
		mainWindow = null;
	});
}

function killTree(pid: number, signal: NodeJS.Signals): void {
	const shortSig = signal.replace(/^SIG/, '');
	try {
		execSync(`pkill -${shortSig} -P ${pid}`, { stdio: 'ignore' });
	} catch {
		// No children or already gone
	}
	try {
		process.kill(pid, signal);
	} catch {
		// Already gone
	}
}

function stopBackend(): void {
	if (!backendProcess?.pid) return;
	const pid = backendProcess.pid;
	backendProcess = null;

	console.log(`Stopping backend (pid ${pid})...`);
	killTree(pid, 'SIGTERM');

	setTimeout(() => killTree(pid, 'SIGKILL'), 5000);
}

app.whenReady().then(async () => {
	ipcMain.on('get-backend-url', (event) => {
		event.returnValue = `http://127.0.0.1:${backendPort}`;
	});

	try {
		await startBackend();
		createWindow();
	} catch (err) {
		console.error('Failed to start:', err);
		app.quit();
	}
});

app.on('window-all-closed', () => {
	console.log('Event: window-all-closed');
	stopBackend();
	app.quit();
});

app.on('before-quit', () => {
	console.log('Event: before-quit');
	stopBackend();
});
