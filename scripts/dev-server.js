#!/usr/bin/env node

const fs = require('fs').promises;
const path = require('path');
const http = require('http');
const { build } = require('./build.js');
const WebSocket = require('ws');

let server = null;
let wss = null;

async function startDevServer() {
  console.log('🛑 Starting development server...');
  
  await build(true);
  
  // Create HTTP server
  server = http.createServer(async (req, res) => {
    // Handle CORS preflight requests
    if (req.method === 'OPTIONS') {
      res.setHeader('Access-Control-Allow-Origin', '*');
      res.setHeader('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
      res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');
      res.writeHead(200);
      res.end();
      return;
    }
    
    // Parse URL to handle query parameters (needed for speaker notes)
    const url = new URL(req.url, `http://${req.headers.host}`);
    let requestPath = url.pathname;
    
    let filePath = path.join(__dirname, '../dist', requestPath === '/' ? 'index.html' : requestPath);
    
    try {
      const stats = await fs.stat(filePath);
      if (stats.isDirectory()) {
        filePath = path.join(filePath, 'index.html');
      }
      
      const data = await fs.readFile(filePath);
      const ext = path.extname(filePath);
      
      let contentType = 'text/html';
      if (ext === '.js') contentType = 'application/javascript';
      else if (ext === '.css') contentType = 'text/css';
      else if (ext === '.json') contentType = 'application/json';
      else if (ext === '.png') contentType = 'image/png';
      else if (ext === '.jpg' || ext === '.jpeg') contentType = 'image/jpeg';
      else if (ext === '.gif') contentType = 'image/gif';
      else if (ext === '.svg') contentType = 'image/svg+xml';
      
      res.setHeader('Content-Type', contentType);
      res.setHeader('Cache-Control', 'no-cache');
      
      // Add CORS headers for speaker notes to work
      res.setHeader('Access-Control-Allow-Origin', '*');
      res.setHeader('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS');
      res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');
      
      res.writeHead(200);
      res.end(data);
    } catch (err) {
      res.writeHead(404);
      res.end('Not found');
    }
  });
  
  // Create WebSocket server attached to HTTP server
  wss = new WebSocket.Server({ server });
  
  const PORT = 8080;
  server.listen(PORT, () => {
    console.log(`✅ Development server started at http://localhost:${PORT}`);
    console.log('🔌 Hot reload WebSocket server running on same port');
    console.log('⚡ Watching for changes...');
  });
}

async function watchFiles() {
  const watchDirs = [
    path.join(__dirname, '../slides/src/css'),
    path.join(__dirname, '../slides/content'),
    path.join(__dirname, '../slides/src/js'),
    path.join(__dirname, '../slides/templates'),
    path.join(__dirname, '../slides/assets')
  ];
  
  for (const dir of watchDirs) {
    try {
      const watcher = require('fs').watch(dir, { recursive: true }, async (eventType, filename) => {
        if (eventType === 'change') {
          console.log(`📝 File changed: ${filename}`);
          console.log('🔄 Rebuilding...');
          await build(true);
          console.log('✅ Rebuild completed!');

          // Notify all connected browsers to reload
          if (wss) {
            wss.clients.forEach(client => {
              if (client.readyState === WebSocket.OPEN) {
                client.send('reload');
              }
            });
          }
        }
      });
    } catch (error) {
      console.log(`⚠️  Could not watch directory: ${dir}`);
    }
  }
}

async function startDevelopment() {
  try {
    await startDevServer();
    await watchFiles();
  } catch (error) {
    console.error('❌ Development server failed:', error.message);
    process.exit(1);
  }
}

process.on('SIGINT', () => {
  console.log('\n🛑 Shutting down development server...');
  if (server) {
    server.close();
  }
  if (wss) {
    wss.close();
  }
  process.exit(0);
});

if (require.main === module) {
  startDevelopment();
}