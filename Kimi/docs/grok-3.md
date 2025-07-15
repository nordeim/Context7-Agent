# File: package.json
```json
{
  "name": "context7-mcp-server",
  "version": "1.0.0",
  "description": "Context7 MCP server exposing search_docs, bookmark_doc & list_bookmarks over WebSocket",
  "private": true,
  "main": "mcp_server.js",
  "scripts": {
    "start": "node mcp_server.js"
  },
  "engines": {
    "node": ">=18"
  },
  "dependencies": {
    "@modelcontextprotocol/sdk": "^1.15.1",
    "express": "^4.18.2",
    "ws": "^8.18.3"
  },
  "license": "MIT"
}
```

# File: mcp_server.js
```javascript
#!/usr/bin/env node
"use strict";

/*
 * Context7 MCP server – Node 18+ (CommonJS)
 * 1) npm install
 * 2) npm start
 *
 * Exposes via WebSocket:
 *   tools.search_docs(query:string, limit?:int) -> [{title,url,snippet}]
 *   tools.bookmark_doc(url:string, note?:string) -> {ok}
 *   tools.list_bookmarks() -> [{url,title,note}]
 */

// Pull in the SDK’s createServer via Node-specific CJS subpath
const { createServer } = require("@modelcontextprotocol/sdk/node/server.js");

const express = require("express");
const { WebSocketServer } = require("ws");

const PORT = 8765;
const app = express();
const wss = new WebSocketServer({ port: PORT + 1 });

// --- In-memory document store and bookmarks --------------------------------
const docs = [
  {
    title: "Quantum computing – Wikipedia",
    url: "https://en.wikipedia.org/wiki/Quantum_computing",
    snippet:
      "Quantum computing is a type of computation that harnesses quantum mechanical phenomena..."
  },
  {
    title: "Surface code explained",
    url: "https://example.com/surface-code",
    snippet:
      "The surface code is a quantum error-correcting code that promises scalable..."
  },
  {
    title: "Shor’s algorithm",
    url: "https://example.com/shor",
    snippet:
      "Shor’s algorithm efficiently factors integers and threatens RSA cryptography."
  }
];
let bookmarks = [];

// --- Instantiate the MCP server ---------------------------------------------
const server = createServer(
  { name: "context7", version: "1.0.0" },
  {
    tools: {
      search_docs: {
        description: "Search documents by free-text query.",
        inputSchema: {
          type: "object",
          properties: {
            query: { type: "string" },
            limit: { type: "integer", default: 5 }
          },
          required: ["query"]
        },
        handler: async ({ query, limit = 5 }) => {
          return docs
            .filter(d =>
              d.title.toLowerCase().includes(query.toLowerCase()) ||
              d.snippet.toLowerCase().includes(query.toLowerCase())
            )
            .slice(0, limit)
            .map(({ title, url, snippet }) => ({ title, url, snippet }));
        }
      },
      bookmark_doc: {
        description: "Bookmark a document URL with an optional note.",
        inputSchema: {
          type: "object",
          properties: {
            url: { type: "string" },
            note: { type: "string" }
          },
          required: ["url"]
        },
        handler: async ({ url, note }) => {
          bookmarks.push({ url, note: note || "" });
          return { ok: true };
        }
      },
      list_bookmarks: {
        description: "List all bookmarks.",
        inputSchema: {},
        handler: async () => bookmarks
      }
    }
  }
);

// --- Bridge MCP ↔ WebSocket -------------------------------------------------
wss.on("connection", ws => {
  ws.on("message", async msg => {
    const req = JSON.parse(msg.toString());
    const res = await server.callTool(req);
    ws.send(JSON.stringify(res));
  });
});

console.log(`Context7 MCP server listening on ws://localhost:${PORT + 1}`);
```

