please carefully review the error, then think deeply and thoroughly using extremely long chain of thought to carefully and systematically for the best implementation option to fix the error, then carefully test and validate your code changes using extensive web searches before generating complete updated replacement file for each file that needs modification. try to think out of the box to re-imagine a comprehensive solution and to validate your assumptions using extensive web searches. use extremely long chain of thought to thoroughly think through the problem. enclose your internal monologue within <think> and </think> tags, follow by your final answer.
give me a complete validated correct replacement file for each.

$ npm start

> context7-mcp-server@1.0.0 start
> node mcp_server.js

node:internal/modules/cjs/loader:664
      throw e;
      ^

Error: Cannot find module '/cdrom/project/Context7-Explorer/Kimi/node_modules/@modelcontextprotocol/sdk/dist/cjs/dist/cjs/types.js'
    at createEsmNotFoundErr (node:internal/modules/cjs/loader:1437:15)
    at finalizeEsmResolution (node:internal/modules/cjs/loader:1426:15)
    at resolveExports (node:internal/modules/cjs/loader:657:14)
    at Function._findPath (node:internal/modules/cjs/loader:749:31)
    at Function._resolveFilename (node:internal/modules/cjs/loader:1387:27)
    at defaultResolveImpl (node:internal/modules/cjs/loader:1057:19)
    at resolveForCJSWithHooks (node:internal/modules/cjs/loader:1062:22)
    at Function._load (node:internal/modules/cjs/loader:1211:37)
    at TracingChannel.traceSync (node:diagnostics_channel:322:14)
    at wrapModuleLoad (node:internal/modules/cjs/loader:235:24) {
  code: 'MODULE_NOT_FOUND',
  path: '/cdrom/project/Context7-Explorer/Kimi/node_modules/@modelcontextprotocol/sdk'
}

Node.js v22.16.0

---
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

// Pull in the SDK’s full CJS bundle, which includes createServer
const { createServer } = require("@modelcontextprotocol/sdk/dist/cjs/types.js");

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

$ diff -u mcp_server.js-original mcp_server.js
```diff
--- mcp_server.js-original   2025-07-15 08:42:47.246321223 +0800
+++ mcp_server.js       2025-07-15 11:35:14.117055340 +0800
@@ -1,24 +1,28 @@
+#!/usr/bin/env node
+"use strict";
+
 /*
- * Context7 MCP server – Node 18+
- * npm install @modelcontextprotocol/sdk ws express
- * node mcp_server.js
+ * Context7 MCP server – Node 18+ (CommonJS)
+ * 1) npm install
+ * 2) npm start
  *
- * Exposes:
+ * Exposes via WebSocket:
  *   tools.search_docs(query:string, limit?:int) -> [{title,url,snippet}]
  *   tools.bookmark_doc(url:string, note?:string) -> {ok}
  *   tools.list_bookmarks() -> [{url,title,note}]
- *
- * In real life you would hit a vector DB; here we fake with a tiny in-mem corpus.
  */
-import { createServer } from "@modelcontextprotocol/sdk/node/server.js";
-import express from "express";
-import { WebSocketServer } from "ws";
+
+// Pull in the SDK’s full CJS bundle, which includes createServer
+const { createServer } = require("@modelcontextprotocol/sdk/dist/cjs/types.js");
+
+const express = require("express");
+const { WebSocketServer } = require("ws");
 
 const PORT = 8765;
 const app = express();
 const wss = new WebSocketServer({ port: PORT + 1 });
 
-// --- Fake document store ------------------------------------------------------
+// --- In-memory document store and bookmarks --------------------------------
 const docs = [
   {
     title: "Quantum computing – Wikipedia",
@@ -39,15 +43,11 @@
       "Shor’s algorithm efficiently factors integers and threatens RSA cryptography."
   }
 ];
-
 let bookmarks = [];
 
-// --- MCP server ---------------------------------------------------------------
+// --- Instantiate the MCP server ---------------------------------------------
 const server = createServer(
-  {
-    name: "context7",
-    version: "1.0.0"
-  },
+  { name: "context7", version: "1.0.0" },
   {
     tools: {
       search_docs: {
@@ -61,14 +61,13 @@
           required: ["query"]
         },
         handler: async ({ query, limit = 5 }) => {
-          const hits = docs
-            .filter((d) =>
+          return docs
+            .filter(d =>
               d.title.toLowerCase().includes(query.toLowerCase()) ||
               d.snippet.toLowerCase().includes(query.toLowerCase())
             )
             .slice(0, limit)
             .map(({ title, url, snippet }) => ({ title, url, snippet }));
-          return hits;
         }
       },
       bookmark_doc: {
@@ -95,13 +94,14 @@
   }
 );
 
-// Bridge MCP ↔ WebSocket so Python can talk to it
-wss.on("connection", (ws) => {
-  ws.on("message", async (msg) => {
-    const req = JSON.parse(msg);
-    const result = await server.callTool(req);
-    ws.send(JSON.stringify(result));
+// --- Bridge MCP ↔ WebSocket -------------------------------------------------
+wss.on("connection", ws => {
+  ws.on("message", async msg => {
+    const req = JSON.parse(msg.toString());
+    const res = await server.callTool(req);
+    ws.send(JSON.stringify(res));
   });
 });
 
 console.log(`Context7 MCP server listening on ws://localhost:${PORT + 1}`);
+
```

