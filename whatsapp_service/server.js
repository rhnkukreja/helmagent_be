// server.js
import 'dotenv/config';
import express from 'express';
import cors from 'cors';
import axios from 'axios';
import qrcode from 'qrcode';
import makeWASocket, {
  DisconnectReason,
  fetchLatestBaileysVersion,
  makeCacheableSignalKeyStore
} from '@whiskeysockets/baileys';
import { supabase } from './supabaseClient.js';
import { useSupabaseAuthState } from './supabase-auth.js';
import pino from 'pino';
import { Boom } from '@hapi/boom';

const app = express();
app.use(cors());
app.use(express.json());

const PORT = 3001; // always run internally

const BACKEND_URL = process.env.BACKEND_URL || 'http://localhost:8000';

// Active in-memory sessions map
const sessions = new Map();

// Logger
const logger = pino({ level: process.env.LOG_LEVEL || 'info' });

/* ============================================================
   START WHATSAPP SESSION
============================================================ */
async function startWhatsAppSession(sessionId) {
  if (sessions.has(sessionId)) {
    console.log(`[${sessionId}] Session already exists, reusing...`);
    return { success: true, status: sessions.get(sessionId).status };
  }

  const sessionInfo = {
    socket: null,
    qr: null,
    status: 'initializing',
    phone: null,
    retries: 0
  };

  sessions.set(sessionId, sessionInfo);

  try {
    await connectToWhatsApp(sessionId, sessionInfo);
    return { success: true, status: 'initializing' };
  } catch (err) {
    console.error(`[${sessionId}] Error during connect:`, err);
    sessions.delete(sessionId);
    return { error: err.message };
  }
}

/* ============================================================
   CONNECT TO WHATSAPP
============================================================ */
async function connectToWhatsApp(sessionId, sessionInfo) {
  console.log(`\n[${sessionId}] ▶️ Connecting to WhatsApp...`);
  const { state, saveCreds } = await useSupabaseAuthState(sessionId, logger);
  const { version, isLatest } = await fetchLatestBaileysVersion();

  console.log(`[${sessionId}] Using WA version ${version.join('.')}, isLatest=${isLatest}`);

  let isFirstConnection = true;
  let connectionTimestamp = Math.floor(Date.now() / 1000);

  const sock = makeWASocket({
    version,
    logger,
    auth: {
      creds: state.creds,
      keys: makeCacheableSignalKeyStore(state.keys, logger)
    },
    browser: ['Chrome', 'Linux', '110.0.0.0'],
    markOnlineOnConnect: false,
    generateHighQualityLinkPreview: true,
    syncFullHistory: false,
    keepAliveIntervalMs: 30000,
    connectTimeoutMs: 60000,
    emitOwnEvents: true,
    fireInitQueries: true
  });

  sessionInfo.socket = sock;

/* ------------------------------------------------------------
   CONNECTION UPDATES with QR timeout + no infinite retry
------------------------------------------------------------ */
    const MAX_RETRIES = 1;
    const QR_VALIDITY_MS = 60000; // 60 seconds

    sock.ev.on('connection.update', async (update) => {
      const { connection, lastDisconnect, qr } = update;

      // 🔹 Send QR only once and make it expire
      if (qr && !sessionInfo.qr) {
        try {
          const qrImage = await qrcode.toDataURL(qr);
          sessionInfo.qr = qrImage;
          sessionInfo.status = 'qr_ready';
          sessionInfo.qr_expiry = Date.now() + QR_VALIDITY_MS;
          console.log(`[${sessionId}] ✅ QR ready — valid for 60s`);

          await notifyBackend(sessionId, 'qr_ready', {
            qr: qrImage,
            expires_in: QR_VALIDITY_MS / 1000
          });

          // ⏰ Expire QR automatically after 60 seconds
          setTimeout(() => {
            if (Date.now() > sessionInfo.qr_expiry && sessionInfo.status !== 'connected') {
              console.log(`[${sessionId}] ⏱️ QR expired after 60s`);
              sessionInfo.qr = null;
              sessionInfo.status = 'expired';
              notifyBackend(sessionId, 'disconnected', { reason: 'qr_expired' });
            }
          }, QR_VALIDITY_MS);
        } catch (err) {
          console.error(`[${sessionId}] QR generation error:`, err);
        }
      }

      if (connection === 'open') {
        sessionInfo.status = 'connected';
        sessionInfo.qr = null;
        sessionInfo.retries = 0;
        const phoneNumber = sock.user?.id?.split(':')[0] || 'unknown';
        const userName = sock.user?.name || sock.user?.verifiedName || 'User';
        sessionInfo.phone = phoneNumber;
        console.log(`[${sessionId}] ✅ Connected as ${phoneNumber} (${userName})`);
        await notifyBackend(sessionId, 'connected', { phone: phoneNumber, name: userName });
      }

      if (connection === 'close') {
        const statusCode = lastDisconnect?.error?.output?.statusCode;
        const reason = lastDisconnect?.error?.output?.payload?.error || 'unknown';
        console.log(`[${sessionId}] ⚠️ Connection closed. Status=${statusCode}, Reason=${reason}`);

        // ⛔ Remove infinite reconnect loop
        if (sessionInfo.retries >= MAX_RETRIES) {
          console.log(`[${sessionId}] ❌ Max retries reached (${MAX_RETRIES}).`);
          sessions.delete(sessionId);
          await notifyBackend(sessionId, 'disconnected', { reason: 'max_retries' });
        } else {
          sessionInfo.retries++;
          console.log(`[${sessionId}] 🔁 Retry ${sessionInfo.retries}/${MAX_RETRIES}...`);
          setTimeout(() => {
            if (sessions.has(sessionId)) connectToWhatsApp(sessionId, sessionInfo);
          }, 3000);
        }
      }

      if (connection === 'connecting') {
        sessionInfo.status = 'connecting';
        console.log(`[${sessionId}] 🔄 Connecting...`);
      }
    });

  /* ------------------------------------------------------------
     CREDS + KEYS UPDATES
  ------------------------------------------------------------ */
  sock.ev.on('creds.update', async () => {
    console.log(`[${sessionId}] 🔐 creds.update triggered — saving`);
    await saveCreds();
  });

  sock.ev.on('keys.update', async () => {
    console.log(`[${sessionId}] 🔑 keys.update triggered — saving`);
    await saveCreds();
  });

  /* ------------------------------------------------------------
     MESSAGES HANDLING
  ------------------------------------------------------------ */
  sock.ev.on('messages.upsert', async ({ messages, type }) => {
    if (type !== 'notify') return;

    for (const msg of messages) {
      if (!msg.message) continue;
      if (msg.key.remoteJid === 'status@broadcast') continue;
      if (msg.message.protocolMessage || msg.message.reactionMessage) continue;

      const ts = msg.messageTimestamp;
      if (ts < connectionTimestamp - 10) continue;
      if (msg.key.fromMe) continue;

      const messageData = {
        id: msg.key.id,
        from: msg.key.remoteJid,
        fromMe: msg.key.fromMe,
        timestamp: msg.messageTimestamp,
        message: extractMessageContent(msg)
      };

      console.log(`[${sessionId}] 📨 New message from ${messageData.from}`);

      await notifyBackend(sessionId, 'message_received', messageData);
    }
  });

  sock.ev.on('messages.update', async (updates) => {
    for (const u of updates) {
      if (u.key.fromMe && u.update.status) {
        const statusText = {
          0: 'pending',
          1: 'sent',
          2: 'delivered',
          3: 'read',
          4: 'played'
        }[u.update.status] || 'unknown';
        console.log(`[${sessionId}] 📤 Sent msg ${u.key.id?.substring(0, 10)}... status=${statusText}`);
      }
    }
  });
}

/* ============================================================
   EXTRACT MESSAGE CONTENT
============================================================ */
function extractMessageContent(msg) {
  const m = msg.message;
  if (m.conversation) return { type: 'text', text: m.conversation };
  if (m.extendedTextMessage) return { type: 'text', text: m.extendedTextMessage.text };
  if (m.imageMessage) return { type: 'image', caption: m.imageMessage.caption || '' };
  if (m.videoMessage) return { type: 'video', caption: m.videoMessage.caption || '' };
  if (m.documentMessage) return { type: 'document', fileName: m.documentMessage.fileName };
  if (m.audioMessage) return { type: 'audio' };
  if (m.stickerMessage) return { type: 'sticker' };
  if (m.contactMessage) return { type: 'contact', displayName: m.contactMessage.displayName };
  if (m.locationMessage)
    return { type: 'location', latitude: m.locationMessage.degreesLatitude, longitude: m.locationMessage.degreesLongitude };
  return { type: 'unknown' };
}

/* ============================================================
   SEND MESSAGE
============================================================ */
async function sendMessage(sessionId, to, text) {
  const session = sessions.get(sessionId);
  if (!session || !session.socket) throw new Error('Session not found or not connected');
  if (session.status !== 'connected') throw new Error('Session is not connected');

  const jid = to.includes('@') ? to : `${to}@s.whatsapp.net`;

  try {
    const sent = await session.socket.sendMessage(jid, { text });
    console.log(`[${sessionId}] ✅ Message sent to ${to}`);
    return { success: true, id: sent.key.id };
  } catch (err) {
    console.error(`[${sessionId}] ❌ Failed to send message:`, err);
    throw err;
  }
}

/* ============================================================
   NOTIFY BACKEND
============================================================ */
async function notifyBackend(sessionId, event, data) {
  try {
    await axios.post(`${BACKEND_URL}/whatsapp/webhook`, {
      session_id: sessionId,
      event,
      data
    });
  } catch (err) {
    console.error(`[${sessionId}] ⚠️ Failed to notify backend:`, err.message);
  }
}

/* ============================================================
   EXPRESS ROUTES
============================================================ */
app.post('/session/start', async (req, res) => {
  const { session_id } = req.body;
  if (!session_id) return res.status(400).json({ error: 'session_id required' });
  console.log(`\n🚀 Starting session: ${session_id}`);
  const result = await startWhatsAppSession(session_id);
  res.json(result);
});

app.get('/session/:sessionId/status', (req, res) => {
  const { sessionId } = req.params;
  const s = sessions.get(sessionId);
  if (!s) return res.json({ status: 'not_found' });
  res.json({ status: s.status, qr: s.qr, phone: s.phone });
});

app.post('/session/:sessionId/send', async (req, res) => {
  const { sessionId } = req.params;
  const { to, text } = req.body;
  if (!to || !text) return res.status(400).json({ error: 'to and text required' });
  try {
    const result = await sendMessage(sessionId, to, text);
    res.json(result);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

app.post('/session/:sessionId/disconnect', async (req, res) => {
  const { sessionId } = req.params;
  const s = sessions.get(sessionId);
  console.log(`🔌 Disconnecting session ${sessionId}`);

  if (s?.socket) {
    try {
      await s.socket.logout();
    } catch (err) {
      console.error(`[${sessionId}] Logout error:`, err);
    }
  }

  sessions.delete(sessionId);

  try {
    await supabase.from('whatsapp_sessions')
      .update({ status: 'disconnected', updated_at: new Date().toISOString() })
      .eq('id', sessionId);
  } catch (err) {
    console.error(`[${sessionId}] ❌ Failed to update Supabase:`, err.message);
  }

  res.json({ success: true, message: 'Session disconnected' });
});

app.get('/sessions', (req, res) => {
  const active = Array.from(sessions.entries()).map(([id, info]) => ({
    session_id: id,
    status: info.status,
    phone: info.phone,
    has_qr: !!info.qr
  }));
  res.json({ sessions: active });
});

app.get('/health', (req, res) => {
  res.json({ status: 'ok', sessions: sessions.size });
});

/* ============================================================
   RESTORE SAVED SESSIONS ON STARTUP
============================================================ */
(async () => {
  console.log('♻️ Restoring saved sessions from Supabase...');
  try {
    const { data, error } = await supabase
      .from('whatsapp_sessions')
      .select('id, status')
      .eq('status', 'connected');

    if (error) throw error;

    for (const s of data) {
      console.log(`🔄 Restoring session ${s.id}...`);
      await startWhatsAppSession(s.id);
    }

    console.log(`✅ Restored ${data.length} session(s) from Supabase`);
  } catch (err) {
    console.error('❌ Failed to restore sessions:', err.message);
  }

  app.listen(PORT, '0.0.0.0', () => {
    console.log(`WhatsApp Service LIVE on 0.0.0.0:${PORT}`);
    console.log(`Backend webhook URL: ${BACKEND_URL}/whatsapp/webhook`);
    console.log(`Total active sessions: ${sessions.size}`);
  });
})();

/* ============================================================
   GRACEFUL SHUTDOWN
============================================================ */
process.on('SIGINT', async () => {
  console.log('\n👋 Shutting down gracefully...');
  for (const [id, s] of sessions.entries()) {
    try {
      if (s.socket) await s.socket.logout();
    } catch (err) {
      console.error(`Error closing session ${id}:`, err);
    }
  }
  process.exit(0);
});