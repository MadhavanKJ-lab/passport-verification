// Server-side proxy to the n8n webhook. The webhook key stays in Vercel environment
// variables and is never sent to the browser.
//   N8N_WEBHOOK_URL   full webhook URL
//   N8N_WEBHOOK_KEY   value for the X-Webhook-Key header
module.exports = async (req, res) => {
  if (req.method !== 'POST') return res.status(405).json({ error: 'POST only' });
  const url = process.env.N8N_WEBHOOK_URL, key = process.env.N8N_WEBHOOK_KEY;
  if (!url || !key) return res.status(500).json({ error: 'Server is not configured' });
  try {
    const upstream = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'X-Webhook-Key': key },
      body: JSON.stringify(req.body || {}),
    });
    const text = await upstream.text();
    res.status(upstream.status).setHeader('Content-Type', 'application/json').send(text);
  } catch (e) {
    res.status(502).json({ error: 'Could not reach the verification service' });
  }
};
