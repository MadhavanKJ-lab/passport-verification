// Tells the page which n8n webhook to call. Set N8N_WEBHOOK_URL in the Vercel project
// settings; it is kept out of the repository.
module.exports = (req, res) => {
  res.setHeader('Cache-Control', 'no-store');
  res.status(200).json({ webhookUrl: process.env.N8N_WEBHOOK_URL || null });
};
