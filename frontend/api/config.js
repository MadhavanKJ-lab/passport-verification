// Vercel limits request bodies to 4.5 MB; the upload is sent as base64 (+33%).
module.exports = (req, res) => {
  res.setHeader('Cache-Control', 'no-store');
  res.status(200).json({ maxBytes: 3 * 1024 * 1024 });
};
